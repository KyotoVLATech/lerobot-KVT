#! /usr/bin/env python
"""Online RL fine-tuning of a pi0.5 policy on the real robot."""
import logging
import os
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import etils.epath as epath
import jax
import numpy as np
import openpi.training.sharding as openpi_sharding
import tqdm
import wandb
from absl import app, flags
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.agents.vla.pi05 import build_pi05
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.data.replay_buffer import _critic_key_to_storage, create_replay_buffer
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.loop_utils import AsyncChunkSampler, EpisodeSink
from expo_ft.utils.timer import Timer
from expo_ft.utils.train_utils import (
    get_batch_info,
    init_logging,
    init_wandb,
    set_compilation_cache_dir,
)
from ml_collections import config_flags

warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_enum("update_type", "episode", ["episode", "step", "batch"], "When to run gradient updates: per episode, per step, or per batch of episodes.")
flags.DEFINE_integer("num_batch", 1, "Number of episodes per update batch (only used when update_type=batch).")
flags.DEFINE_integer("num_updates", 0, "Upstream-style fixed number of gradient updates per trigger (episode/step/batch). 0 = derive the count from --step_interval instead.")
flags.DEFINE_integer("step_interval", 1, "One gradient update per this many collected transitions. update_type=step runs it every step_interval env steps; episode/batch runs the accumulated equivalent at the episode boundary (remainder carries over).")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 500_000, "Number of training steps.")
flags.DEFINE_integer("replay_capacity", 0, "Replay capacity; 0 uses max_steps.")
flags.DEFINE_integer("replay_prefetch", 3, "Number of replay batches to prefetch.")
flags.DEFINE_string("replay_storage_dir", None, "Store replay images in temporary disk-backed arrays here.")
flags.DEFINE_boolean("update_batch_prefetch", True, "Prepare the next update batch during the current update.")
flags.DEFINE_boolean("host_update_batches", False, "Keep UTD batches on CPU; stream critic minibatches to GPU (RealTimeEXPOFTLearner only).")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save agent checkpoint every N steps. When 0 and checkpoint_model=True, no interval saving (save at end only).")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_string("client_host", "0.0.0.0", "Bind host to listen on; the rollout client dials in.")
flags.DEFINE_integer("client_port", 8102, "Bind port to listen on.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")
flags.DEFINE_integer(
    "delay", 0,
    "Simulated main-actor inference latency (env-steps) for RealTimeEXPOFTLearner. "
    "0 <= delay <= replan_steps. >0 enables the delayed + prefix-inpainted "
    "rollout and the matching delayed critic backup. Ignored by other learners.",
)
flags.DEFINE_float("sim_latency", 0.0, "Simulated extra inference latency in ms added to each sample_actions call; 0 disables.")

flags.DEFINE_string("dataset_path", "", "Path to the dataset.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)


def main(_):
    init_logging()
    if FLAGS.replay_capacity < 0 or FLAGS.replay_prefetch < 1:
        raise ValueError("replay_capacity must be >= 0 and replay_prefetch must be >= 1")
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}"
        )
    set_compilation_cache_dir(f"sync-{FLAGS.config.model_cls}")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    # iLoHa demos use the same traj.hdf5 layout (scripts/iloha/export_lerobot_to_hdf5.py).
    if FLAGS.config_task.env_type in ("droid", "iloha"):
        dataset = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
            streaming=True,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    if not dataset:
        raise ValueError(f"No demos loaded from {FLAGS.dataset_path}.")
    example_action = dataset[0]['actions'][np.newaxis]

    model_cls = FLAGS.config.model_cls
    if FLAGS.host_update_batches and model_cls != "RealTimeEXPOFTLearner":
        raise ValueError("--host_update_batches requires RealTimeEXPOFTLearner")
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "RealTimeEXPOFTLearner":
        from expo_ft.agents.alg.realtime_expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # Same string run_client returns as task_description, without needing the env yet.
    task_description = FLAGS.config_task.language_instruction

    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )

    critic_camera_keys = tuple(getattr(FLAGS.config_task, "critic_camera_keys", CRITIC_CAMERA_KEYS))
    rb_args = {
        "config": FLAGS.config,
        "example_action": example_action,
        "capacity": FLAGS.replay_capacity or FLAGS.max_steps,
        "task_description": task_description,
        "replan_steps": FLAGS.replan_steps,
        "seed": FLAGS.seed,
        "delay": FLAGS.delay,
        "critic_camera_keys": critic_camera_keys,
        "storage_dir": FLAGS.replay_storage_dir,
    }
    replay_buffer = create_replay_buffer(**rb_args)
    # offline_ratio=0 seeds demos into replay_buffer; no separate offline pool is
    # sampled. Share its storage rather than allocating another full-size pool.
    offline_replay_buffer = (
        create_replay_buffer(**rb_args) if FLAGS.offline_ratio > 0 else replay_buffer
    )

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
        replay_prefetch=FLAGS.replay_prefetch,
        host_batches=FLAGS.host_update_batches,
    )
    # Demos are now in replay storage. Release their original HDF5 arrays without
    # removing any examples from the sampling pool.
    del dataset

    critic_example = {
        _critic_key_to_storage(k): offline_replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
        for k in critic_camera_keys
    }
    critic_example["state"] = offline_replay_buffer.dataset_dict['state'][0][np.newaxis]
    critic_example["actions"] = offline_replay_buffer.dataset_dict['actions'][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
        critic_example)
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    if model_cls == "RealTimeEXPOFTLearner":
        agent_kwargs["delay"] = FLAGS.delay
    agent_kwargs["critic_camera_keys"] = critic_camera_keys

    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=FLAGS.config_task.edit_action_xyzg,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        if hasattr(agent, "cache_infer_params"):
            agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    if FLAGS.host_update_batches:
        agent = agent.offload_target_actor()
    # load_agent stores initialization arrays in agent_kwargs too. Release these
    # GPU references after moving the live target policy to host memory.
    del actor_train_state, target_actor_params, agent_kwargs

    # The env is created only now: EnvClientWrapper blocks until the rollout client
    # dials in, and the agent build above is the slow part.
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }
    logging.info("Creating environment...")
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port
    )
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    episode_log = EpisodeState(ep_len=0)
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    if resuming:
        logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    # Guards the replay buffer: the prefetch thread samples while the loop inserts.
    buffer_lock = threading.Lock()

    batch_processor.on_episode_start()
    timer = Timer()
    # Runs main-actor inference in the background while the robot executes the last `delay` actions.
    sampler = AsyncChunkSampler(
        agent, delay=FLAGS.delay, replan_steps=FLAGS.replan_steps,
        sim_latency_ms=FLAGS.sim_latency, timer=timer,
        executed_prefix=FLAGS.config_task.env_type == "iloha",
    )
    # Buffer inserts, NFS buffer saves, wandb logs and checkpoints queue here and flush at episode end.
    sink = EpisodeSink(
        batch_processor, buffer_lock, checkpoint_manager, checkpoint_dir_path, save_checkpoint,
        save_buffer=FLAGS.checkpoint_buffer, checkpoint_model=FLAGS.checkpoint_model,
        checkpoint_interval=FLAGS.checkpoint_interval, start_step=start_step,
    )
    # env.reset() runs in the background so the episode-end updates overlap the robot homing.
    reset_executor = ThreadPoolExecutor(max_workers=1)
    # One-ahead batch prefetch: the CPU-bound replay image gather overlaps the GPU update.
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if FLAGS.update_batch_prefetch else None
    pending_batch = None

    def _fetch(rng):
        with buffer_lock:
            return batch_processor.next_batch(rng)

    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng, pending_batch
        sink.flush_transitions()
        for _ in tqdm.tqdm(range(num_updates), disable=not FLAGS.tqdm):
            update_start = time.time()
            if prefetch_pool is None:
                batch, actor_batch, combine_rng = _fetch(combine_rng)
            else:
                if pending_batch is None:
                    pending_batch = prefetch_pool.submit(_fetch, combine_rng)
                batch, actor_batch, combine_rng = pending_batch.result()
                # Kick off the next gather now so it overlaps this update.
                pending_batch = prefetch_pool.submit(_fetch, combine_rng)
            metrics["batch_info"] = get_batch_info(batch)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v
            # Do not retain the previous batch while allocating the next one.
            del batch, actor_batch

    def updates_due():
        """Upstream --num_updates: a fixed count per trigger; else one per step_interval transitions."""
        nonlocal transitions_since_update
        if FLAGS.num_updates > 0:
            return FLAGS.num_updates
        n_updates, transitions_since_update = divmod(transitions_since_update, FLAGS.step_interval)
        return n_updates

    dt = 1.0 / FLAGS.config_task.control_hz
    done = False
    last_control_start = time.time()
    # Iloha holds its previous absolute target already. Do not start its episode
    # deadline with a dummy action before the first inference has compiled.
    if FLAGS.config_task.env_type != "iloha":
        env.step(FLAGS.config_task.example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    episodes_since_update = 0
    transitions_since_update = 0
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
    first_step_of_episode = True

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        step_metrics = {}
        timer.reset()
        timer.tick("loop_time")
        timer.tick("total")

        with timer.context("obs"):
            observation = env.get_observation()

        # Observation history for the replan_steps < delay fallback (no-op otherwise).
        sampler.observe(observation)
        with timer.context("info"):
            done, success, reward, mask = env.get_info_for_step()
        sink.label_last(reward, mask, done)  # outcome of the previous action
        if done:
            # There is no terminal dispatch/record_step, but its reward still counts.
            episode_log.ep_return += reward

        if not done:
            # Terminal observations label the previous action; do not dispatch another.
            sampler.launch(agent, observation, len(action_plan), action_type)
            with timer.context("plan"):
                if not action_plan and action_type != "human":
                    sample_start = time.time()
                    action_chunk, agent, sample_info = sampler.sample(agent, observation, step_metrics)
                    episode_log.sample_info_history.append(sample_info)
                    action_chunk = np.asarray(jax.device_get(action_chunk[:FLAGS.replan_steps]))
                    training_log.record_sample_time(time.time() - sample_start, step_metrics)
                    action_plan.extend(action_chunk)
                    sampler.launch(agent, observation, len(action_plan), action_type)
                else:
                    episode_log.sample_info_history.append(
                        episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                    )

            sleep_left = 0.0 if last_control_start is None else dt - (time.time() - last_control_start)
            step_metrics["timing/sleep_left_ms"] = sleep_left * 1000.0
            with timer.context("wait"):
                if sleep_left > 0:
                    time.sleep(sleep_left)

            has_action = bool(action_plan)
            action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
            with timer.context("act"):
                last_control_start = time.time()
                real_action, action_type = env.step(action.tolist())
            episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

            if action_type == "human":
                action_plan.clear()
                sampler.on_human_takeover()
            elif has_action:
                sampler.record_executed(agent, observation, real_action)

            if has_action or action_type == "human":
                transition = {
                    "observations": observation,
                    "actions": real_action,
                    "is_hil": (action_type == "human"),
                }
                sink.record_transition(i, transition)
                transitions_since_update += 1

        # Wire time of this step's env RPCs, as measured by the client wrapper.
        step_metrics["timing/network_ms"] = env.pop_network_ms()
        timer.tock("total")
        timer.tick("post_step")
        can_update = training_log.ep_count >= 10 and i >= FLAGS.batch_size
        if FLAGS.update_type == "step" and can_update:
            if FLAGS.num_updates > 0:
                run_agent_updates(FLAGS.num_updates, step_metrics)
            elif i % FLAGS.step_interval == 0:
                run_agent_updates(1, step_metrics)

        log_timing = not done and not first_step_of_episode
        first_step_of_episode = done  # next iteration starts a new episode if this one ended
        if done:
            # Reset starts now; the updates below run while the robot homes.
            pending_reset = reset_executor.submit(env.reset)
            sink.flush_transitions()
            with buffer_lock:
                batch_processor.on_episode_done(success)
            last_control_start = None
            sampler.on_episode_end()

            if not can_update:
                transitions_since_update = 0  # no catch-up burst when updates first unlock
            if FLAGS.update_type == "episode" and can_update:
                run_agent_updates(updates_due(), step_metrics)
            elif FLAGS.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= FLAGS.num_batch:
                    run_agent_updates(updates_due(), step_metrics)
                    episodes_since_update = 0

            training_log.on_episode_done(episode_log, success, step_metrics)
            episode_log.reset()
            with buffer_lock:
                batch_processor.on_episode_start()

            # Drain queued inserts, NFS saves and logs; interval checkpoint if due.
            sink.flush_episode(i, agent)

            pending_reset.result()  # re-raises reset failures on the main thread
            env.pop_network_ms()  # discard the reset round-trip so it doesn't inflate the next step
            done = False
            action_type = "policy"
            action_plan.clear()

        timer.tock("post_step")
        timer.tock("loop_time")
        step_metrics.update({f"timing/{k}_ms": v for k, v in timer.get_times_ms().items()})

        if not log_timing:
            step_metrics = {k: v for k, v in step_metrics.items()
                            if not k.startswith("timing/")}
        training_log.maybe_add_success_rate(i, step_metrics)
        sink.record_log(i, step_metrics)

    sink.flush_all()

    if FLAGS.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, FLAGS.max_steps)
            logging.info(f"Saved final agent checkpoint at step {FLAGS.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()

    sampler.close()
    if prefetch_pool is not None:
        prefetch_pool.shutdown(wait=False)
    reset_executor.shutdown(wait=True)


if __name__ == "__main__":
    app.run(main)
