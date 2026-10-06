"""Verify real iLoHa EXPO-FT data/model/update paths without robot I/O.

Run in libs/expo-ft's Python environment, from libs/expo-ft:
  .venv/bin/python ../../scripts/iloha_rl_verify.py --stage dataset
  .venv/bin/python ../../scripts/iloha_rl_verify.py --stage update

GPU smoke tests use a bounded subset of real demo transitions for storage, but
UTD, batch size and candidate counts default to 20, 64 and 32 respectively.
They do not train the robot or
overwrite checkpoints. A successful smoke test is not a full replay-capacity test.
"""

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np

EXPO_ROOT = Path(__file__).resolve().parents[1] / "libs/expo-ft"
sys.path.insert(0, str(EXPO_ROOT))


def log(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def audit_dataset(root):
    episodes = sorted(root.glob("*/traj.hdf5"), key=lambda p: int(p.parent.name))
    if not episodes:
        raise ValueError(f"No traj.hdf5 episodes in {root}")
    frames = 0
    for path in episodes:
        with h5py.File(path, "r") as f:
            action = f["action/joint_position"]
            n = len(action)
            if n < 1 or action.shape != (n, 14):
                raise ValueError(f"Invalid actions: {path}: {action.shape}")
            obs = f["saved_observation"]
            if obs["state"].shape != (n, 14) or len(obs["prompt"]) != n:
                raise ValueError(f"Invalid state/prompt: {path}")
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                image = obs[f"{name}_image"]
                if image.shape != (n, 3, 224, 224) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid image: {path}: {name}: {image.shape}/{image.dtype}")
            if not np.isfinite(action[:]).all() or not np.isfinite(obs["state"][:]).all():
                raise ValueError(f"Nonfinite state/action: {path}")
            prompts = set(obs["prompt"].asstr()[:])
            if prompts != {"Grab the edge of the towel and fold it twice."}:
                raise ValueError(f"Unexpected prompts: {path}: {prompts}")
            frames += n
    log("dataset_valid", episodes=len(episodes), frames=frames,
        decoded_image_gib=frames * 3 * 3 * 224 * 224 / 2**30)
    return episodes


def gpu_verify(args, episodes):
    import jax
    from configs.model.realtime_expo_ft_pi_config import get_config
    from configs.task.iloha_towel import get_config as task_config
    from expo_ft.agents.alg.realtime_expo_ft import load_agent
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import _critic_key_to_storage, create_replay_buffer
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.utils.loop_utils import AsyncChunkSampler
    from openpi.training import sharding

    update_logger = logging.getLogger("expo_ft.agents.alg.realtime_expo_ft")
    update_logger.setLevel(logging.INFO)
    update_logger.addHandler(logging.StreamHandler())
    update_logger.propagate = False

    devices = jax.devices()
    log("devices", devices=[str(d) for d in devices])
    if not all(d.platform == "gpu" for d in devices):
        raise RuntimeError("GPU verification requires a JAX GPU backend")
    config = get_config()
    config.N = config.n_edit_samples = config.filter_N = args.candidates
    config.actor_microbatch_size = args.actor_microbatch_size
    config.filter_n_edit = 1
    config.edit_scale = .1
    config.filter_add_delayed_obs = True
    config.valids_keep_terminal_windows = True
    config.pi05_config_name = "expo_pi05_iloha_lora_finetune_sft_joint"
    config.pi05_weight_loader_path = str(args.checkpoint.resolve())
    config.pi05_assets_dir = str(EXPO_ROOT / "assets/expo_pi05_iloha_lora_finetune_sft_joint")
    config.pi05_asset_id = "iloha_towel_high"
    task = task_config()
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    start = time.perf_counter()
    actor, train_state, target, kwargs, metadata = build_pi05(
        config, 42, mesh, data_sharding, replicated, False, task.language_instruction,
    )
    jax.block_until_ready(train_state)
    log("checkpoint_restored", seconds=time.perf_counter() - start)
    if args.stage == "restore":
        return

    # Bound diagnostic host memory; the training launch script still loads all demos.
    data = process_droid_dataset(str(args.dataset), task, episode_indices=[0])[:args.frames]
    if len(data) <= 16:
        raise ValueError("Smoke test needs at least 17 real demo transitions")
    obs = dict(data[0]["observations"])
    buffer = create_replay_buffer(
        config, data[0]["actions"][None], len(data) + 32, task.language_instruction,
        8, 42, delay=5, critic_camera_keys=task.critic_camera_keys,
    )
    processor = BatchProcessor(
        buffer, buffer, data_sharding, args.batch_size, args.utd, 0., True, False,
        dataset=data, replay_prefetch=1, host_batches=args.host_batches,
    )
    example = {
        _critic_key_to_storage(k): buffer.dataset_dict[_critic_key_to_storage(k)][0][None]
        for k in task.critic_camera_keys
    }
    example["state"] = buffer.dataset_dict["state"][0][None]
    example["actions"] = buffer.dataset_dict["actions"][0][None]
    image, state, action = buffer.convert_to_critic_format(example)
    actor.action_dim = action.squeeze().shape[-1]
    actor.state_dim = state.squeeze().shape[-1]
    kwargs.update(delay=5, critic_camera_keys=task.critic_camera_keys)
    agent = load_agent(
        42, image.squeeze(), action.squeeze(), state.squeeze(), actor, train_state,
        target, kwargs, metadata, mesh, data_sharding, replicated, False, 8,
        task.language_instruction, False,
    )
    if args.host_batches:
        agent = agent.offload_target_actor()
    del train_state, target, kwargs
    log("learner_initialized", stored_demo_frames=len(data), batch_size=args.batch_size,
        utd=args.utd, candidates=args.candidates, host_batches=args.host_batches)
    sampler = AsyncChunkSampler(agent, delay=5, replan_steps=8, executed_prefix=True)
    sampler.observe(dict(obs))
    start = time.perf_counter()
    actions, agent, info = sampler.sample(agent, dict(obs), {})
    actions = np.asarray(jax.device_get(actions))
    if actions.shape != (8, 14) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid policy output: {actions.shape}")
    log("inference_pass", seconds=time.perf_counter() - start, action_shape=list(actions.shape))
    for i, row in enumerate(actions):
        if i:
            sampler.observe(dict(data[i]["observations"]))
        sampler.record_executed(agent, data[i]["observations"], row)
    sampler.observe(dict(data[8]["observations"]))
    start = time.perf_counter()
    actions, agent, info = sampler.sample(agent, dict(data[8]["observations"]), {})
    jax.block_until_ready(actions)
    log("rtc_prefix_inference_pass", seconds=time.perf_counter() - start, delay=info["delay"])
    for offset in (8, 16):
        for i, row in enumerate(np.asarray(jax.device_get(actions))):
            if i:
                sampler.observe(dict(data[offset + i]["observations"]))
            sampler.record_executed(agent, data[offset + i]["observations"], row)
        sampler.observe(dict(data[offset + 8]["observations"]))
        start = time.perf_counter()
        actions, agent, info = sampler.sample(agent, dict(data[offset + 8]["observations"]), {})
        jax.block_until_ready(actions)
        log("rtc_warm_inference", seconds=time.perf_counter() - start)
    sampler.close()
    if args.stage == "infer":
        return

    for update_index in range(args.updates):
        start = time.perf_counter()
        batch, actor_batch, _ = processor.next_batch(jax.random.PRNGKey(142 + update_index))
        log("batch_prepared", critic_batch=batch["actions"].shape[0], actor_batch=actor_batch["actions"].shape[0])
        previous_step = int(jax.device_get(agent.actor_train_state.step))
        previous_critic_step = int(jax.device_get(agent.critic.step))
        def actor_signature(state):
            leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
            return str(treedef), {str(path): (tuple(x.shape), str(x.dtype), bool(getattr(x, "weak_type", False)))
                                 for path, x in leaves if hasattr(x, "shape")}
        previous_signature = actor_signature(agent.actor_train_state)
        agent, metrics = agent.update(agent, batch, args.utd, actor_batch)
        jax.block_until_ready(agent)
        values = {k: np.asarray(jax.device_get(v)) for k, v in metrics.items()}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite training metrics")
        current_step = int(jax.device_get(agent.actor_train_state.step))
        if current_step != previous_step + 1:
            raise ValueError(f"Actor did not update: {previous_step} -> {current_step}")
        critic_step = int(jax.device_get(agent.critic.step))
        current_signature = actor_signature(agent.actor_train_state)
        changed = {k: [v, current_signature[1].get(k)] for k, v in previous_signature[1].items()
                   if v != current_signature[1].get(k)}
        log("actor_signature", tree_changed=previous_signature[0] != current_signature[0], changed_leaves=changed)
        if critic_step != previous_critic_step + args.utd:
            raise ValueError(f"Critic update count mismatch: {previous_critic_step} -> {critic_step}")
        log("update_pass", seconds=time.perf_counter() - start, actor_step=current_step,
            critic_step=critic_step,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            memory_stats=devices[0].memory_stats(),
            metrics={k: float(v) for k, v in values.items() if v.ndim == 0})
        del batch, actor_batch, metrics
    sampler = AsyncChunkSampler(agent, delay=5, replan_steps=8, executed_prefix=True)
    try:
        sampler.observe(dict(obs))
        actions, agent, _ = sampler.sample(agent, dict(obs), {})
        actions = np.asarray(jax.device_get(actions))
        if actions.shape != (8, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid inference output after training")
        log("post_update_inference_pass", action_shape=list(actions.shape))
    finally:
        sampler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dataset", "restore", "infer", "update"), default="dataset")
    parser.add_argument("--dataset", type=Path, default=EXPO_ROOT / "data/iloha_towel/success")
    parser.add_argument("--checkpoint", type=Path, default=EXPO_ROOT / "checkpoints/iloha_towel_rtc_offline/pi_rtc_iloha_towel_high_maxdelay10/checkpoints/10000/params")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--host-batches", action="store_true")
    parser.add_argument("--actor-microbatch-size", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()
    if args.frames < 33:
        parser.error("--frames must be at least 33")
    if min(args.batch_size, args.updates, args.candidates, args.utd) < 1:
        parser.error("--batch-size, --candidates, --utd and --updates must be positive")
    if args.actor_microbatch_size < 0 or (args.actor_microbatch_size and args.batch_size % args.actor_microbatch_size):
        parser.error("--actor-microbatch-size must be 0 or a positive divisor of --batch-size")
    episodes = audit_dataset(args.dataset)
    if args.stage != "dataset":
        gpu_verify(args, episodes)


if __name__ == "__main__":
    main()
