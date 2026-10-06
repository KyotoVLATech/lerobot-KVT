#!/usr/bin/env python
"""Robot-free checks of an RTC-SFT checkpoint before online RL (RealTimeEXPOFTLearner).

1. Latency of the real-time sampling path used by AsyncChunkSampler:
     sample_pre_cache (pi0.5, runs in the background while `delay` actions execute) and
     sample_actions (edit + critic selection, blocking at the replan boundary).
   The pre-cache must finish within delay / control_hz seconds.
2. Open-loop action error on demo episodes: base-actor chunks (delay 0) vs. the demo's next
   `replan_steps` actions, against a "hold the current state" baseline.

Usage (same model/config flags as scripts/iloha_towel/run_server.sh):
    python scripts/iloha/check_policy.py --config_task=configs/task/iloha_towel.py \
        --config=configs/model/realtime_expo_ft_pi_config.py --config.pi05_weight_loader_path=... \
        --dataset_path=./data/iloha_towel/success --episodes=0,50,100,150,200
"""

import logging
import time

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import expo_ft.agents  # noqa: F401  (import order: avoids a circular import)
from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.agents.alg.realtime_expo_ft import load_agent
from expo_ft.agents.vla.pi05 import build_pi05
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.train_utils import init_logging

import openpi.training.sharding as openpi_sharding

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "", "traj.hdf5 episode directory.")
flags.DEFINE_string("episodes", "0,50,100,150,200", "Comma-separated episode indices for the open-loop check.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("replan_steps", 8, "Action-chunk execution horizon.")
flags.DEFINE_integer("delay", 5, "Deployed --delay (env steps).")
flags.DEFINE_integer("latency_reps", 30, "Timed repetitions per latency measurement.")
flags.DEFINE_integer("open_loop_stride", 8, "Evaluate every N-th step of each episode.")
config_flags.DEFINE_config_file("config", "configs/model/realtime_expo_ft_pi_config.py", "Model config.",
                                lock_config=False)
config_flags.DEFINE_config_file("config_task", "configs/task/iloha_towel.py", "Task config.", lock_config=False)


def _timed(fn, reps):
    out = fn()  # warm-up / compile
    jax.block_until_ready(out[0])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out[0])
        ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts), out


def main(_):
    init_logging()
    logger = logging.getLogger(__name__)
    task = FLAGS.config_task
    episodes = [int(e) for e in FLAGS.episodes.split(",")]
    dataset = process_droid_dataset(FLAGS.dataset_path, task, episode_indices=episodes)

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    example_action = dataset[0]["actions"][np.newaxis]
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding, False, task.language_instruction,
    )

    critic_camera_keys = tuple(getattr(task, "critic_camera_keys", CRITIC_CAMERA_KEYS))
    rb = create_replay_buffer(config=FLAGS.config, example_action=example_action, capacity=4,
                              task_description=task.language_instruction, replan_steps=FLAGS.replan_steps,
                              seed=FLAGS.seed, delay=FLAGS.delay, critic_camera_keys=critic_camera_keys)
    rb.insert_dataset(dataset[:2])
    critic_example = {_critic_key_to_storage(k): rb.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
                      for k in critic_camera_keys}
    critic_example["state"] = rb.dataset_dict["state"][0][np.newaxis]
    critic_example["actions"] = rb.dataset_dict["actions"][0][np.newaxis]
    ex_obs, ex_state, ex_action = rb.convert_to_critic_format(critic_example)
    actor.action_dim = ex_action.squeeze().shape[-1]
    actor.state_dim = ex_state.squeeze().shape[-1]
    agent_kwargs["delay"] = FLAGS.delay
    agent_kwargs["critic_camera_keys"] = critic_camera_keys
    agent = load_agent(
        seed=FLAGS.seed, example_observation=ex_obs.squeeze(), example_action=ex_action.squeeze(),
        example_state=ex_state.squeeze(), actor=actor, actor_train_state=actor_train_state,
        target_actor_params=target_actor_params, agent_kwargs=agent_kwargs, metadata=vla_metadata,
        mesh=mesh, data_sharding=data_sharding, replicated_sharding=replicated_sharding, resume=False,
        replan_steps=FLAGS.replan_steps, default_prompt=task.language_instruction,
        edit_action_xyzg=task.edit_action_xyzg,
    )

    obs = dict(dataset[100 if len(dataset) > 100 else 0]["observations"])
    r, d = FLAGS.replan_steps, FLAGS.delay
    dt_ms = 1000.0 / task.control_hz

    # ---- 1. latency ----
    t_boot, (chunk, _, info) = _timed(lambda: agent.sample_actions(dict(obs)), FLAGS.latency_reps)
    prefix = info["executed_padded"][r - d:r]
    t_pre, (precached, _) = _timed(lambda: agent.sample_pre_cache(dict(obs), prefix_padded=prefix),
                                   FLAGS.latency_reps)
    t_fast, _ = _timed(lambda: agent.sample_actions(dict(obs), precached=precached, delay=d), FLAGS.latency_reps)

    def fmt(t):
        return f"median {np.median(t):6.1f} ms | p90 {np.percentile(t, 90):6.1f} ms | max {t.max():6.1f} ms"

    print("\n================ latency (N=%d candidates, delay=%d, %d Hz) ================"
          % (agent.N, d, task.control_hz))
    print(f"boot  sample_actions (pi0.5 + select, blocking) : {fmt(t_boot)}")
    print(f"async sample_pre_cache (pi0.5 inpainted)        : {fmt(t_pre)}")
    print(f"      budget = delay x step = {d} x {dt_ms:.1f} = {d * dt_ms:.0f} ms"
          f"  -> min delay for p90 = {int(np.ceil(np.percentile(t_pre, 90) / dt_ms))}")
    print(f"sync  sample_actions (edit + critic, blocking)  : {fmt(t_fast)}"
          f"  (step budget {dt_ms:.1f} ms)")

    # ---- 2. open-loop error vs demos ----
    ends = [i for i, t in enumerate(dataset) if t["dones"]]
    starts = [0] + [e + 1 for e in ends[:-1]]
    err_policy, err_hold = [], []
    for s, e in zip(starts, ends):
        for t in range(s, e + 1 - r, FLAGS.open_loop_stride):
            o = dict(dataset[t]["observations"])
            pred, agent, _ = agent.sample_actions(o, only_base_actions=True)
            pred = np.asarray(jax.device_get(pred))[:r]
            gt = np.stack([dataset[t + k]["actions"] for k in range(r)])
            err_policy.append(np.abs(pred - gt))
            err_hold.append(np.abs(np.asarray(o["state"])[None] - gt))
    err_policy, err_hold = np.stack(err_policy), np.stack(err_hold)  # (n, r, 14)

    names = [f"L{j}" for j in range(6)] + ["Lgrip"] + [f"R{j}" for j in range(6)] + ["Rgrip"]
    print("\n================ open-loop |pred - demo| (ALOHA units; %d chunks from %d episodes) ================"
          % (len(err_policy), len(starts)))
    print("joint   " + " ".join(f"{n:>6}" for n in names))
    print("policy  " + " ".join(f"{v:6.3f}" for v in err_policy.mean((0, 1))))
    print("hold    " + " ".join(f"{v:6.3f}" for v in err_hold.mean((0, 1))))
    print("by chunk step (mean over joints): policy "
          + " ".join(f"{v:.3f}" for v in err_policy.mean((0, 2)))
          + " | hold " + " ".join(f"{v:.3f}" for v in err_hold.mean((0, 2))))
    print(f"overall: policy {err_policy.mean():.4f}  hold {err_hold.mean():.4f}  "
          f"(ratio {err_policy.mean() / err_hold.mean():.2f})")
    logger.info("done")


if __name__ == "__main__":
    app.run(main)
