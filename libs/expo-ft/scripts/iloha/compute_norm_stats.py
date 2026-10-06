"""Compute OpenPI normalization stats from expo-ft traj.hdf5 episodes.

Same statistics as OpenPI's scripts/compute_norm_stats.py -- `state` and `actions` after the
config's repack + data transforms, with actions taken as `action_horizon` chunks clamped at the
episode end (LeRobot's delta_timestamps padding) -- but read from the HDF5 export, since the
OpenPI-pinned lerobot cannot load LeRobot v3.0 datasets.

Usage:
    python scripts/iloha/compute_norm_stats.py \
        --config_name expo_pi05_iloha_lora_finetune_sft_joint \
        --dataset_path ./data/iloha_towel/success \
        --assets_dir ./assets/expo_pi05_iloha_lora_finetune_sft_joint \
        --asset_id iloha_towel_high
"""

import argparse
import os

import h5py
import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config

from expo_ft.env.droid_utils import _discover_episode_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--asset_id", required=True)
    parser.add_argument("--action_key", default="joint_position")
    parser.add_argument("--num_data", type=int, default=0, help="Max episodes (0 = all).")
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    transform = lambda d: d  # noqa: E731
    for t in [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]:
        transform = (lambda f, g: (lambda d: g(f(d))))(transform, t)
    horizon = config.model.action_horizon

    ep_dirs = _discover_episode_dirs(args.dataset_path)
    if args.num_data > 0:
        ep_dirs = ep_dirs[:args.num_data]

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    dummy = np.zeros((3, 1, 1), dtype=np.uint8)  # images are not part of the stats
    for ep in tqdm.tqdm(ep_dirs, desc="episodes"):
        with h5py.File(os.path.join(ep, "traj.hdf5"), "r") as f:
            states = np.asarray(f["saved_observation"]["state"])
            actions = np.asarray(f["action"][args.action_key])
        T = len(actions)
        ep_states, ep_actions = [], []
        for t in range(T):
            idx = np.minimum(np.arange(t, t + horizon), T - 1)
            out = transform({
                "cam_high_image": dummy, "cam_left_wrist_image": dummy, "cam_right_wrist_image": dummy,
                "state": states[t].copy(), "actions": actions[idx].copy(), "prompt": "",
            })
            ep_states.append(out["state"])
            ep_actions.append(out["actions"])
        stats["state"].update(np.stack(ep_states))
        stats["actions"].update(np.stack(ep_actions))

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    out_dir = os.path.join(args.assets_dir, args.asset_id)
    normalize.save(out_dir, norm_stats)
    for k, s in norm_stats.items():
        print(k, "q01", np.round(s.q01, 3), "\n", k, "q99", np.round(s.q99, 3))
    print(f"Wrote {out_dir}/norm_stats.json")


if __name__ == "__main__":
    main()
