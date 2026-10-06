"""Check lossless streaming/disk replay against eager/RAM storage using real demos.

Run from libs/expo-ft with its .venv/bin/python and JAX_PLATFORMS=cpu.
--full also seeds every demo into a 500,000-frame disk-backed replay buffer.
Only temporary scratch storage is written; robot hardware is never accessed.
"""

import argparse
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

EXPO_ROOT = Path(__file__).resolve().parents[1] / "libs/expo-ft"
sys.path.insert(0, str(EXPO_ROOT))


def main():
    from configs.model.realtime_expo_ft_pi_config import get_config
    from configs.task.iloha_towel import get_config as task_config
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import process_droid_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = get_config()
    config.pi05_config_name = "expo_pi05_iloha_lora_finetune_sft_joint"
    config.pi05_assets_dir = str(EXPO_ROOT / "assets/expo_pi05_iloha_lora_finetune_sft_joint")
    config.pi05_asset_id = "iloha_towel_high"
    config.valids_keep_terminal_windows = True
    task = task_config()
    path = str(EXPO_ROOT / "data/iloha_towel/success")
    eager = process_droid_dataset(path, task, num_data=1)
    streamed = process_droid_dataset(path, task, num_data=1, streaming=True)
    assert len(eager) == len(streamed)
    for expected, actual in zip(eager, streamed, strict=True):
        for key in ("actions", "rewards", "masks", "dones"):
            np.testing.assert_array_equal(expected[key], actual[key])
        for key in expected["observations"]:
            np.testing.assert_array_equal(expected["observations"][key], actual["observations"][key])
    print(json.dumps({"event": "streaming_matches_eager", "frames": len(eager)}), flush=True)
    transitions = eager[:128]
    kwargs = {
        "config": config, "example_action": transitions[0]["actions"][None], "capacity": 160,
        "task_description": task.language_instruction, "replan_steps": 8, "seed": 42, "delay": 5,
        "critic_camera_keys": task.critic_camera_keys,
    }
    with tempfile.TemporaryDirectory(prefix="iloha-rl-replay-", dir="/tmp") as scratch:
        ram = create_replay_buffer(**kwargs)
        disk = create_replay_buffer(**kwargs, storage_dir=scratch)
        ram.insert_dataset(transitions)
        disk.insert_dataset(transitions)
        for key in ram.dataset_dict:
            np.testing.assert_array_equal(ram.dataset_dict[key][:len(ram)], disk.dataset_dict[key][:len(disk)])
        ram_batch = ram.sample_jax(64 * 20)
        disk_batch = disk.sample_jax(64 * 20)
        for key in ram_batch:
            np.testing.assert_array_equal(np.asarray(ram_batch[key]), np.asarray(disk_batch[key]))
        print(json.dumps({"event": "disk_replay_matches_ram", "sampled_frames": 1280}), flush=True)
        del ram, disk, ram_batch, disk_batch, eager, transitions, streamed
        gc.collect()
        if args.full:
            dataset = process_droid_dataset(path, task, streaming=True)
            kwargs.update(capacity=500_000)
            full = create_replay_buffer(**kwargs, storage_dir=scratch)
            start = time.perf_counter()
            full.insert_dataset(dataset)
            assert len(full) == len(dataset)
            assert full.count_episodes_chronological() == 222
            assert np.all(full.dataset_dict["is_success"][:len(full)])
            assert float(np.sum(full.dataset_dict["rewards"][:len(full)])) == 222.
            sample = full.sample_jax(64)
            assert all(np.isfinite(np.asarray(v)).all() for v in sample.values())
            for image in ("base_image", "left_wrist_image", "right_wrist_image"):
                full.dataset_dict[image].flush()
            print(json.dumps({
                "event": "all_demos_seeded", "frames": len(full), "episodes": 222,
                "capacity": full._capacity, "seconds": time.perf_counter() - start,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "allocated_disk_gib": sum(p.stat().st_blocks * 512 for p in Path(scratch).rglob("*.npy")) / 2**30,
            }), flush=True)
            del sample, full
            gc.collect()


if __name__ == "__main__":
    main()
