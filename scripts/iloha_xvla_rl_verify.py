#!/usr/bin/env python3
"""Run real X-VLA inference + EXPO/VLA updates without robot I/O; report peak VRAM."""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iloha_xvla_rl import EXPOLearner, XVLAAdapter, make_transition, parse_args  # noqa: E402


def main():
    args = parse_args()
    args.batch_size = 2
    args.vla_batch_size = 1
    args.vla_update_every = 1
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    adapter = XVLAAdapter(args)
    trainable_parameters = sum(p.numel() for p in adapter.trainable)
    obs = {f"observation.images.{cam}": np.zeros((480, 640, 3), dtype=np.uint8)
           for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    obs["task"] = "Fold the towel."
    cache = adapter.encode(obs)
    agent = EXPOLearner(adapter, cache["rl_features"].shape[-1], args)
    sample_start = time.monotonic()
    action = agent.choose([cache])
    if not torch.isfinite(action).all():
        raise AssertionError("Nonfinite action")
    sample_seconds = time.monotonic() - sample_start
    before = [p.detach().clone() for p in adapter.trainable]
    for done in (False, True):
        executed = action[0].detach().cpu().numpy()
        transition = make_transition(cache, executed, cache, [0] * (adapter.horizon - 1) + [0.7 if done else 0],
                                     done, adapter.horizon, args.discount)
        transition.episode_reward = 0.7
        agent.replay.append(transition)
    metrics = agent.update()
    if not any(not torch.equal(a, b) for a, b in zip(before, adapter.trainable, strict=True)):
        raise AssertionError("X-VLA trainable parameters did not change")
    with tempfile.TemporaryDirectory(prefix="xvla-expo-verify-") as directory:
        destination = Path(directory)
        agent.save(destination)
        # Reload the saved processors and weights, as iloha_eval.py does.
        # Release the first full VLA before loading to keep the 4090 memory bound.
        args.policy_path = str(destination / "pretrained_model")
        del before, agent, adapter
        if args.device == "cuda":
            torch.cuda.empty_cache()
        adapter = XVLAAdapter(args)
        restored = EXPOLearner(adapter, cache["rl_features"].shape[-1], args)
        restored.restore(destination)
        assert restored.updates == 1 and len(restored.replay) == 2
        restored_action = restored.choose([adapter.encode(obs)])
        assert torch.isfinite(restored_action).all()
    report = {"metrics": metrics, "sample_seconds": sample_seconds,
              "trainable_vla_parameters": trainable_parameters,
              "elapsed_seconds": time.monotonic() - start, "device": args.device,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if args.device == "cuda" else None,
              "note": "Synthetic RGB observations; validates execution/memory, not task learning or robot safety."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
