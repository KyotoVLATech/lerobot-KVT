# Iloha online RL verification — 2026-10-06

## Result

**Real GPU learning now passes on this RTX 4090** with batch=16, UTD=10,
candidates=8, CPU-streamed batches, four actor microbatches of four samples,
the platform allocator, and a disk-backed target policy updated in bounded
chunks. The effective actor batch remains 16; target EMA is retained.
Three consecutive complete updates passed (actor steps 1/2/3, critic steps
10/20/30), with finite metrics, followed by finite 8x14 action inference.
Update times were 35.78, 26.79 and 26.24 seconds; warm pre-update inference
was 110–112 ms per eight-action chunk. Peak process RSS was 13.68 GiB.

This GPU check sampled 128 real demo transitions, not simultaneous full-dataset
replay plus training. All 222 episodes / 86,896 frames were separately validated.
Physical robot rollouts and long-duration operation remain unverified. No robot
or camera was accessed and no existing checkpoint was overwritten.

The test ran in a cgroup limited to 20 GiB RAM and 1 GiB swap. Cgroup usage
reached the RAM ceiling (including reclaimable mapped-file/cache pages), while
the process completed normally with exit code 0. The learner launch script now
uses the same safety limits. An earlier unprotected attempt caused a host OOM;
subsequent protected failures killed only the verification process.

Successful command, from `libs/expo-ft`:

```bash
systemd-run --user --scope --collect --unit=iloha-rl-gpu-verify -p MemoryMax=20G -p MemorySwapMax=1G /usr/bin/env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform JAX_COMPILATION_CACHE_DIR=/tmp/iloha-rl-jax-cache HF_HOME=/tmp/iloha-rl-hf OPENPI_DATA_HOME=/tmp/iloha-rl-openpi .venv/bin/python ../../scripts/iloha_rl_verify.py --stage update --batch-size 16 --candidates 8 --utd 10 --host-batches --actor-microbatch-size 4 --updates 3
```

The remaining sections record earlier verification and failed configurations;
their pre-fix failure statements do not describe the current successful result.

Hardware: RTX 4090, 24,564 MiB VRAM; host RAM 32 GiB. The learner environment was
installed separately in `libs/expo-ft/.venv`. OpenPI fork `real-time-expo-ft` was
cloned at commit `2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8` and configured using
the repository's iLoHa configuration patch.

## Checks performed

- Dataset: `libs/expo-ft/data/iloha_towel/success`, all 222 episodes and 86,896
  frames. Three uint8 CHW images `(3,224,224)`, finite 14D state/action, consistent
  lengths and the expected prompt. Decoded image storage is 36.546 GiB.
- Streaming loader: all transitions of episode 0 (301 frames) match the eager
  loader exactly, including observations, actions, rewards, masks and dones.
- SSD-backed replay: saved values and a seeded sample of 1,280 transitions match
  the RAM implementation exactly. All 86,896 frames were inserted into a buffer
  with capacity 500,000, with 222 terminal successes. The full insertion took
  182 seconds, used 36.546 GiB of scratch disk, and reached 19.703 GiB peak RSS
  in the standalone CPU verification process. This is not a measurement of
  simultaneous full replay + GPU training.
- Checkpoint:
  `libs/expo-ft/checkpoints/iloha_towel_rtc_offline/pi_rtc_iloha_towel_high_maxdelay10/checkpoints/10000/params`.
  Shape/dtype validation, online actor restore and target initialization pass.
  Target initialization copies restored params rather than building a second
  model and unused optimizer state, avoiding an observed initialization OOM.
- RealTimeEXPOFTLearner: construction, candidate selection, 8x14 finite action
  output and 5-step executed-prefix inference pass with N=32, n_edit_samples=32,
  filter_N=32. Warm prefix inference took approximately 148–158 ms per chunk.
  Initial compilation takes seconds; these warm timings exclude actual camera
  capture, robot state reads, network transfer and motor dispatch. Sustained
  real-robot 30 Hz control has **not** been verified.
- WebSocket: real EnvClient/serve_learner exchange using a simulated robot passes
  serialization, observations, action feedback, numeric 0/1 verdicts and terminal
  action suppression. This is a protocol test, not a physical robot rollout.
- Training: batch_size=64, UTD=20 (critic batch 1,280), original candidates and
  actor training enabled. With JAX memory fraction 0.85, OOM occurs transferring
  the batch to GPU (192,675,840-byte allocation). At 0.95, OOM occurs normalizing
  images to float32 (770,703,360-byte allocation). No gradient update completed.
- Ten hardware-free regression tests, static checks and launch-script syntax
  checks pass. Numeric input is `1` + Enter for success and `0` + Enter for
  failure; state_source remains measured by default, as requested.

## Reproduction

Run from `libs/expo-ft` with the learner environment:

```bash
.venv/bin/python ../../scripts/iloha_rl_verify.py --stage dataset
JAX_PLATFORMS=cpu OPENPI_DATA_HOME=/tmp/iloha-rl-openpi .venv/bin/python ../../scripts/iloha_rl_replay_verify.py --full
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 HF_HOME=/tmp/iloha-rl-hf OPENPI_DATA_HOME=/tmp/iloha-rl-openpi .venv/bin/python ../../scripts/iloha_rl_verify.py --stage infer
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HF_HOME=/tmp/iloha-rl-hf OPENPI_DATA_HOME=/tmp/iloha-rl-openpi .venv/bin/python ../../scripts/iloha_rl_verify.py --stage update
```

The GPU smoke test stores 128 real transitions to bound diagnostic host memory;
it retains the original update batch sizes and candidate counts. The separate
`--full` replay test validates all demos and the original replay capacity. No
learning defaults were reduced to obtain those original checks; see the
subsequent user-requested settings and streaming checks below.

From the repository root, using the robot environment:

```bash
.venv/bin/python scripts/iloha_rl_protocol_verify.py
.venv/bin/python -m unittest discover -s tests -p test_iloha_rl_pipeline.py
```

## CPU-streamed UTD implementation and verification

The launch script now uses the user's requested batch size 16, UTD 10 and
N/n_edit_samples/filter_N 8, plus `--host_update_batches`. Full demo coverage
(`num_data=0`), replay capacity, trainable parameters and learning rates remain
unchanged. The current launch default is memory fraction 0.95, preallocation
disabled, and the platform allocator; the 0.85 results below are historical.

Replay sampling snapshots the entire UTD batch on CPU, including normalization
and mixed-replay shuffle. Only one critic minibatch is transferred, augmented and
updated at a time, with completion waits before the next transfer. The code
also supports the original 1,280 -> 64 x 20 schedule; the current requested
schedule is 160 -> 16 x 10. Augmentation uses the original full-batch random key
stream to preserve per-image draws and update order. Policy/edit/temperature
updates still occur once after all critic updates. JIT stages return only changed
state, not unchanged model copies. Actor-stage JIT reuses donated actor/target
buffers; callers must use the returned learner, not reuse the old training state.

Real-checkpoint / real-demo GPU verification used 128 real transitions in
diagnostic replay, separate from the previous all-demo CPU replay test:

| Settings | Outcome |
| --- | --- |
| batch=16, UTD=10, candidates=8, CPU streaming, memory fraction 0.85 | CPU batch preparation passes; first critic update OOM (2,384,168,912-byte allocation). |
| Same, memory fraction 0.95 | All 10 critic updates pass; base-policy update OOM. |
| Same, plus actor/target buffer reuse | All 10 critic updates pass; base-policy update still OOM (7,527,754,016-byte allocation). |

Allocation sizes are failed requests, not a measurement of how much additional
VRAM would make training fit. No complete learner update, second update or
post-update inference passed. CPU streaming resolves full-UTD-batch residency,
but does not by itself resolve the policy-training peak.

Reproduce from `libs/expo-ft`:

```bash
JAX_PLATFORMS=cpu .venv/bin/python ../../scripts/iloha_rl_streaming_tests.py
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HF_HOME=/tmp/iloha-rl-hf OPENPI_DATA_HOME=/tmp/iloha-rl-openpi .venv/bin/python ../../scripts/iloha_rl_verify.py --stage update --batch-size 16 --candidates 8 --utd 10 --host-batches --updates 2
```

Five CPU regression tests pass: CPU/device sample equivalence and snapshot
ownership, full/sliced augmentation equivalence, CPU-only mixed replay, actual
UTD orchestration versus the original scan using toy losses, and the real
donated actor-update wrapper using a tiny policy. They verify update counts,
RNG progression and last-update metrics. These are not a numerical equivalence
test of the full VLA. The original ten hardware-free tests and selected static,
shell-syntax and diff checks also pass.

## Remaining work before robot rollouts

CPU UTD scheduling is implemented, but base-policy training still needs memory
improvements (e.g. gradient accumulation retaining the effective batch) or more
GPU memory. Batch/candidate/UTD reductions above were made at the user's explicit
request; demo coverage was retained.
Do not treat the successful inference check as a successful online RL check.

Disk replay preserves capacity using sparse temporary files, but a fully populated
500,000-frame buffer requires about 210.3 GiB for images alone, versus roughly
122 GiB currently available. Long-running replay and saved rollout buffers also
need sufficient additional storage. Scratch arrays used in verification were
removed automatically; the original dataset and checkpoint were retained.

Robot power is not needed to address the current GPU training failure. After a
real gradient update passes, physical rollout, manual verdict, reset/standby and
measured-state timing still need testing with the powered robot and operator.
