# Flipping Data Flywheel Refactor Plan

## Summary

Refactor `data_prepare.py` so the iterative offline data flywheel is configuration-driven instead of being encoded as repeated `if LOAD_COLLECTED:` blocks.

Only refactor flipping. Do not modify `data_prepare_folding.py`.

The refactor should preserve the current zarr schema, reward behavior, and processed data format as much as possible, while removing hard-coded active paths and repeated collected-data merge blocks.

## Key Changes

- Add YAML config support.
- Document the `PyYAML` dependency in the root installation guide.
- Support `python data_prepare.py --config path/to/config.yaml`.
- Add a default config at `configs/data_prepare.yaml`.
- Replace `LOAD_FROM_DATA` / `LOAD_COLLECTED` with a single `mode`:
  - `raw_to_npy`: raw teleop `demo_*.npy` -> processed `.npy`
  - `build_zarr`: processed teleop `.npy` + all configured rollout sources -> new zarr
  - `extend_zarr`: existing base zarr + new rollout sources -> new zarr, without modifying the base zarr
- Represent flywheel batches as named sources in YAML:
  - `teleop_sources`: processed `.npy` sources
  - `rollout_sources`: offline rollout `.h5` directory sources
  - keep all known rollout paths in the same YAML; use CLI flags to decide whether this run uses no rollouts, all rollouts, a rollout round, or a named rollout source
- Add CLI overrides:
  - `--mode raw_to_npy|build_zarr|extend_zarr`
  - `--include-rollouts` / `--no-rollouts`
  - `--rollout-round ROUND`
  - `--rollout-source NAME`
  - `--base-zarr-path PATH`
  - `--zarr-output-path PATH`
- Collapse all repeated `if LOAD_COLLECTED:` logic into one reusable rollout ingestion function.
- Keep existing zarr public fields:
  - `data/point_cloud`
  - `data/next_point_cloud`
  - `data/state`
  - `data/next_state`
  - `data/action`
  - `data/next_action`
  - `data/reward`
  - `data/return`
  - `data/done`
  - `data/timeout`
  - `meta/episode_ends`
- Replace `os.system('rm -rf ...')` with `shutil.rmtree`.
- Keep the old smoothness behavior, but expose it in config as `smooth_penalty` instead of the current misspelled `smooth_panelty`.

## Config Shape

```yaml
mode: build_zarr
overwrite: true

num_points: 1024
max_episode_len: 2000
lambda_penalty: 0.05
smooth_penalty: 0.01
only_success: false
min_episode_len: 30
include_rollouts: false

raw_teleop_dir: /path/to/raw_teleop_demo_dir
processed_npy_output: /path/to/processed/data_flipping_task.npy

base_zarr_path: null
zarr_output_path: /path/to/output/data_flipping.zarr

teleop_sources:
  - name: teleop_main
    path: /path/to/processed/data_flipping_task.npy
  - name: teleop_extra
    path: /path/to/processed/data_flipping_extra.npy

rollout_sources:
  - name: off2off_001
    round: "001"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/001/online_ft
  - name: off2off_002
    round: "002"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/002/online_ft
  - name: off2off_003
    round: "003"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/003/online_ft
  - name: off2off_004
    round: "004"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/004/online_ft
```

Replace placeholder paths with the local paths for the current machine/run.

## Data Flow

### `raw_to_npy`

- Read raw teleop files from `raw_teleop_dir`.
- Input files are `demo_*.npy`.
- For each frame, use:
  - `depth`
  - `depth_scale`
  - `ee_euler_action`
  - `qpos`
- Convert depth to point cloud with `depth2pc`.
- Downsample point cloud with `point_cloud_downsample`.
- Save processed dict to `processed_npy_output`.
- Do not write zarr in this mode.

### `build_zarr`

- Load all `teleop_sources`.
- Append transitions from processed `.npy` data.
- Load all `rollout_sources`.
- Append transitions from rollout `.h5` files.
- Write a new zarr at `zarr_output_path`.

### `extend_zarr`

- Read `base_zarr_path`.
- Load only newly configured rollout sources.
- Merge base zarr data plus new rollout transitions.
- Write the merged result to `zarr_output_path`.
- Never modify `base_zarr_path` in place.
- Use zarr `source_manifest` attrs to reject duplicate rollout source names in append-style runs.

## Rollout Source Handling

Each rollout source:

- Has `name`, `type`, and `path`.
- Currently only needs to support `type: h5_rollout_dir`.
- Recursively or consistently traverse the current two-level layout used by the existing script:
  - iterate subfolders under `path`
  - read `.h5` files inside each subfolder
- Expected `.h5` keys:
  - `point_cloud`
  - `state`
  - `action`
  - `next_action`
  - `reward`
  - `done`
  - `timeout`
  - `is_success`
- Skip trajectories shorter than `min_episode_len`.
- If `only_success` is true, skip trajectories where `is_success.any()` is false.
- If success is true, force the final reward to `1` before applying current penalties.
- Print per-source summary:
  - total trajectories
  - successful trajectories
  - skipped trajectories
  - appended episodes
  - appended transitions

## Reward And Return Behavior

Preserve the current behavior:

- Terminal success reward can be penalized by trajectory length:
  - `reward -= lambda_penalty * demo_length / max_episode_len`
- Smoothness penalty:
  - `reward -= smooth_penalty * ||action_t - action_{t-1}||`
- Discounted return:
  - `return[t] = reward[t] + gamma * return[t+1] * not_done[t]`
  - use `gamma = 0.99`

## Implementation Notes

Recommended functions:

- `load_config(path)`
- `process_raw_teleop_to_npy(config)`
- `load_processed_npy(path)`
- `append_processed_transitions(data, buffers, source_name, config)`
- `append_rollout_source(source, buffers, config)`
- `load_base_zarr(path, buffers)`
- `write_zarr(buffers, output_path, overwrite)`
- `compute_return(reward, not_done, gamma=0.99)`
- `safe_prepare_output_dir(path, overwrite)`

Use a shared in-memory `buffers` structure for:

- `point_cloud`
- `next_point_cloud`
- `state`
- `next_state`
- `action`
- `next_action`
- `reward`
- `done`
- `timeout`
- `episode_ends`

## Compatibility Requirements

- Keep current zarr dataset names.
- Keep current zarr dtype conventions:
  - point clouds and numeric arrays as `float32`
  - `done` and `timeout` as `bool`
  - `episode_ends` as `int64`
- Keep current chunking behavior close to the existing script:
  - leading chunk size `100`
  - preserve per-array trailing dimensions.
- Do not require changing training code that reads the zarr.
- Do not modify old zarr in `extend_zarr` mode.

## Static Cleanup Requirements

After refactor:

- No active hard-coded `/media/...` or `/home/...` paths in `data_prepare.py`.
- No active repeated `if LOAD_COLLECTED:` blocks.
- No active `LOAD_FROM_DATA` / `LOAD_COLLECTED` globals.
- No `os.system('rm -rf')`.
- No duplicate final `Saved zarr file` print.
- Keep comments concise and useful.

## Test Plan

Run:

```bash
python -m py_compile data_prepare.py
python data_prepare.py --help
```

Also verify by static search:

```bash
rg -n "LOAD_FROM_DATA|LOAD_COLLECTED|os\\.system|rm -rf|/media/|/home/" data_prepare.py
rg -n "if LOAD_COLLECTED" data_prepare.py
```

Expected:

- No active old flag usage.
- No active hard-coded data paths.
- No shell-based deletion.
- Help text shows `--config`.

If sample data is available, run one small config for each mode:

- `raw_to_npy`
- `build_zarr`
- `extend_zarr`

Validate zarr output contains:

```text
data/point_cloud
data/next_point_cloud
data/state
data/next_state
data/action
data/next_action
data/reward
data/return
data/done
data/timeout
meta/episode_ends
```

## Assumptions

- This task only refactors `data_prepare.py`.
- `data_prepare_folding.py` stays unchanged for now.
- YAML is required even though it adds `PyYAML`.
- The iterative offline flywheel needs all historical collected rollout batches to remain mergeable.
- The default YAML should preserve the current active data sources from the script.
- `extend_zarr` writes a new output zarr rather than appending in place.
