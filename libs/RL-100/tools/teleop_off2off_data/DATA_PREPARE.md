# Data Preparation

`data_prepare.py` prepares datasets from teleop demonstrations and offline rollout data. It is driven by YAML config instead of editing flags and paths inside the script.

Default config:

```text
configs/data_prepare.yaml
```

Run:

```bash
python data_prepare.py --config configs/data_prepare.yaml
```

Useful overrides:

```bash
# Ignore policy collected rollout sources even if they are listed in YAML.
python data_prepare.py --config configs/data_prepare.yaml --no-rollouts

# Full rebuild from teleop sources plus all enabled policy collected rollout sources.
python data_prepare.py --config configs/data_prepare.yaml --mode build_zarr --include-rollouts

# Full rebuild from teleop sources plus one rollout round.
python data_prepare.py --config configs/data_prepare.yaml --mode build_zarr --rollout-round 003

# Extend an existing zarr with one new rollout source and write a new zarr.
python data_prepare.py --config configs/data_prepare.yaml --mode extend_zarr --rollout-source off2off_004 --base-zarr-path /path/to/base.zarr --zarr-output-path /path/to/new.zarr
```

## Modes

`raw_to_npy` converts raw teleop collection files into a processed `.npy` dictionary.

Input:

```text
raw_teleop_dir/demo_*.npy
```

Output:

```text
processed_npy_output
```

`build_zarr` builds a zarr dataset from configured sources:

```text
teleop_sources   # processed .npy files
rollout_sources  # offline rollout .h5 directories
```

`extend_zarr` reads an existing `base_zarr_path`, appends configured rollout sources, and writes a new `zarr_output_path`. It never modifies the base zarr in place.

## Data Flywheel

Iterative offline collected data is represented as named YAML sources. When a new collected rollout batch is ready, add a new item under `rollout_sources`:

```yaml
rollout_sources:
  - name: off2off_004
    round: "004"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/online_ft
```

This replaces the old pattern of copying another `if LOAD_COLLECTED:` block into the script. The sources can stay in one YAML file; the CLI decides whether this run uses none, all, a round, or a single named source.

Use `build_zarr --include-rollouts` when:

- reward logic changed
- point cloud processing changed
- action/state schema changed
- filtering rules changed
- making a final reproducible training dataset

Use `extend_zarr` when:

- only a new policy collected rollout batch was added
- preprocessing and reward logic are unchanged
- a faster flywheel iteration is preferred

`extend_zarr` reads the old zarr and writes a new zarr. It does not modify the old zarr in place. The script records `source_manifest` in zarr attrs and refuses to append a source name that already exists in the base zarr manifest.

## Zarr Layout

The writer preserves the public dataset schema expected by training code:

```text
data/
  point_cloud
  next_point_cloud
  state
  next_state
  action
  next_action
  reward
  return
  done
  timeout

meta/
  episode_ends
```

## Main Config Fields

```yaml
mode: build_zarr
overwrite: true
include_rollouts: false
num_points: 1024
max_episode_len: 2000
lambda_penalty: 0.05
smooth_penalty: 0.01
only_success: false
min_episode_len: 30
raw_teleop_dir: /path/to/raw/demo_dir
processed_npy_output: /path/to/processed.npy
base_zarr_path: null
zarr_output_path: /path/to/output.zarr
teleop_sources: []
rollout_sources: []
```

## Notes

- `smooth_penalty` is the corrected config name for the old `smooth_panelty` behavior.
- Output overwrites use `shutil.rmtree`, not shell `rm -rf`.
- With zarr v3, the script uses zarr's current `create_array` API; with zarr v2, it uses `create_dataset`.
- `rollout_sources` may be listed in YAML while `include_rollouts: false`; this keeps paths centralized without changing default behavior.
