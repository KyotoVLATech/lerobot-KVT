from pathlib import Path

import zarr

merged_path = Path(__file__).resolve().parent / 'exp_local' / 'data' / 'merged_small.zarr'
merged = zarr.open(str(merged_path), 'r')
target = merged['meta']['episode_ends'][-1]
print('episode_ends last =', target)

for k, arr in merged['data'].items():
    print(f'{k:15s} shape[0]={arr.shape[0]}  diff={arr.shape[0]-target}')
