import os
from glob import glob
from pathlib import Path

import zarr
import numcodecs
import numpy as np
from tqdm import tqdm

def merge_zarr_folder(src_folder: str,
                      out_path: str,
                      chunk_size: int = 100):
    """
    把 src_folder 里所有 .zarr 目录合并成 out_path 这一个 .zarr
    """
    # 收集所有 .zarr 目录
    zarr_files = sorted(glob(os.path.join(src_folder, '*.zarr')))
    if not zarr_files:
        raise RuntimeError(f'在 {src_folder} 下未找到任何 .zarr 文件')

    print(f'共找到 {len(zarr_files)} 个 zarr 文件，开始合并…')

    # 用第一个文件推断数据结构
    z0 = zarr.open(zarr_files[0], mode='r')
    data_keys = list(z0['data'].keys())
    data_shapes = {k: z0['data'][k].shape[1:] for k in data_keys}
    data_dtypes = {k: z0['data'][k].dtype    for k in data_keys}

    # 创建输出 zarr
    os.makedirs(out_path, exist_ok=True)        # 确保目录存在
    out_root = zarr.open(out_path, mode='w')
    out_data = out_root.create_group('data')
    out_meta = out_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # 预建数据集（空，待 append）
    for k in data_keys:
        out_data.create_dataset(
            k,
            shape=(0,)+data_shapes[k],
            dtype=data_dtypes[k],
            chunks=(chunk_size,)+data_shapes[k],
            compressor=compressor,
            overwrite=True
        )

    out_meta.create_dataset(  # 记录每个 episode 终止索引
        'episode_ends',
        shape=(0,),
        dtype='int64',
        chunks=(chunk_size,),
        compressor=compressor,
        overwrite=True
    )
    out_meta.create_dataset(  # 记录该 episode 对应的 ckpt 名
        'ckpt_name',
        shape=(0,),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),  # 允许可变长 UTF-8 字符串
        chunks=(chunk_size,),
        compressor=compressor,
        overwrite=True
    )

    # 迭代合并
    total_steps = 0
    for zarr_dir in tqdm(zarr_files, desc='Merging'):
        z = zarr.open(zarr_dir, mode='r')

        # 1) 写 data
        n_steps = z['data'][data_keys[0]].shape[0]
        for k in data_keys:
            out_data[k].append(z['data'][k][:], axis=0)

        # 2) 写 meta
        ep_ends = z['meta']['episode_ends'][:]
        out_meta['episode_ends'].append(ep_ends + total_steps, axis=0)
        ckpt_label = Path(zarr_dir).name          # 目录名作为标签
        out_meta['ckpt_name'].append(
            [ckpt_label]*len(ep_ends), axis=0
        )

        total_steps += n_steps

    print(f'合并完成！输出文件：{out_path}')
    print(f'总 step 数：{total_steps}')
    print(f'数据键：{data_keys}')

if __name__ == '__main__':

    drm_root = Path(__file__).resolve().parent
    data_dir = drm_root / 'exp_local' / 'data'

    merge_zarr_folder(
        src_folder=str(data_dir),   # <-- 输入：含 *.zarr 的目录
        out_path=str(data_dir / 'merged_small1.zarr'),       # <-- 输出：合并后的 zarr
        chunk_size =100                           # 可按需调整
    )
