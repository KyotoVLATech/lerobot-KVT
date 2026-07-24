#!/usr/bin/env python3
"""
Truly parallel replay buffer loader that splits each array into multiple jobs.
"""

import zarr
import numpy as np
from tqdm import tqdm
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

RL100_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, RL100_ROOT)

def load_array_chunk(args):
    """Load a chunk of an array"""
    zarr_path, key, start_idx, end_idx, dst_array, offset = args
    
    # Open zarr in this thread
    root = zarr.open(zarr_path, 'r')
    src_array = root['data'][key]
    
    # Load chunk directly into shared destination array
    # Using a view to write to the correct position
    dst_view = dst_array[offset:offset + (end_idx - start_idx)]
    dst_view[:] = src_array[start_idx:end_idx]
    
    return end_idx - start_idx

def parallel_load_array(zarr_path, key, array_info, num_workers=32):
    """Load a single array using multiple workers"""
    
    shape = array_info['shape']
    dtype = array_info['dtype']
    chunks = array_info['chunks']
    
    # Pre-allocate destination array
    dst_array = np.empty(shape, dtype=dtype)
    
    # Determine job size - split along first dimension
    total_items = shape[0]
    chunk_size = chunks[0] if chunks else 100
    
    # Create more jobs for better parallelism
    # Each job loads one or few chunks
    jobs_per_worker = 4  # Create 4x more jobs than workers for better load balancing
    total_jobs = min(num_workers * jobs_per_worker, (total_items + chunk_size - 1) // chunk_size)
    items_per_job = max(chunk_size, total_items // total_jobs)
    
    # Round to chunk boundaries for better performance
    items_per_job = ((items_per_job + chunk_size - 1) // chunk_size) * chunk_size
    
    # Create job list
    jobs = []
    for start_idx in range(0, total_items, items_per_job):
        end_idx = min(start_idx + items_per_job, total_items)
        jobs.append((zarr_path, key, start_idx, end_idx, dst_array, start_idx))
    
    # Load in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(load_array_chunk, jobs))
    
    return dst_array

def fast_parallel_load_zarr(zarr_path, num_workers=128, keys=None):
    """
    Fast loading that truly utilizes all available workers.
    Splits each large array into multiple parallel jobs.
    
    Args:
        zarr_path: Path to zarr directory
        num_workers: Number of parallel workers
        keys: Optional list of keys to load. If None, loads all keys.
    """
    
    print(f"Fast parallel loading from: {zarr_path}")
    print(f"Using {num_workers} workers")
    if keys:
        print(f"Loading only keys: {keys}")
    
    # Open zarr
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store, mode='r')
    
    # Load metadata
    meta = {}
    if 'meta' in root:
        print("Loading metadata...")
        for key, value in root['meta'].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]
    
    # Analyze arrays
    data = {}
    array_info = {}
    
    if 'data' in root:
        print("\nAnalyzing arrays...")
        total_size = 0
        all_keys = list(root['data'].keys())
        
        # Filter keys if specified
        if keys:
            load_keys = [k for k in keys if k in all_keys]
            missing_keys = [k for k in keys if k not in all_keys]
            if missing_keys:
                print(f"Warning: Requested keys not found in zarr: {missing_keys}")
        else:
            load_keys = all_keys
        
        for key in load_keys:
            arr_meta = root['data'][key]
            info = {
                'shape': arr_meta.shape,
                'dtype': arr_meta.dtype,
                'chunks': arr_meta.chunks,
                'nbytes': np.prod(arr_meta.shape) * np.dtype(arr_meta.dtype).itemsize
            }
            array_info[key] = info
            total_size += info['nbytes']
            print(f"  {key}: {info['shape']} ({info['nbytes']/(1024**2):.1f} MB)")
        
        print(f"\nTotal data size to load: {total_size/(1024**3):.1f} GB")
    
    # Sort arrays by size
    sorted_keys = sorted(array_info.keys(), 
                        key=lambda k: array_info[k]['nbytes'], 
                        reverse=True)
    
    # Load arrays
    start_time = time.time()
    
    # Load small arrays serially (< 100 MB)
    small_arrays = [k for k in sorted_keys if array_info[k]['nbytes'] < 100 * 1024 * 1024]
    large_arrays = [k for k in sorted_keys if array_info[k]['nbytes'] >= 100 * 1024 * 1024]
    
    if small_arrays:
        print(f"\nLoading {len(small_arrays)} small arrays...")
        for key in tqdm(small_arrays):
            data[key] = root['data'][key][:]
    
    # Load large arrays with true parallelism
    if large_arrays:
        print(f"\nLoading {len(large_arrays)} large arrays with {num_workers} workers...")
        
        # Calculate workers per array based on size
        total_large_size = sum(array_info[k]['nbytes'] for k in large_arrays)
        
        with tqdm(total=len(large_arrays)) as pbar:
            for key in large_arrays:
                # Allocate workers proportional to array size
                array_size = array_info[key]['nbytes']
                array_workers = max(1, int(num_workers * array_size / total_large_size))
                array_workers = min(array_workers, num_workers)
                
                # For very large arrays, use more workers
                if array_size > 10 * 1024**3:  # > 10 GB
                    array_workers = max(array_workers, num_workers // 2)
                
                pbar.set_postfix_str(f"Loading {key} with {array_workers} workers")
                
                # Load array
                data[key] = parallel_load_array(
                    zarr_path, key, array_info[key], 
                    num_workers=array_workers
                )
                
                pbar.update(1)
    
    total_time = time.time() - start_time
    total_gb = sum(info['nbytes'] for info in array_info.values()) / (1024**3)
    
    print(f"\nLoading complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total data: {total_gb:.1f} GB")
    print(f"Speed: {total_gb / total_time:.1f} GB/s")
    
    return {
        'meta': meta,
        'data': data
    }

def test_parallel_loading():
    """Test the truly parallel loading"""
    
    zarr_path = os.path.join(RL100_ROOT, "data", "cloth_rgb_processed_action_rechunked.zarr")
    
    if not os.path.exists(zarr_path):
        print(f"Dataset not found: {zarr_path}")
        return
    
    # Test with different worker counts
    for num_workers in [32, 64, 128]:
        print(f"\n{'='*80}")
        print(f"Testing with {num_workers} workers")
        print('='*80)
        
        start = time.time()
        data = fast_parallel_load_zarr(zarr_path, num_workers=num_workers)
        elapsed = time.time() - start
        
        # Convert to ReplayBuffer
        from rl_100.common.replay_buffer import ReplayBuffer
        buffer = ReplayBuffer(root=data)
        
        print(f"\nTotal loading time: {elapsed:.1f}s")
        
        # Only test once if it's fast enough
        if elapsed < 30:
            print(f"\nOptimal worker count: {num_workers}")
            break

if __name__ == "__main__":
    test_parallel_loading()
