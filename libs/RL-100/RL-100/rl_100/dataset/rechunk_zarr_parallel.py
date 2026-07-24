#!/usr/bin/env python3
"""
Parallel rechunking for zarr datasets.
Much faster for large image datasets.
"""

import zarr
import numpy as np
from tqdm import tqdm
import os
import shutil
import multiprocessing as mp
from functools import partial
import time
import numcodecs
import tempfile

RL100_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def copy_array_chunk(args):
    """Worker function to copy a chunk of an array"""
    src_path, dst_path, key, start_idx, end_idx, shape, dtype, chunks, compressor_config = args
    
    # Open zarr arrays in each process
    src = zarr.open(src_path, 'r')
    dst = zarr.open(dst_path, 'r+')
    
    # Reconstruct compressor
    if compressor_config:
        compressor = numcodecs.get_codec(compressor_config)
    else:
        compressor = None
    
    # Get arrays
    src_arr = src['data'][key]
    dst_arr = dst['data'][key]
    
    # Copy chunk
    if len(shape) == 4:  # Image data
        dst_arr[start_idx:end_idx] = src_arr[start_idx:end_idx]
    else:  # Other data
        dst_arr[start_idx:end_idx] = src_arr[start_idx:end_idx]
    
    return end_idx - start_idx

def parallel_copy_array(src_path, dst_path, key, shape, dtype, chunks, compressor, num_workers=None):
    """Copy a single array in parallel"""
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # For small arrays, use single process
    array_size_mb = np.prod(shape) * np.dtype(dtype).itemsize / (1024**2)
    if array_size_mb < 100:  # < 100MB
        src = zarr.open(src_path, 'r')
        dst = zarr.open(dst_path, 'r+')
        dst['data'][key][:] = src['data'][key][:]
        return
    
    # Determine chunk size for parallel processing
    total_items = shape[0]
    
    # For image arrays, process in batches of images
    if len(shape) == 4:
        # Process in chunks aligned with destination chunks
        items_per_job = chunks[0] * 2  # Process 2 destination chunks at a time
    else:
        items_per_job = max(1000, total_items // (num_workers * 4))
    
    # Create job list
    jobs = []
    compressor_config = compressor.get_config() if compressor else None
    
    for start_idx in range(0, total_items, items_per_job):
        end_idx = min(start_idx + items_per_job, total_items)
        jobs.append((
            src_path, dst_path, key, start_idx, end_idx, 
            shape, dtype, chunks, compressor_config
        ))
    
    # Process in parallel
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_items, desc=f"Copying {key}", unit='items') as pbar:
            for items_copied in pool.imap_unordered(copy_array_chunk, jobs):
                pbar.update(items_copied)

def fast_rechunk_dataset(input_path: str, output_path: str, num_workers: int = None):
    """
    Fast parallel rechunking optimized for cloth dataset.
    """
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers
    
    print(f"Fast rechunking using {num_workers} workers")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Optimized chunks for cloth dataset
    chunk_configs = {
        # Image arrays: 100 images per chunk for ~73MB chunks
        'rgb_head': (100, 320, 240, 3),
        'rgb_left_hand': (100, 320, 240, 3),
        'rgb_right_hand': (100, 320, 240, 3),
        'next_rgb_head': (100, 320, 240, 3),
        'next_rgb_left_hand': (100, 320, 240, 3),
        'next_rgb_right_hand': (100, 320, 240, 3),
        
        # State/action arrays: larger chunks
        'state': (10000, 26),
        'next_state': (10000, 26),
        'action': (10000, 26),
        'next_action': (10000, 26),
        
        # Small arrays: single chunk
        'done': (126213, 1),
        'reward': (50000, 1),
        'return': (50000, 1),
        'timeout': (126213, 1),
    }
    
    # Fast compressor
    compressor = numcodecs.Blosc(cname='lz4', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
    
    # Check if output exists
    if os.path.exists(output_path):
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_path)
    
    # Open source
    src = zarr.open(input_path, 'r')
    
    # Create destination structure
    dst = zarr.open(output_path, 'w')
    
    # Copy metadata (small, do it serially)
    if 'meta' in src:
        print("\nCopying metadata...")
        src_meta = src['meta']
        dst_meta = dst.create_group('meta')
        
        for key in src_meta.keys():
            arr = src_meta[key]
            dst_meta.create_dataset(
                key, 
                data=arr[:], 
                chunks=arr.shape,
                compressor=compressor
            )
    
    # Create data group and arrays
    if 'data' in src:
        src_data = src['data']
        dst_data = dst.create_group('data')
        
        # First pass: create all arrays with proper structure
        print("\nCreating array structure...")
        array_infos = {}
        
        for key in src_data.keys():
            src_arr = src_data[key]
            shape = src_arr.shape
            dtype = src_arr.dtype
            
            # Get chunks
            if key in chunk_configs:
                chunks = chunk_configs[key]
            else:
                # Default chunking
                if len(shape) == 4:  # Image
                    chunks = (100, shape[1], shape[2], shape[3])
                elif len(shape) == 2:
                    chunks = (min(10000, shape[0]), shape[1])
                else:
                    chunks = shape
            
            # Create array
            dst_data.create_dataset(
                key,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor
            )
            
            array_infos[key] = {
                'shape': shape,
                'dtype': dtype,
                'chunks': chunks,
                'size_mb': np.prod(shape) * np.dtype(dtype).itemsize / (1024**2)
            }
            
            print(f"  {key}: {shape} -> chunks {chunks} ({array_infos[key]['size_mb']:.1f} MB)")
    
    # Close and reopen to ensure structure is saved
    del src
    del dst
    
    # Second pass: copy data in parallel
    print(f"\nCopying data using {num_workers} workers...")
    start_time = time.time()
    
    # Sort by size - process large arrays first
    sorted_keys = sorted(array_infos.keys(), key=lambda k: array_infos[k]['size_mb'], reverse=True)
    
    # Process arrays
    for key in sorted_keys:
        info = array_infos[key]
        print(f"\nProcessing {key} ({info['size_mb']:.1f} MB)...")
        
        # Use parallel copy for large arrays
        if info['size_mb'] > 100:
            parallel_copy_array(
                input_path, output_path, key,
                info['shape'], info['dtype'], info['chunks'],
                compressor, num_workers
            )
        else:
            # Small arrays - copy directly
            src = zarr.open(input_path, 'r')
            dst = zarr.open(output_path, 'r+')
            dst['data'][key][:] = src['data'][key][:]
            print(f"  Copied {key}")
    
    elapsed = time.time() - start_time
    
    # Calculate sizes
    src_size_gb = sum(os.path.getsize(os.path.join(dirpath, f))
                      for dirpath, _, files in os.walk(input_path)
                      for f in files) / (1024**3)
    dst_size_gb = sum(os.path.getsize(os.path.join(dirpath, f))
                      for dirpath, _, files in os.walk(output_path)
                      for f in files) / (1024**3)
    
    total_data_gb = sum(info['size_mb'] for info in array_infos.values()) / 1024
    
    print(f"\n{'='*60}")
    print("Rechunking complete!")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Speed: {total_data_gb / elapsed:.1f} GB/s")
    print(f"Original size: {src_size_gb:.2f} GB")
    print(f"New size: {dst_size_gb:.2f} GB")
    print(f"Compression ratio: {src_size_gb/dst_size_gb:.2f}x")
    print(f"Output: {output_path}")


def benchmark_workers():
    """Test different numbers of workers to find optimal setting"""
    
    input_path = os.path.join(RL100_ROOT, "data", "cloth_rgb_processed_action.zarr")
    
    print("Benchmarking optimal worker count...")
    print("Testing with first image array only...")
    
    # Find first image array
    src = zarr.open(input_path, 'r')
    image_key = None
    for key in src['data'].keys():
        if src['data'][key].shape[-1] == 3 and len(src['data'][key].shape) == 4:
            image_key = key
            break
    
    if not image_key:
        print("No image array found")
        return
    
    print(f"Testing with {image_key}: {src['data'][image_key].shape}")
    
    # Test different worker counts
    for num_workers in [1, 2, 4, 8, 16]:
        test_output = os.path.join(tempfile.gettempdir(), f"rechunk_test_{num_workers}.zarr")
        
        if os.path.exists(test_output):
            shutil.rmtree(test_output)
        
        # Create minimal structure
        dst = zarr.open(test_output, 'w')
        dst_data = dst.create_group('data')
        
        src_arr = src['data'][image_key]
        dst_data.create_dataset(
            image_key,
            shape=src_arr.shape,
            chunks=(100, 320, 240, 3),
            dtype=src_arr.dtype,
            compressor=numcodecs.Blosc(cname='lz4', clevel=1)
        )
        
        del dst
        
        # Time the copy
        start = time.time()
        parallel_copy_array(
            input_path, test_output, image_key,
            src_arr.shape, src_arr.dtype, (100, 320, 240, 3),
            numcodecs.Blosc(cname='lz4', clevel=1),
            num_workers
        )
        elapsed = time.time() - start
        
        size_gb = src_arr.nbytes / (1024**3)
        speed = size_gb / elapsed
        
        print(f"Workers: {num_workers:2d} - Time: {elapsed:6.1f}s - Speed: {speed:6.1f} GB/s")
        
        # Cleanup
        shutil.rmtree(test_output)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_workers()
    else:
        # Default paths for cloth dataset
        input_path = os.path.join(RL100_ROOT, "data", "cloth_rgb_processed_action.zarr")
        output_path = os.path.join(RL100_ROOT, "data", "cloth_rgb_processed_action_rechunked.zarr")
        
        # Check worker count argument
        num_workers = None
        if len(sys.argv) > 1:
            try:
                num_workers = int(sys.argv[1])
                print(f"Using {num_workers} workers")
            except:
                pass
        
        fast_rechunk_dataset(input_path, output_path, num_workers)
