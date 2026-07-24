import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import pickle
import os
import tempfile
from typing import Dict, Any, Optional
import zarr
from rl_100.common.fast_replay_buffer_parallel import fast_parallel_load_zarr


class SharedMemoryManager:
    """Manager for shared memory arrays across multiple processes"""
    
    def __init__(self):
        self.shm_handles = {}
        self.array_info = {}
        self.meta_info = None
        
    def create_from_zarr(self, zarr_path: str, num_workers: int = 128, keys: list = None) -> Dict[str, Any]:
        """Load zarr data into shared memory"""
        print(f"Loading data into shared memory from: {zarr_path}")
        if keys:
            print(f"Loading only keys: {keys}")
        
        # Load data using existing fast loader
        loaded_data = fast_parallel_load_zarr(zarr_path, num_workers=num_workers, keys=keys)
        
        # Store metadata separately
        self.meta_info = loaded_data.get('meta', {})
        
        # Create shared memory for each array
        shared_data = {'meta': self.meta_info, 'data': {}}
        
        for key, array in loaded_data['data'].items():
            # Create shared memory for this array
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            
            # Copy data to shared memory
            shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shared_array[:] = array[:]
            
            # Store handle and info
            self.shm_handles[key] = shm
            self.array_info[key] = {
                'shape': array.shape,
                'dtype': array.dtype,
                'shm_name': shm.name
            }
            
            # Add to shared data dict
            shared_data['data'][key] = shared_array
            
            print(f"  Created shared memory for {key}: {array.shape}")
        
        return shared_data
    
    def get_from_shared_memory(self) -> Dict[str, Any]:
        """Attach to existing shared memory arrays"""
        shared_data = {'meta': self.meta_info, 'data': {}}
        
        for key, info in self.array_info.items():
            try:
                # Attach to existing shared memory
                shm = shared_memory.SharedMemory(name=info['shm_name'])
                
                # Create array view
                shared_array = np.ndarray(
                    info['shape'], 
                    dtype=info['dtype'], 
                    buffer=shm.buf
                )
                
                shared_data['data'][key] = shared_array
                self.shm_handles[key] = shm
                
            except FileNotFoundError:
                print(f"Warning: Shared memory segment {info['shm_name']} not found for key {key}")
                raise RuntimeError(f"Shared memory not available for key {key}")
            except Exception as e:
                print(f"Error attaching to shared memory for key {key}: {e}")
                raise
        
        return shared_data
    
    def cleanup(self):
        """Clean up shared memory"""
        for key, shm in self.shm_handles.items():
            try:
                shm.close()
                if hasattr(shm, '_name'):  # Only unlink if we created it
                    shm.unlink()
            except FileNotFoundError:
                # Already cleaned up
                pass
            except Exception as e:
                print(f"Warning: Failed to cleanup shared memory for {key}: {e}")
                pass
    
    def save_info(self, filepath: str):
        """Save array info and metadata for other processes"""
        info = {
            'array_info': self.array_info,
            'meta_info': self.meta_info
        }
        with open(filepath, 'wb') as f:
            pickle.dump(info, f)
    
    def load_info(self, filepath: str):
        """Load array info and metadata"""
        with open(filepath, 'rb') as f:
            info = pickle.load(f)
        self.array_info = info['array_info']
        self.meta_info = info['meta_info']


def setup_shared_memory_dataset(zarr_path: str, info_path: str = None, keys: list = None) -> tuple:
    """
    Setup shared memory for dataset. Should be called once before spawning processes.
    
    Args:
        zarr_path: Path to zarr dataset
        info_path: Optional path for info file
        keys: Optional list of keys to load. If None, loads all keys.
    
    Returns:
        info_path: Path to the info file that workers should use
        shm_manager: SharedMemoryManager instance (keep alive during training)
    """
    if info_path is None:
        # Create temporary file for sharing info
        temp_dir = tempfile.gettempdir()
        info_path = os.path.join(temp_dir, f'ddp_dataset_info_{os.getpid()}.pkl')
    
    # Create shared memory manager and load data
    shm_manager = SharedMemoryManager()
    shared_data = shm_manager.create_from_zarr(zarr_path, keys=keys)
    
    # Save info for worker processes
    shm_manager.save_info(info_path)
    
    return info_path, shm_manager


def get_shared_memory_data(info_path: str) -> tuple:
    """
    Get shared memory data in worker process.
    
    Returns:
        shared_data: Dictionary with shared memory arrays
        shm_manager: SharedMemoryManager instance for this process
    """
    shm_manager = SharedMemoryManager()
    shm_manager.load_info(info_path)
    shared_data = shm_manager.get_from_shared_memory()
    
    return shared_data, shm_manager