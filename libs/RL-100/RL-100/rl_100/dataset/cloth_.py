from typing import Dict
import torch
import numpy as np
import copy
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.unidpg.utils import RewardScaling
from scipy.spatial.transform import Rotation as R
from typing import Dict
from termcolor import cprint
from tqdm import tqdm
from copy import deepcopy
import zarr
import threading
from queue import Queue
import time

def compute_return(reward, not_done, gamma: float = 0.99
    ) -> np.ndarray:
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_

def quat_to_rotmtx_batch(quat_batch):
    # Ensure the input is a NumPy array with shape (N, 4) where N is the batch size
    quat_batch = np.array(quat_batch)
    if quat_batch.shape[-1] != 4:
        raise ValueError("Each quaternion must have exactly 4 components [x, y, z, w]")
    # Create a rotation object from the batch of quaternions
    rotation = R.from_quat(quat_batch)
    # Convert the rotation object to a batch of rotation matrices
    rot_matrix_batch = rotation.as_matrix()  # This returns a (N, 3, 3) array of rotation matrices
    return rot_matrix_batch

def rotmtx_to_9d_batch(batch):
    """Transposes a batch of matrices and then flattens them.
    Args:
        batch (numpy.ndarray): A 3D array of shape (batch_size, rows, cols).
    Returns:
        numpy.ndarray: A 2D array of shape (batch_size, rows * cols) with flattened matrices.
    """
    # Transpose the batch of matrices
    transposed_batch = batch.transpose(0, 2, 1)  # Shape becomes (batch_size, cols, rows)
    # Flatten each transposed matrix
    flattened_batch = transposed_batch.reshape(batch.shape[0], -1)  # Shape becomes (batch_size, rows * cols)
    flattened_batch = flattened_batch.astype(np.float32)
    return flattened_batch

def transform_to_9d_batch(transform_array, gripper_pos_array=None):
    """
        transform_array: [batch_size, 7], columns arranged in [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        gripper_pos_array: [batch_size, 1]
    """
    transform_input = np.array(transform_array, dtype=np.float32)
    gripper_pos_array = np.array(gripper_pos_array, dtype=np.float32)
    rotation_matrix = quat_to_rotmtx_batch(transform_input[:,3:7])
    rot_9d_vec = rotmtx_to_9d_batch(rotation_matrix)

    if gripper_pos_array is None:
        source_list = [transform_input[:,0:3], rot_9d_vec]
    else:
        source_list = [transform_input[:,0:3], rot_9d_vec, gripper_pos_array]
    output = np.concatenate(source_list, axis = -1)
    return output

class Cloth(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            scale_strategy=None,
            pre_image_norm=False, 
            rgb_head_shape=(3, 320, 240),
            rgb_right_hand_shape=(3, 320, 240),
            rgb_left_hand_shape=(3, 320, 240),
            use_velocity=False,
            prefetch_size=4,
            num_workers=2,
            ):
        super().__init__()
        self.task_name = task_name
        self.rgb_head_shape = rgb_head_shape
        self.rgb_right_hand_shape = rgb_right_hand_shape
        self.rgb_left_hand_shape = rgb_left_hand_shape
        self.img_shape = {'rgb_head': rgb_head_shape, 'rgb_right_hand': rgb_right_hand_shape, 'rgb_left_hand': rgb_left_hand_shape, 'next_rgb_head': rgb_head_shape, 'next_rgb_right_hand': rgb_right_hand_shape, 'next_rgb_left_hand': rgb_left_hand_shape}
        self.use_velocity = use_velocity
        
        # Use memory-mapped access
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path, mode='r')
        
        # Setup prefetching
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.prefetch_queue = Queue(maxsize=prefetch_size)
        self.index_queue = Queue()
        self.stop_event = threading.Event()
        
        # construct scaled reward and return
        if scale_strategy == 'dynamic':
            print('scaling reward dynamically')
            self.reward_norm = RewardScaling(1, gamma=0.99)
            self.gamma = 0.99
            self.scale_strategy = scale_strategy
            cprint('rewards and returns will be scaled on-the-fly during sampling', 'green')
        else:
            self.reward_norm = None
            self.scale_strategy = None
            self.gamma = 0.99

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # Start prefetch workers
        self.workers = []
        for _ in range(num_workers):
            worker = threading.Thread(target=self._prefetch_worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _prefetch_worker(self):
        """Worker thread that prefetches data"""
        while not self.stop_event.is_set():
            try:
                idx = self.index_queue.get(timeout=0.1)
                if idx is None:
                    break
                    
                # Fetch data
                sample = self.sampler.sample_sequence(idx)
                
                # Process images to float32
                for cam in ['rgb_head', 'rgb_right_hand', 'rgb_left_hand', 
                           'next_rgb_head', 'next_rgb_right_hand', 'next_rgb_left_hand']:
                    sample[cam] = sample[cam].astype(np.float32)
                
                # # Process actions
                # left_action = transform_to_9d_batch(sample['action'][:,:7],sample['action'][:,[7]])
                # right_action = transform_to_9d_batch(sample['action'][:,8:15],sample['action'][:,[15]])
                # sample['action'] = np.concatenate([left_action, right_action], axis=-1)
                # next_left_action = transform_to_9d_batch(sample['next_action'][:,:7],sample['next_action'][:,[7]])
                # next_right_action = transform_to_9d_batch(sample['next_action'][:,8:15],sample['next_action'][:,[15]])
                # sample['next_action'] = np.concatenate([next_left_action, next_right_action], axis=-1)
                
                sample = dict_apply(sample, lambda x: x.astype(np.float32))
                data = self._sample_to_data(sample)
                torch_data = dict_apply(data, torch.from_numpy)
                
                self.prefetch_queue.put((idx, torch_data))
            except:
                continue
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Add index to prefetch queue
        self.index_queue.put(idx)
        
        # Try to get prefetched data
        deadline = time.time() + 5.0  # 5 second timeout
        while time.time() < deadline:
            try:
                prefetch_idx, data = self.prefetch_queue.get(timeout=0.1)
                if prefetch_idx == idx:
                    return data
                else:
                    # Wrong index, put it back
                    self.prefetch_queue.put((prefetch_idx, data))
            except:
                continue
        
        # Fallback: fetch directly if prefetch failed
        sample = self.sampler.sample_sequence(idx)
        for cam in ['rgb_head', 'rgb_right_hand', 'rgb_left_hand', 
                   'next_rgb_head', 'next_rgb_right_hand', 'next_rgb_left_hand']:
            sample[cam] = sample[cam].astype(np.float32)

        # left_action = transform_to_9d_batch(sample['action'][:,:7],sample['action'][:,[7]])
        # right_action = transform_to_9d_batch(sample['action'][:,8:15],sample['action'][:,[15]])
        # sample['action'] = np.concatenate([left_action, right_action], axis=-1)
        # next_left_action = transform_to_9d_batch(sample['next_action'][:,:7],sample['next_action'][:,[7]])
        # next_right_action = transform_to_9d_batch(sample['next_action'][:,8:15],sample['next_action'][:,[15]])
        # sample['next_action'] = np.concatenate([next_left_action, next_right_action], axis=-1)
        # sample = dict_apply(sample, lambda x: x.astype(np.float32))
        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def __del__(self):
        """Cleanup prefetch workers"""
        self.stop_event.set()
        for _ in range(self.num_workers):
            self.index_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=1.0)
    
    # ... rest of the methods are the same as the original Cloth class
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        rgb_head = sample['rgb_head'][:,]
        rgb_right_hand = sample['rgb_right_hand'][:,]
        rgb_left_hand = sample['rgb_left_hand'][:,]
        next_rgb_head = sample['next_rgb_head'][:,]
        next_rgb_right_hand = sample['next_rgb_right_hand'][:,]
        next_rgb_left_hand = sample['next_rgb_left_hand'][:,]
        next_agent_pos = sample['next_state'][:,].astype(np.float32)
        
        if not self.use_velocity:
            agent_pos = np.concatenate([agent_pos[:, :6], agent_pos[:, 12:19], agent_pos[:, [-1]]], axis=-1)
            next_agent_pos = np.concatenate([next_agent_pos[:, :6], next_agent_pos[:, 12:19], next_agent_pos[:, [-1]]], axis=-1)
        
        # Handle reward scaling on-the-fly if needed
        reward = sample['reward'].astype(np.float32)
        return_val = sample.get('return', np.zeros_like(reward)).astype(np.float32)
        
        if self.scale_strategy == 'dynamic' and hasattr(self, 'reward_norm'):
            # Apply dynamic reward scaling
            scaled_reward = reward.copy()
            done = sample['done'].astype(np.bool_)
            
            # Scale rewards dynamically
            temp_norm = deepcopy(self.reward_norm)
            for i in range(len(reward)):
                if done[i]:
                    temp_norm.reset()
                else:
                    scaled_reward[i] = temp_norm(reward[i].item())
            reward = scaled_reward
            
            # Compute return on-the-fly
            return_val = compute_return(reward, 1 - done, self.gamma)
        
        data = {
            'obs': {
                'rgb_head': rgb_head,
                'rgb_right_hand': rgb_right_hand,
                'rgb_left_hand': rgb_left_hand,
                'agent_pos': agent_pos,
            },
            'next_obs': {
                'rgb_head': next_rgb_head,
                'rgb_right_hand': next_rgb_right_hand,
                'rgb_left_hand': next_rgb_left_hand,
                'agent_pos': next_agent_pos,
            }, 
            'reward': reward,
            'not_done': 1. - sample['done'].astype(np.bool_),
            'return': return_val,
            'action': sample['action'].astype(np.float32),
            'next_action': sample['next_action'].astype(np.float32)
        }

        return data
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def get_normalizer(self, mode='limits', **kwargs):
        # Sample-based normalizer to avoid loading all data
        num_samples = min(10000, self.replay_buffer.n_steps)
        
        if self.replay_buffer.n_steps > num_samples:
            indices = np.random.choice(self.replay_buffer.n_steps, size=num_samples, replace=False)
        else:
            indices = np.arange(self.replay_buffer.n_steps)
        
        # Sample data in chunks
        chunk_size = 1000
        agent_pos_list = []
        next_agent_pos_list = []
        action_list = []
        next_action_list = []
        
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            
            # 方案1：使用循环处理每个索引
            for idx in chunk_indices:
                agent_pos_list.append(self.replay_buffer['state'][idx].astype(np.float32))
            state_chunk = np.stack(agent_pos_list)

            # 方案2：如果索引是连续的，使用切片
            if np.all(np.diff(chunk_indices) == 1):  # 检查索引是否连续
                start, end = chunk_indices[0], chunk_indices[-1] + 1
                state_chunk = self.replay_buffer['state'][start:end].astype(np.float32)
            
            next_state_chunk = self.replay_buffer['next_state'][chunk_indices].astype(np.float32)
            
            if not self.use_velocity:
                state_chunk = np.concatenate([state_chunk[:, :6], state_chunk[:, 12:19], state_chunk[:, [-1]]], axis=-1)
                next_state_chunk = np.concatenate([next_state_chunk[:, :6], next_state_chunk[:, 12:19], next_state_chunk[:, [-1]]], axis=-1)
            
            agent_pos_list.append(state_chunk)
            next_agent_pos_list.append(next_state_chunk)
            
            action_chunk = self.replay_buffer['action'][chunk_indices]
            next_action_chunk = self.replay_buffer['next_action'][chunk_indices]
            
            # left_action = transform_to_9d_batch(action_chunk[:,:7], action_chunk[:,[7]])
            # right_action = transform_to_9d_batch(action_chunk[:,8:15], action_chunk[:,[15]])
            action_list.append(action_chunk)
            
            # next_left_action = transform_to_9d_batch(next_action_chunk[:,:7], next_action_chunk[:,[7]])
            # next_right_action = transform_to_9d_batch(next_action_chunk[:,8:15], next_action_chunk[:,[15]])
            next_action_list.append(next_action_chunk)
        
        agent_pos = np.concatenate(agent_pos_list, axis=0)
        next_agent_pos = np.concatenate(next_agent_pos_list, axis=0)
        action = np.concatenate(action_list, axis=0)
        next_action = np.concatenate(next_action_list, axis=0)
        
        data = {
            'action': action,
            'agent_pos': agent_pos,
            'next_action': next_action,
            'next_agent_pos': next_agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer