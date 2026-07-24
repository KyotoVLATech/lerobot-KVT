from typing import Dict, Optional
import torch
import numpy as np
import os
import copy
from rl_100.dataset.cloth import Cloth
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.shared_memory_utils import get_shared_memory_data
from rl_100.common.fast_replay_buffer_parallel import fast_parallel_load_zarr
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.unidpg.utils import RewardScaling
from tqdm import tqdm


class ClothShared(Cloth):
    """
    Cloth dataset that uses shared memory to avoid duplicating data across DDP processes.
    Also uses fast_parallel_load_zarr for faster loading.
    
    Usage:
        1. In main process before spawning:
           from rl_100.common.shared_memory_utils import setup_shared_memory_dataset
           info_path, shm_manager = setup_shared_memory_dataset(zarr_path)
           # Pass info_path to dataset config
           
        2. In each DDP process:
           dataset = ClothShared(zarr_path=zarr_path, shared_memory_info_path=info_path, ...)
    """
    
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
            shared_memory_info_path: Optional[str] = None,
            ):
        # Don't call super().__init__ to avoid loading data twice
        # Instead, manually set attributes
        self.task_name = task_name
        self.rgb_head_shape = rgb_head_shape
        self.rgb_right_hand_shape = rgb_right_hand_shape
        self.rgb_left_hand_shape = rgb_left_hand_shape
        self.img_shape = {'rgb_head': rgb_head_shape, 'rgb_right_hand': rgb_right_hand_shape, 'rgb_left_hand': rgb_left_hand_shape}
        self.use_velocity = use_velocity
        
        # Load data from shared memory if info path provided
        if shared_memory_info_path and os.path.exists(shared_memory_info_path):
            try:
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Loading from shared memory...")
                shared_data, self.shm_manager = get_shared_memory_data(shared_memory_info_path)
                
                data = shared_data
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Successfully loaded from shared memory")
                
                # Check memory usage - should be minimal for non-rank-0 processes
                import psutil
                process = psutil.Process()
                memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Memory usage after shared memory load: {memory_gb:.2f} GB")
                
            except Exception as e:
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Error loading from shared memory: {e}")
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Falling back to fast loader...")
                
                # Fall back to fast loading if shared memory fails
                data = fast_parallel_load_zarr(
                    zarr_path, 
                    num_workers=128
                )
                self.shm_manager = None
        else:
            # Fall back to fast loading if no shared memory
            print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] No shared memory info path, using fast loader...")
            if shared_memory_info_path:
                print(f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] Info path exists? {os.path.exists(shared_memory_info_path)}")
            
            # Load all keys
            data = fast_parallel_load_zarr(
                zarr_path, 
                num_workers=128
            )
            self.shm_manager = None
        
        # Create replay buffer with loaded data
        self.replay_buffer = ReplayBuffer(root=data)
        
        # Continue with rest of initialization (same as parent class)
        # construct scaled reward and return
        if scale_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma=0.99)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            
            self.replay_buffer['reward'] = rewards.reshape(-1, 1)
            
            # if return is not pre-computed
            if 'return' not in self.replay_buffer:
                returns = compute_return(
                    self.replay_buffer['reward'],
                    1 - self.replay_buffer['done'],
                    gamma=0.99
                )
                self.replay_buffer['return'] = returns.reshape(-1, 1)
            else:
                # scale return
                returns = self.replay_buffer['return'].flatten()
                mean_return = returns.mean()
                std_return = returns.std()
                returns = (returns - mean_return) / (std_return + 1e-8)
                self.replay_buffer['return'] = returns.reshape(-1, 1)
                print('pre-computed return is scaled')
        elif scale_strategy == 'number':
            print('scaling reward by number')
            assert 'return' in self.replay_buffer
            rewards = self.replay_buffer['reward'].flatten()
            rewards = rewards / 250
            self.replay_buffer['reward'] = rewards.reshape(-1, 1)
            
            returns = self.replay_buffer['return'].flatten() 
            returns = returns / 250
            self.replay_buffer['return'] = returns.reshape(-1, 1)

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

    def get_normalizer(self, mode='limits', **kwargs):
        from copy import deepcopy
        from rl_100.model.common.normalizer import LinearNormalizer
        
        agent_pos = deepcopy(self.replay_buffer['state'][:,].astype(np.float32)) # (agent_posx2, block_posex3)
        next_agent_pos = deepcopy(self.replay_buffer['next_state'][:,].astype(np.float32)) # (agent_posx2, block_posex3)
        if not self.use_velocity:
            agent_pos = np.concatenate([agent_pos[:, :6], agent_pos[:, 12:19], agent_pos[:, -1:]], axis=-1) # concat left_hand_pos 6, left_gripper_pos 1, right_hand_pos 6, right_gripper_pos 1
            next_agent_pos = np.concatenate([next_agent_pos[:, :6], next_agent_pos[:, 12:19], next_agent_pos[:, -1:]], axis=-1) # concat left_hand_pos 6, left_gripper_pos 1, right_hand_pos 6, right_gripper_pos 1

        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': agent_pos,
            'next_action': self.replay_buffer['next_action'],
            'next_agent_pos': next_agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data
    
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        rgb_head = sample['rgb_head'][:,].astype(np.float32) # (T, H, W, C) - keep original format
        rgb_right_hand = sample['rgb_right_hand'][:,].astype(np.float32) # (T, H, W, C) - keep original format
        rgb_left_hand = sample['rgb_left_hand'][:,].astype(np.float32) # (T, H, W, C) - keep original format
        next_agent_pos = sample['next_state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        
        # DON'T transpose - augmentation expects (T, H, W, C) format
        # rgb images stay as (T, H, W, C)
        
        # help me to remove the dim [6:12] and [19:25] to get the agent_pos without velocity
        if not self.use_velocity:
            agent_pos = np.concatenate([agent_pos[:, :6], agent_pos[:, 12:19], agent_pos[:, [-1]]], axis=-1) # concat left_hand_pos 6, left_gripper_pos 1, right_hand_pos 6, right_gripper_pos 1
            next_agent_pos = np.concatenate([next_agent_pos[:, :6], next_agent_pos[:, 12:19], next_agent_pos[:, [-1]]], axis=-1) # concat left_hand_pos 6, left_gripper_pos 1, right_hand_pos 6, right_gripper_pos 1
        data = {
            'obs': {
                'rgb_head': rgb_head, # T, H, W, C (320, 240, 3)
                'rgb_right_hand': rgb_right_hand, # T, H, W, C (320, 240, 3)
                'rgb_left_hand': rgb_left_hand, # T, H, W, C (320, 240, 3)
                'agent_pos': agent_pos, # T, D_pos
            },
            'next_obs': {
                'agent_pos': next_agent_pos, # T, D_pos
            }, 
            'reward': sample['reward'].astype(np.float32), # T, D_action
            'not_done': 1. - sample['done'].astype(np.bool_), # T, D_action
            'return': sample['return'].astype(np.float32), # T, D_action
            'action': sample['action'].astype(np.float32), # T, D_action
            'next_action': sample['next_action'].astype(np.float32) # T, D_action
        }

        return data
    
    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(10)
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        # 图像数据已经是RGB格式，直接使用
        rgb_head = sample['rgb_head'][:, ].astype(np.float32) 
        rgb_right_hand = sample['rgb_right_hand'][:, ].astype(np.float32) 
        rgb_left_hand = sample['rgb_left_hand'][:, ].astype(np.float32) 
        if not self.use_velocity:
            agent_pos = np.concatenate([agent_pos[:, :6], agent_pos[:, 12:19], agent_pos[:, [-1]]], axis=-1) # concat left_hand_pos 6, left_gripper_pos 1, right_hand_pos 6, right_gripper_pos 1
        
        shape_info = {
        'obs': {
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'rgb_head': (n_obs_steps,) + rgb_head.shape[1:],
            'rgb_right_hand': (n_obs_steps,) + rgb_right_hand.shape[1:],
            'rgb_left_hand': (n_obs_steps,) + rgb_left_hand.shape[1:],
        },
        'action': (n_action_steps, sample['action'].shape[-1]),
        }
        return shape_info
        
    def __del__(self):
        """Clean up shared memory handles when dataset is deleted"""
        if hasattr(self, 'shm_manager') and self.shm_manager is not None:
            # Only close, don't unlink (main process will handle cleanup)
            for shm in self.shm_manager.shm_handles.values():
                try:
                    shm.close()
                except:
                    pass


# Helper function from parent class
def compute_return(reward, not_done, gamma: float = 0.99) -> np.ndarray:
    size_ = len(reward)
    return_ = np.zeros((size_, 1))
    pre_return = 0
    for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
        return_[i] = reward[i] + gamma * pre_return * not_done[i]
        pre_return = return_[i]
    return return_