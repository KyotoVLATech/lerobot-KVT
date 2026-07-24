from typing import Dict
import torch
import numpy as np
import copy
import zarr
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.fast_replay_buffer_parallel import fast_parallel_load_zarr
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.unidpg.utils import RewardScaling
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict
from termcolor import cprint
from tqdm import tqdm
from copy import deepcopy
import os

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
            ):
        super().__init__()
        self.task_name = task_name
        self.rgb_head_shape = rgb_head_shape
        self.rgb_right_hand_shape = rgb_right_hand_shape
        self.rgb_left_hand_shape = rgb_left_hand_shape
        self.img_shape = {'rgb_head': rgb_head_shape, 'rgb_right_hand': rgb_right_hand_shape, 'rgb_left_hand': rgb_left_hand_shape, 'next_rgb_head': rgb_head_shape, 'next_rgb_right_hand': rgb_right_hand_shape, 'next_rgb_left_hand': rgb_left_hand_shape}
        self.use_velocity = use_velocity        
        # Use fast loading with only required keys
        required_keys = ['state', 'action', 'rgb_head', 'rgb_left_hand', 'rgb_right_hand', 'next_rgb_head', 'next_rgb_right_hand', 'next_rgb_left_hand', 'next_state', 'next_action', 'reward', 'done', 'timeout', 'return']
        print("Loading data with fast parallel loader...")
        data = fast_parallel_load_zarr(
            zarr_path, 
            num_workers=128,
            keys=required_keys
        )
        
        self.replay_buffer = ReplayBuffer(root=data)
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=['state', 'action', 'rgb_head', 'rgb_left_hand', 'rgb_right_hand', 'next_rgb_head', 'next_rgb_right_hand', 'next_rgb_left_hand', 'next_state', 'next_action', 'reward', 'done', 'timeout', 'return'])
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
            self.replay_buffer.root['data']['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer.root['data']['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma=0.99)
            self.reward_norm = reward_norm
            cprint('reward and return scaled', 'green')

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
    def reward_scaling(self, scaling_strategy = 'dynamic', gamma = 0.99):
        if scaling_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self.replay_buffer['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma)
        

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

            # 'reward': self.replay_buffer['reward'],
            # 'not_done': 1. - self.replay_buffer['done'],
            # 'return': self.replay_buffer['return'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    
    def _sample_to_data(self, sample):
        # 优化：减少不必要的数据类型转换和切片操作
        agent_pos = sample['state'].astype(np.float32) # 直接转换，避免[:,]切片
        rgb_head = sample['rgb_head'].astype(np.float32) # 直接转换为float32
        rgb_right_hand = sample['rgb_right_hand'].astype(np.float32) 
        rgb_left_hand = sample['rgb_left_hand'].astype(np.float32)
        next_rgb_head = sample['next_rgb_head'].astype(np.float32)
        next_rgb_right_hand = sample['next_rgb_right_hand'].astype(np.float32)
        next_rgb_left_hand = sample['next_rgb_left_hand'].astype(np.float32)
        next_agent_pos = sample['next_state'].astype(np.float32)
        
        # 处理agent_pos的velocity移除（如果需要）
        if not self.use_velocity:
            agent_pos = np.concatenate([agent_pos[:, :6], agent_pos[:, 12:19], agent_pos[:, [-1]]], axis=-1)
            next_agent_pos = np.concatenate([next_agent_pos[:, :6], next_agent_pos[:, 12:19], next_agent_pos[:, [-1]]], axis=-1)
        
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
            'reward': sample['reward'].astype(np.float32),
            'not_done': 1. - sample['done'].astype(np.bool_),
            'return': sample['return'].astype(np.float32),
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
        # pre process action
        shape_info = {
        'obs': {
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'rgb_head': (n_obs_steps,) + rgb_head.shape[1:],
            'rgb_right_hand': (n_obs_steps,) + rgb_right_hand.shape[1:],
            'rgb_left_hand': (n_obs_steps,) + rgb_left_hand.shape[1:],
        },
        'action': (n_action_steps, sample['action'].shape[-1]),
        }
        print(shape_info)
        return shape_info
    def get_all_data(self,) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(range(self.replay_buffer['action'].shape[0]))
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_length(self, ):
        return len(self.sampler.indices)
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 直接获取sample，减少中间步骤
        # time_start = time.time()
        sample = self.sampler.sample_sequence(idx)
        # time1 = time.time()
        # print('time1 :{}'.format(time1 - time_start))
        data = self._sample_to_data(sample)
        # print('time2: {}'.format(time.time()- time1))
        torch_data = dict_apply(data, torch.from_numpy)
        # time_end = time.time()
        # print(f"Time taken for __getitem__: {time_end - time_start} seconds")
        return torch_data

