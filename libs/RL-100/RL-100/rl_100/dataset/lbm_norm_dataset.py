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

from termcolor import cprint
from tqdm import tqdm
def compute_return(reward, not_done, gamma: float == 0.99
    ) -> np.ndarray:
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_
class LBMNormDataset(BaseDataset):
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
            use_img=False
            ):
        super().__init__()
        self.task_name = task_name
        self.use_img = use_img
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['ee_pose_quat', 'action_quat', 'point_cloud', 'next_ee_pose_quat', 'next_action_quat', 'next_point_cloud', 'reward', 'done', 'timeout', 'return'])
        # construct scaled reward and return
        # import pdb; pdb.set_trace()
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
        # self.replay_buffer.root {'meta', 'data'}

        # for key, value in self.replay_buffer.items():
        #     cprint(f'Replay Buffer: {key}, shape {value.shape}, dtype {value.dtype}, range {value.min():.2f}~{value.max():.2f}', 'green')
        # cprint("--------------------------", 'green')
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
        
        full_val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=0,
            seed=seed)
        zero_train_mask = ~full_val_mask
        zero_train_mask = downsample_mask(
            mask=zero_train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.full_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=zero_train_mask)

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
        all_data = self.get_all_action_chunk()
        all_action_chunk = all_data['action']
        data = {
            'action': all_action_chunk,
            'agent_pos': self.replay_buffer['ee_pose_quat'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],

            'next_action': self.replay_buffer['next_action_quat'],
            'next_agent_pos': self.replay_buffer['next_ee_pose_quat'][...,:],
            'next_point_cloud': self.replay_buffer['next_point_cloud'],

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
        agent_pos = sample['ee_pose_quat'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # image = sample['img'][:,].astype(np.float32) # (T, 3, 64, 64)

        # image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image
        
        next_agent_pos = sample['next_ee_pose_quat'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        next_point_cloud = sample['next_point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # next_image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                # 'image': image, # T, 84, 84, 3
            },
            'next_obs': {
                'point_cloud': next_point_cloud, # T, 1024, 6
                'agent_pos': next_agent_pos, # T, D_pos
                # 'image': next_image, # T, 84, 84, 3
            }, 
            'reward': sample['reward'].astype(np.float32), # T, D_action
            'not_done': 1. - sample['done'].astype(np.bool_), # T, D_action
            'return': sample['return'].astype(np.float32), # T, D_action
            'action': sample['action_quat'].astype(np.float32)-sample['ee_pose_quat'][[0],:], # T, D_action
            'next_action': sample['next_action_quat'].astype(np.float32)-sample['next_ee_pose_quat'][[0],:] # T, D_action
        }

        return data
    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(10)
        agent_pos = sample['ee_pose_quat'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image
        # image = sample['img'][:,].astype(np.float32) # (T, 3, 64, 64)
 
        shape_info = {
        'obs': {
            'point_cloud': (n_obs_steps,) + point_cloud.shape[1:],
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'image': (n_obs_steps,) + image.shape[1:],
        },
        'action': (n_action_steps, sample['action_quat'].shape[-1]),
        }
        return shape_info
    def get_all_action_chunk(self,) -> Dict[str, torch.Tensor]:
        # 获取所有采样的索引
        all_samples = [self.full_sampler.sample_sequence(i) for i in range(len(self.full_sampler))]
        # 合并所有样本
        # 这里假设每个sample是dict，且每个key的value都是numpy数组，可以直接拼接
        merged = {}
        merged['action_quat'] = np.stack([sample['action_quat'] for sample in all_samples], axis=0)
        merged['ee_pose_quat'] = np.stack([sample['ee_pose_quat'] for sample in all_samples], axis=0)
        data = {'action': merged['action_quat'].astype(np.float32)-merged['ee_pose_quat'][:,[0],:]}
        return data

    def get_length(self, ):
        return len(self.sampler.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

