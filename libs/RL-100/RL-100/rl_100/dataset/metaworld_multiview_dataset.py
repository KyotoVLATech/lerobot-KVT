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
from tqdm import tqdm
from termcolor import cprint

def compute_return(reward, not_done, gamma: float = 0.99) -> np.ndarray:
    size_ = len(reward)
    return_ = np.zeros((size_, 1))
    pre_return = 0
    for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
        return_[i] = reward[i] + gamma * pre_return * not_done[i]
        pre_return = return_[i]
    return return_

class MetaworldMultiViewDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            scale_strategy=None,
            pre_image_norm=False,
            img_shape_meta=None,
            ):
        super().__init__()
        
        # 从img_shape_meta中提取图像键名
        if img_shape_meta is None:
            # 默认图像键名（向后兼容）
            image_keys = ['image_gripperPOV', 'image_corner2', 'image_topview']
        else:
            # 从img_shape_meta中提取所有图像相关的键
            image_keys = [key for key in img_shape_meta.keys() if key.startswith('image_')]
        
        # 动态构建所有需要的键
        keys = ['state', 'action', 'next_state', 'next_action', 'reward', 'done', 'full_state', 'return']
        
        # 添加图像键和对应的next图像键
        for img_key in image_keys:
            keys.append(img_key)
            keys.append(f'next_{img_key}')
        
        self.image_keys = image_keys
        self.img_shape_meta = img_shape_meta
        
        # 打印调试信息
        cprint(f"Detected image keys: {image_keys}", "cyan")
        cprint(f"Loading data with keys: {keys}", "cyan")
        
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        
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
        
    def reward_scaling(self, scaling_strategy='dynamic', gamma=0.99):
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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'next_action': self.replay_buffer['next_action'],
            'next_agent_pos': self.replay_buffer['next_state'][...,:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def get_length(self) -> int:
        """Get dataset length - required by train.py"""
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        next_agent_pos = sample['next_state'][:,].astype(np.float32)
        
        # 动态处理多视角图像
        obs_dict = {'agent_pos': agent_pos}
        next_obs_dict = {'agent_pos': next_agent_pos}
        
        # 根据image_keys动态加载图像数据
        for img_key in self.image_keys:
            if img_key in sample:
                obs_dict[img_key] = sample[img_key][:,].astype(np.float32)
            else:
                cprint(f"Warning: {img_key} not found in sample data", "yellow")
            
            next_img_key = f'next_{img_key}'
            if next_img_key in sample:
                next_obs_dict[img_key] = sample[next_img_key][:,].astype(np.float32)
            else:
                cprint(f"Warning: {next_img_key} not found in sample data", "yellow")
        
        data = {
            'obs': obs_dict,
            'next_obs': next_obs_dict,
            'reward': sample['reward'].astype(np.float32),
            'not_done': 1. - sample['done'].astype(np.bool_),
            'return': sample['return'].astype(np.float32),
            'action': sample['action'].astype(np.float32),
            'next_action': sample['next_action'].astype(np.float32)
        }
        return data

    def get_shape_info(self, horizon, n_obs_steps):
        """Get shape information for the dataset"""
        sample = self.sampler.sample_sequence(0)
        data = self._sample_to_data(sample)
        
        obs_shape = {}
        for key, value in data['obs'].items():
            obs_shape[key] = value.shape[1:]  # Remove batch dimension
            
        action_shape = data['action'].shape[1:]
        
        return {
            'obs': obs_shape,
            'action': action_shape
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data