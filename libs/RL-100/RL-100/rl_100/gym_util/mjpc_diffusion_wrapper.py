import gym
import numpy as np
import os
import pytorch3d.ops as torch3d_ops
import torch
from termcolor import cprint
from rl_100.gym_util.mujoco_point_cloud import PointCloudGenerator
from typing import NamedTuple, Any
from dm_env import StepType

ADROIT_PC_TRANSFORM = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(45)), np.sin(np.radians(45))],
                    [0, -np.sin(np.radians(45)), np.cos(np.radians(45))]])

ADROIT_HAND_FINGER_NAMES = {'palm',
            'ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal',
            'mfknuckle', 'mfproximal', 'mfmiddle', 'mfdistal',
            'rfknuckle', 'rfproximal', 'rfmiddle', 'rfdistal',
            'lfknuckle', 'lfproximal', 'lfmiddle', 'lfdistal',
            'thbase', 'thproximal', 'thmiddle', 'thdistal',}

ENV_POINT_CLOUD_CONFIG = {
    
    'adroit_hammer': {
        'min_bound': [-10, -10, -0.099],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_door': {
       'min_bound': [-10, -10, -0.499],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_pen': {
        'min_bound': [-10, -10, -0.79],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['vil_camera'],
        'transform': None,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 0.]),
    },

}


def _resolve_render_device_id(default=0):
    for env_name in ('MUJOCO_EGL_DEVICE_ID', 'EGL_DEVICE_ID'):
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == '':
            continue
        try:
            return int(raw_value)
        except ValueError:
            pass
    return default

def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='uniform'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb
    """
    if num_points == 'all': # use all points
        return point_cloud
    if point_cloud.shape[0] <= num_points:
        # pad with zeros
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], 6))], axis=0)
        return point_cloud
    
    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud
    

class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    observation_pointcloud: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

    
class MujocoPointcloudWrapperAdroit(gym.Wrapper):
    """
    fetch point cloud from mujoco and add it to obs
    """
    def __init__(self, env, env_name:str, use_point_crop=True, with_pointcloud=True):
        super().__init__(env)
        self.env_name = env_name
        self.with_pointcloud = with_pointcloud
        # point cloud cropping
        self.min_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('min_bound', None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('max_bound', None)
        
        self.use_point_crop = use_point_crop
        cprint(f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", 'green')

        
        # point cloud sampling
        self.num_points = ENV_POINT_CLOUD_CONFIG[env_name].get('num_points', 512)
        self.point_sampling_method = ENV_POINT_CLOUD_CONFIG[env_name].get('point_sampling_method', 'uniform')
        if self.with_pointcloud:
            cprint(f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
            assert self.point_sampling_method in ['uniform', 'fps'], \
                f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
            self.pc_generator = PointCloudGenerator(
                sim=env.get_mujoco_sim(),
                cam_names=ENV_POINT_CLOUD_CONFIG[env_name]['cam_names'])
            self.render_device_id = _resolve_render_device_id()
        else:
            cprint("[MujocoPointcloudWrapper] point cloud disabled; using zero placeholder points", 'yellow')
            self.pc_generator = None
            self.render_device_id = None
        self.pc_transform = ENV_POINT_CLOUD_CONFIG[env_name].get('transform', None)
        self.pc_scale = ENV_POINT_CLOUD_CONFIG[env_name].get('scale', None)
        self.pc_offset = ENV_POINT_CLOUD_CONFIG[env_name].get('offset', None)
        self.point_cloud_dim = 6
        self.observation_space = gym.spaces.Dict(dict(self.env.observation_space.spaces))
        self.observation_space.spaces['point_cloud'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_points, self.point_cloud_dim),
            dtype=np.float32
        )
    
    def _empty_point_cloud(self):
        return np.zeros((self.num_points, self.point_cloud_dim), dtype=np.float32)

    def _empty_depth(self):
        depth_space = self.observation_space.spaces.get('depth', None)
        if depth_space is None:
            return None
        return np.zeros(depth_space.shape, dtype=depth_space.dtype)

    def get_point_cloud(self, use_RGB=True):
        if not self.with_pointcloud:
            return self._empty_point_cloud(), self._empty_depth()
        save_img_dir = None
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
            save_img_dir=save_img_dir, device_id=self.render_device_id) # (N, 6), xyz+rgb
        
        
        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
            
        # sampling to fixed number of points
        point_cloud = point_cloud_sampling(point_cloud=point_cloud, 
                                           num_points=self.num_points, 
                                           method=self.point_sampling_method)
        
        if not use_RGB:
            point_cloud = point_cloud[:, :3]
        point_cloud = point_cloud.astype(np.float32, copy=False)
        return point_cloud, depth


    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        point_cloud, depth = self.get_point_cloud()
        
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()
        point_cloud, depth = self.get_point_cloud()
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        return obs_dict
