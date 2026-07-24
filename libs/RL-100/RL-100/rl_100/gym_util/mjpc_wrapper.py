import gym
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
import os

from termcolor import cprint
from rl_100.gym_util.mujoco_point_cloud import PointCloudGenerator
from typing import NamedTuple, Any
from dm_env import StepType

ADROIT_PC_TRANSFORM = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(45)), np.sin(np.radians(45))],
                    [0, -np.sin(np.radians(45)), np.cos(np.radians(45))]])

ENV_POINT_CLOUD_CONFIG = {
    # adroit
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


def _count_egl_devices():
    """Count available EGL devices via ctypes. Returns None on failure."""
    try:
        import ctypes
        from ctypes import pointer, c_int, c_void_p, POINTER, CFUNCTYPE
        egl = ctypes.cdll.LoadLibrary('libEGL.so.1')
        egl.eglGetProcAddress.restype = c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        addr = egl.eglGetProcAddress(b'eglQueryDevicesEXT')
        if not addr:
            return None
        FUNC = CFUNCTYPE(c_int, c_int, POINTER(c_void_p), POINTER(c_int))
        query_fn = FUNC(addr)
        max_devices = 16
        devices = (c_void_p * max_devices)()
        num_devices = c_int(0)
        query_fn(max_devices, devices, pointer(num_devices))
        return num_devices.value if num_devices.value > 0 else None
    except Exception:
        return None


def _resolve_render_device_id(default=0):
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    for env_name in ('MUJOCO_EGL_DEVICE_ID', 'EGL_DEVICE_ID'):
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == '':
            continue
        try:
            device_id = int(raw_value)
            num_egl = _count_egl_devices()
            if num_egl is not None and device_id >= num_egl:
                return 0
            return device_id
        except ValueError:
            pass
    if cuda_vis:
        first_gpu = int(cuda_vis.split(',')[0])
        num_egl = _count_egl_devices()
        if num_egl is not None and first_gpu < num_egl:
            return first_gpu
    return default

def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='fps'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == 'all': # use all points
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
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
    observation_depth: Any
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
    def __init__(self, env, env_name:str, use_point_crop=True):
        super().__init__(env)
        self.env_name = env_name
        # point cloud cropping
        self.min_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('min_bound', None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('max_bound', None)

        self.use_point_crop = use_point_crop
        cprint(f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", 'green')
        
        # point cloud sampling
        self.num_points = ENV_POINT_CLOUD_CONFIG[env_name].get('num_points', 512)
        self.point_sampling_method = ENV_POINT_CLOUD_CONFIG[env_name].get('point_sampling_method', 'uniform')
        cprint(f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
        assert self.point_sampling_method in ['uniform', 'fps'], \
            f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
        
        # point cloud generator
        self.pc_generator = PointCloudGenerator(sim=env.get_mujoco_sim(),
                                                cam_names=ENV_POINT_CLOUD_CONFIG[env_name]['cam_names'])
        self.render_device_id = _resolve_render_device_id()
        self.pc_transform = ENV_POINT_CLOUD_CONFIG[env_name].get('transform', None)
        self.pc_scale = ENV_POINT_CLOUD_CONFIG[env_name].get('scale', None)
        self.pc_offset = ENV_POINT_CLOUD_CONFIG[env_name].get('offset', None)

    

    def get_point_cloud(self, use_RGB=True):

        # set save_img_dir to save images for debugging
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
        return point_cloud, depth


    def step(self, action):
        timestep = self.env.step(action)
        point_cloud, depth = self.get_point_cloud()
        
        # wrap point cloud into obs
        if 'adroit' in self.env_name: # adroit uses a namedtuple for obs
            # so we need to create a new namedtuple
            timestep = ExtendedTimeStepAdroit(step_type=timestep.step_type,
                                         reward=timestep.reward,
                                         discount=timestep.discount,
                                         observation=timestep.observation,
                                         observation_sensor=timestep.observation_sensor,
                                         observation_pointcloud=point_cloud,
                                         observation_depth=depth,
                                         action=timestep.action,
                                         n_goal_achieved=timestep.n_goal_achieved,
                                         time_limit_reached=timestep.time_limit_reached)                        
        else:
            raise NotImplementedError
        return timestep

    def reset(self):
        timestep = self.env.reset()
        point_cloud, depth = self.get_point_cloud()
        
        # wrap point cloud into obs
        if 'adroit' in self.env_name: # adroit uses a namedtuple for obs
            # so we need to create a new namedtuple
            timestep = ExtendedTimeStepAdroit(step_type=timestep.step_type,
                                         reward=timestep.reward,
                                         discount=timestep.discount,
                                         observation=timestep.observation,
                                         observation_sensor=timestep.observation_sensor,
                                         observation_pointcloud=point_cloud,
                                         observation_depth=depth,
                                         action=timestep.action,
                                         n_goal_achieved=timestep.n_goal_achieved,
                                         time_limit_reached=timestep.time_limit_reached)                        
        else:
            raise NotImplementedError
        return timestep


    


