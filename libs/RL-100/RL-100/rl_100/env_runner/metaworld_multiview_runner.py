import wandb
import numpy as np
import torch
import collections
import tqdm
from rl_100.env import MetaWorldEnv, MetaWorldMultiViewEnv
from rl_100.gym_util.multistep_wrapper import MultiStepWrapper
from rl_100.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from rl_100.policy.base_policy import BasePolicy
from rl_100.common.pytorch_util import dict_apply
from rl_100.env_runner.base_runner import BaseRunner
import rl_100.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 env_num=1,
                 with_pointcloud=True,
                 img_shape_meta=None,
                 record_video=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.with_pointcloud = with_pointcloud
        self.record_video = record_video
        
        # 从img_shape_meta中提取图像键名和相机名称
        if img_shape_meta is None:
            # 默认相机配置（向后兼容）
            self.image_keys = ['image_corner', 'image_corner2', 'image_gripper']
            camera_names = ['corner', 'corner2', 'gripperPOV']
        else:
            # 从img_shape_meta中提取所有图像相关的键
            self.image_keys = [key for key in img_shape_meta.keys() if key.startswith('image_')]
            # 从图像键名中提取相机名称（去掉'image_'前缀）
            camera_names = [key.replace('image_', '') for key in self.image_keys]
        
        self.img_shape_meta = img_shape_meta
        cprint(f"MetaworldRunner - Detected image keys: {self.image_keys}", "cyan")
        cprint(f"MetaworldRunner - Camera names: {camera_names}", "cyan")

        def env_fn(task_name, record_video=False):
            inner_env = MetaWorldMultiViewEnv(
                task_name=task_name, device=device,
                rgb_size=render_size, camera_names=camera_names)
            if record_video:
                inner_env = SimpleVideoRecordingWrapper(inner_env)
            return MultiStepWrapper(
                inner_env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name, record_video=self.record_video)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    if self.with_pointcloud:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    
                    # 动态处理多视角图像 - 将相机名称转换为image_前缀格式
                    for img_key in self.image_keys:
                        # 从image_key中提取相机名称
                        camera_name = img_key.replace('image_', '')
                        if camera_name in obs_dict:
                            # 使用image_前缀的键名传递给policy
                            obs_dict_input[img_key] = obs_dict[camera_name].unsqueeze(0).to(torch.float)
                        else:
                            cprint(f"Warning: camera {camera_name} not found in obs_dict", "yellow")
                    
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_returns'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data

    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    if self.with_pointcloud:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    
                    # 动态处理多视角图像 - 将相机名称转换为image_前缀格式
                    for img_key in self.image_keys:
                        # 从image_key中提取相机名称
                        camera_name = img_key.replace('image_', '')
                        if camera_name in obs_dict:
                            # 使用image_前缀的键名传递给policy
                            obs_dict_input[img_key] = obs_dict[camera_name].unsqueeze(0).to(torch.float)
                        else:
                            cprint(f"Warning: camera {camera_name} not found in obs_dict", "yellow")
                    
                    action_dict = policy.sample_action(obs_dict_input, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae = use_gae, iql=iql, Q=Q, repeat_num=repeat_num)
            
                    

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_returns'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        if self.record_video and save_video:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data['sim_video_eval'] = videos_wandb

        if self.record_video:
            _ = env.reset()

        return log_data

