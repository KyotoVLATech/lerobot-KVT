import wandb
import numpy as np
import torch
import collections
import tqdm
from rl_100.env import MetaWorldEnv
from rl_100.gym_util.multistep_wrapper import MultiStepWrapper
from rl_100.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from rl_100.policy.base_policy import BasePolicy
from rl_100.common.pytorch_util import dict_apply
from rl_100.env_runner.base_runner import BaseRunner
import rl_100.common.logger_util as logger_util
from termcolor import cprint
from rl_100.stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

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
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points, rgb_size=84)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        def make_env(task_name):
            def _init():
                return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points, rgb_size=84)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
            return _init
        self.env_num = env_num
        self.eval_episodes = int(eval_episodes / env_num)
        self.env_fns = [make_env(self.task_name) for _ in range(env_num)]
        self.vec_env = SubprocVecEnv(self.env_fns, 'spawn')
        self.env = env_fn(self.task_name)

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
        env = self.vec_env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()
            indx = 0
            done = False
            traj_reward = 0
            is_success = [False] * self.env_num
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']
                indx += 1
                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.any(done)
                for i in range(self.env_num):
                    is_success[i] = is_success[i] or max(info[i]['success'])
            print(f"indx: {indx}")
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
        

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        
        # if save_video:
        #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        #     log_data[f'sim_video_eval'] = videos_wandb

        # _ = env.reset()
        # videos = None

        return log_data

    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.vec_env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()
            indx = 0
            done = False
            traj_reward = 0
            is_success = [False] * self.env_num
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                    action_dict = policy.sample_action(obs_dict_input, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae = use_gae, iql=iql, Q=Q, repeat_num=repeat_num)
            
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']

                obs, reward, done, info = env.step(action)
                indx += 1

                traj_reward += reward
                done = np.all(done)
                for i in range(self.env_num):
                    is_success[i] = is_success[i] or max(info[i]['success'])
            print(f"indx: {indx}")
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
        

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        
        # if save_video:
        #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        #     log_data[f'sim_video_eval'] = videos_wandb

        # _ = env.reset()
        # videos = None

        return log_data
