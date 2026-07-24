import wandb
import numpy as np
import torch
import tqdm
from rl_100.env import make_dmc_env, make_dmc_env_2d
from rl_100.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from rl_100.gym_util.multistep_wrapper import MultiStepWrapper
from rl_100.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from rl_100.policy.base_policy import BasePolicy
from rl_100.common.pytorch_util import dict_apply
from rl_100.env_runner.base_runner import BaseRunner
import rl_100.common.logger_util as logger_util
from termcolor import cprint
import time
from rl_100.stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class DMCRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 env_num=20,
                 seed=42,
                 with_pointcloud=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.with_pointcloud = with_pointcloud
        steps_per_render = max(10 // fps, 1)
        if self.with_pointcloud:
            self.env = make_dmc_env(task_name, n_obs_steps, n_action_steps,
                                    2, seed)
        else:
            self.env = make_dmc_env_2d(task_name, n_obs_steps, n_action_steps,
                                    2, seed)
        def make_env():
            if self.with_pointcloud:
                def _init():
                    return make_dmc_env(task_name, n_obs_steps, n_action_steps,
                                    2, seed)
            else:
                def _init():
                    return make_dmc_env_2d(task_name, n_obs_steps, n_action_steps,
                                    2, seed)
            return _init
        self.env_num = env_num

        self.eval_episodes = int(eval_episodes / env_num)
        self.env_fns = [make_env() for _ in range(env_num)]
        self.vec_env = SubprocVecEnv(self.env_fns, 'spawn')

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.vec_env


        all_first_done_return = []
        all_returns = []

        time1 = time.time()
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            actual_step_count = 0
            episode_reward  = np.zeros(self.env_num)
            first_done = np.zeros(self.env_num)
            first_done_return = np.zeros(self.env_num)
            while not done:
                # create obs dict
                np_obs_dict = obs
                # device transfer
                
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    batch_size = obs_dict['image'].shape[0]
                    if self.with_pointcloud:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    else:
                        obs_dict_input['point_cloud'] = torch.zeros(batch_size, self.n_obs_steps, 512, 3).to(device)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']#.squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                episode_reward = episode_reward + reward
                actual_step_count += 1
                # all_goal_achieved.append(info['goal_achieved']
                for i, entry in enumerate(done):
                    if not first_done[i]:
                        first_done[i] = entry
                        first_done_return[i] = episode_reward[i]

                done = np.all(done)
            all_returns.append(episode_reward)
            all_first_done_return.append(first_done_return)
            # all_success_rates.append(info['goal_achieved'])
        time2 = time.time()
        print('eval time (40 traj.):', time2 - time1)   

        # log
        log_data = dict()

        log_data['test_mean_score'] = np.mean(all_first_done_return)
        log_data['mean_returns'] = np.mean(all_first_done_return)

        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_returns)}", 'green')

        self.logger_util_test.record(np.mean(all_first_done_return))
        self.logger_util_test10.record(np.mean(all_first_done_return))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env_method('get_video')[-1]
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num):
        device = policy.device
        dtype = policy.dtype
        env = self.vec_env


        all_first_done_return = []
        all_returns = []

        time1 = time.time()
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            actual_step_count = 0
            episode_reward  = np.zeros(self.env_num)
            first_done = np.zeros(self.env_num)
            first_done_return = np.zeros(self.env_num)
            while not done:
                # create obs dict
                np_obs_dict = obs
                # device transfer
                
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    if self.with_pointcloud:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    else:
                        obs_dict_input['point_cloud'] = torch.zeros(batch_size, self.n_obs_steps, 512, 3).to(device)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                    action_dict = policy.sample_action(obs_dict_input, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae = use_gae, iql=iql, Q=Q, repeat_num=repeat_num)
            
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']#.squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                episode_reward = episode_reward + reward
                actual_step_count += 1
                # all_goal_achieved.append(info['goal_achieved']
                for i, entry in enumerate(done):
                    if not first_done[i]:
                        first_done[i] = entry
                        first_done_return[i] = episode_reward[i]

                done = np.all(done)
            all_returns.append(episode_reward)
            all_first_done_return.append(first_done_return)
            # all_success_rates.append(info['goal_achieved'])
        time2 = time.time()
        print('eval time (40 traj.):', time2 - time1)   

        # log
        log_data = dict()

        log_data['test_mean_score'] = np.mean(all_first_done_return)
        log_data['mean_returns'] = np.mean(all_first_done_return)

        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_returns)}", 'green')

        self.logger_util_test.record(np.mean(all_first_done_return))
        self.logger_util_test10.record(np.mean(all_first_done_return))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env_method('get_video')[-1]
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data