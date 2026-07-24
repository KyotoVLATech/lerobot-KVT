import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from rl_100.env import DexArtEnv
from rl_100.gym_util.multistep_wrapper import MultiStepWrapper
from rl_100.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from rl_100.policy.base_policy import BasePolicy
from rl_100.common.pytorch_util import dict_apply
from rl_100.env_runner.base_runner import BaseRunner
import rl_100.common.logger_util as logger_util
from rl_100.stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 env_num=10,
                 with_pointcloud=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        def make_env(is_test):
            def _init():
                return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
            return _init
        # self.env_train = env_fn(is_test=False)
        # self.env = env_fn(is_test=True)

        self.env_num = env_num
        self.env = env_fn(is_test=False)
        self.env_fns = [make_env(is_test=False) for _ in range(env_num)]
        self.vec_env = SubprocVecEnv(self.env_fns, 'spawn')



        self.episode_train = int(n_train / env_num)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.vec_env

        all_returns_train = []
        all_success_rates_train = []
        all_first_done_return = []
        all_first_success_rates_train = []
        all_returns = []


        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()
            policy.reset()

            done = False
            reward_sum = np.zeros(self.env_num)
            first_done = np.zeros(self.env_num)
            first_done_success = np.zeros(self.env_num)
            first_done_return = np.zeros(self.env_num)
            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = obs
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    action_dict = policy.predict_action(obs_dict_input)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                for i, entry in enumerate(done):
                    if not first_done[i]:
                        first_done[i] = entry
                        first_done_success[i] = info[i]['success']
                        first_done_return[i] = reward_sum[i]
                done = np.all(done)

                if done:
                    break
            all_returns_train.append(reward_sum)
            for j in range(self.env_num):
                all_success_rates_train.append(info[j]['success'])
            all_first_success_rates_train.append(first_done_success)
            all_first_done_return.append(first_done_return)

       

        SR_mean_train = np.mean(all_success_rates_train)
        fist_mean_SR_train = np.mean(all_first_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)
        first_returns_mean_train = np.mean(all_first_done_return)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates'] = fist_mean_SR_train
        log_data['mean_success_rates_pre'] = SR_mean_train
        log_data['mean_returns'] = first_returns_mean_train
        log_data['mean_returns_pre'] = returns_mean_train

        log_data['test_mean_score'] = fist_mean_SR_train

        self.logger_util_train.record(fist_mean_SR_train)
        self.logger_util_train10.record(fist_mean_SR_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {fist_mean_SR_train:.3f}", 'green')

        # visualize sim
        videos_train = env_train.env_method('get_video')[-1]

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.vec_env

        all_returns_train = []
        all_success_rates_train = []
        all_first_done_return = []
        all_first_success_rates_train = []
        all_returns = []


        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()
            policy.reset()

            done = False
            reward_sum = np.zeros(self.env_num)
            first_done = np.zeros(self.env_num)
            first_done_success = np.zeros(self.env_num)
            first_done_return = np.zeros(self.env_num)
            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = obs
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot']
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                    action_dict = policy.predict_action(obs_dict_input)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                for i, entry in enumerate(done):
                    if not first_done[i]:
                        first_done[i] = entry
                        first_done_success[i] = info[i]['success']
                        first_done_return[i] = reward_sum[i]
                done = np.all(done)

                if done:
                    break
            all_returns_train.append(reward_sum)
            for j in range(self.env_num):
                all_success_rates_train.append(info[j]['success'])
            all_first_success_rates_train.append(first_done_success)
            all_first_done_return.append(first_done_return)

       

        SR_mean_train = np.mean(all_success_rates_train)
        fist_mean_SR_train = np.mean(all_first_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)
        first_returns_mean_train = np.mean(all_first_done_return)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates'] = fist_mean_SR_train
        log_data['mean_success_rates_pre'] = SR_mean_train
        log_data['mean_returns'] = first_returns_mean_train
        log_data['mean_returns_pre'] = returns_mean_train

        log_data['test_mean_score'] = fist_mean_SR_train

        self.logger_util_train.record(fist_mean_SR_train)
        self.logger_util_train10.record(fist_mean_SR_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {fist_mean_SR_train:.3f}", 'green')

        # visualize sim
        videos_train = env_train.env_method('get_video')[-1]

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data