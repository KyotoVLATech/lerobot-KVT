import wandb
import numpy as np
import torch
import tqdm
from rl_100.env import AdroitEnv
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



class AdroitRunner(BaseRunner):
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
                 with_pointcloud=True,
                 eval_seed=0,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        if 'pen' in task_name:
            self.success_threshold = 20
        else:
            self.success_threshold = 25

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        def make_env():
            def _init():
                return MultiStepWrapper(
                        SimpleVideoRecordingWrapper(
                            MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                        env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_steps,
                        reward_agg_method='sum',
                    )
            return _init
        self.env_num = env_num
        self.eval_episodes = int(eval_episodes / env_num)
        self.env = env_fn()
        self.env_fns = [make_env() for _ in range(env_num)]
        self.vec_env = SubprocVecEnv(self.env_fns, 'spawn')

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.eval_seed = eval_seed

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def _save_rng_states(self):
        state = {
            'numpy': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_states(self, state):
        np.random.set_state(state['numpy'])
        torch.random.set_rng_state(state['torch'])
        if 'cuda' in state:
            torch.cuda.set_rng_state_all(state['cuda'])

    def _seed_eval_episode(self, episode_idx):
        seed = int(self.eval_seed + episode_idx * self.env_num)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.vec_env.seed(seed)
        return seed

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.vec_env

        all_goal_achieved = []
        all_success_rates = []
        all_first_done_return = []
        all_returns = []
        hard_success = 0
        first_done_hard_success = 0 

        rng_state = self._save_rng_states()
        try:
            time1 = time.time()
            for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                self._seed_eval_episode(episode_idx)

                obs = env.reset()
                policy.reset()

                done = False
                num_goal_achieved = np.zeros(self.env_num)
                actual_step_count = 0
                episode_reward  = np.zeros(self.env_num)
                first_done = np.zeros(self.env_num)
                first_done_goal_achieved = np.zeros(self.env_num)
                first_done_return = np.zeros(self.env_num)
                while not done:
                    np_obs_dict = obs
                    obs_dict = dict_apply(np_obs_dict,
                                          lambda x: torch.from_numpy(x).to(
                                              device=device))
                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                        obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                        action_dict = policy.predict_action(obs_dict_input, deterministic=True)

                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action']
                    obs, reward, done, info = env.step(action)
                    episode_reward = episode_reward + reward
                    actual_step_count += 1
                    for i, entry in enumerate(info):
                        num_goal_achieved[i] += entry['goal_achieved']
                    for i, entry in enumerate(done):
                        if not first_done[i]:
                            first_done[i] = entry
                            first_done_goal_achieved[i] = num_goal_achieved[i]
                            first_done_return[i] = episode_reward[i]

                    done = np.all(done)
                all_returns.append(episode_reward)
                all_first_done_return.append(first_done_return)
                for each_num_goal_achieved in num_goal_achieved:
                    if each_num_goal_achieved > self.success_threshold:
                        hard_success += 1
                for each_first_done_goal_achieved in first_done_goal_achieved:
                    if each_first_done_goal_achieved > self.success_threshold:
                        first_done_hard_success += 1
                all_goal_achieved.append(num_goal_achieved)
            time2 = time.time()
            print('eval time (40 traj.):', time2 - time1)
        finally:
            self._restore_rng_states(rng_state)

        # log
        log_data = dict()
        all_success_rates_pre = hard_success / (self.eval_episodes * self.env_num)
        all_success_rates = first_done_hard_success / (self.eval_episodes * self.env_num)


        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates

        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_first_done_return)

        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"test_mean_score pre: {all_success_rates_pre}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_first_done_return)}", 'green')

        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env_method('get_video')[-1]
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

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

        all_goal_achieved = []
        all_success_rates = []
        all_first_done_return = []
        all_returns = []
        hard_success = 0
        first_done_hard_success = 0 

        rng_state = self._save_rng_states()
        try:
            time1 = time.time()
            for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                self._seed_eval_episode(episode_idx)

                obs = env.reset()
                policy.reset()

                done = False
                num_goal_achieved = np.zeros(self.env_num)
                actual_step_count = 0
                episode_reward  = np.zeros(self.env_num)
                first_done = np.zeros(self.env_num)
                first_done_goal_achieved = np.zeros(self.env_num)
                first_done_return = np.zeros(self.env_num)
                while not done:
                    np_obs_dict = obs
                    obs_dict = dict_apply(np_obs_dict,
                                          lambda x: torch.from_numpy(x).to(
                                              device=device))
                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                        obs_dict_input['image'] = (obs_dict['image']).to(torch.float)
                        action_dict = policy.sample_action(
                            obs_dict_input,
                            dynamics=dynamics,
                            first_action=first_action,
                            get_np=get_np,
                            use_gae=use_gae,
                            iql=iql,
                            Q=Q,
                            repeat_num=repeat_num,
                        )

                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action']
                    obs, reward, done, info = env.step(action)
                    episode_reward = episode_reward + reward
                    actual_step_count += 1
                    for i, entry in enumerate(info):
                        num_goal_achieved[i] += entry['goal_achieved']
                    for i, entry in enumerate(done):
                        if not first_done[i]:
                            first_done[i] = entry
                            first_done_goal_achieved[i] = num_goal_achieved[i]
                            first_done_return[i] = episode_reward[i]

                    done = np.all(done)
                all_returns.append(episode_reward)
                all_first_done_return.append(first_done_return)
                for each_num_goal_achieved in num_goal_achieved:
                    if each_num_goal_achieved > self.success_threshold:
                        hard_success += 1
                for each_first_done_goal_achieved in first_done_goal_achieved:
                    if each_first_done_goal_achieved > self.success_threshold:
                        first_done_hard_success += 1
                all_goal_achieved.append(num_goal_achieved)
            time2 = time.time()
            print('eval time (40 traj.):', time2 - time1)
        finally:
            self._restore_rng_states(rng_state)

        # log
        log_data = dict()
        all_success_rates_pre = hard_success / (self.eval_episodes * self.env_num)
        all_success_rates = first_done_hard_success / (self.eval_episodes * self.env_num)


        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates

        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_first_done_return)

        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"test_mean_score pre: {all_success_rates_pre}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_first_done_return)}", 'green')

        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env_method('get_video')[-1]
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
