import inspect
import wandb
import numpy as np
import torch
import tqdm
from rl_100.env import make_dmc_env
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


def _filter_supported_kwargs(method, kwargs):
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


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
                 with_pointcloud=True,
                 gamma=0.99,
                 ):
        super().__init__(output_dir)
        self._gamma = gamma
        self.task_name = task_name
        steps_per_render = max(10 // fps, 1)

        self._seed = seed
        self._env_counter = 0
        self.env = make_dmc_env(task_name, n_obs_steps, n_action_steps,
                                  2, seed)
        self.eval_episodes = eval_episodes

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def make_env(self, record_video=True):
        self._env_counter += 1
        return make_dmc_env(self.task_name, self.n_obs_steps, self.n_action_steps,
                            2, self._seed + self._env_counter)

    def _stack_obs(self, obs_list, device):
        """Stack list of single-env obs dicts into batched obs dict."""
        batched = {}
        for key in ['point_cloud', 'agent_pos', 'image']:
            if key in obs_list[0]:
                batched[key] = torch.from_numpy(
                    np.stack([obs[key] for obs in obs_list], axis=0)
                ).to(device=device, dtype=torch.float)
        return batched

    def run(self, policy: BasePolicy, use_cm=False, distill2mean=False, eval_env_num=1):
        if eval_env_num <= 1:
            return self._run_serial(policy, use_cm=use_cm, distill2mean=distill2mean)

        device = policy.device
        all_returns = []

        episode_idx = 0
        while episode_idx < self.eval_episodes:
            wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
            envs = [self.make_env(record_video=(j == 0)) for j in range(wave_size)]
            obs_list = [envs[j].reset() for j in range(wave_size)]
            policy.reset()

            active = [True] * wave_size
            ep_rewards = [0.0] * wave_size

            while any(active):
                active_indices = [j for j in range(wave_size) if active[j]]
                active_obs = [obs_list[j] for j in active_indices]
                obs_batched = self._stack_obs(active_obs, device)

                with torch.no_grad():
                    predict_kwargs = _filter_supported_kwargs(
                        policy.predict_action,
                        {'deterministic': True, 'use_cm': use_cm, 'distill2mean': distill2mean})
                    action_dict = policy.predict_action(obs_batched, **predict_kwargs)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())
                actions = np_action_dict['action']

                for k, j in enumerate(active_indices):
                    obs, reward, done, info = envs[j].step(actions[k])
                    ep_rewards[j] += reward
                    if np.all(done):
                        active[j] = False
                    else:
                        obs_list[j] = obs

            for j in range(wave_size):
                all_returns.append(ep_rewards[j])

            for e in envs:
                if hasattr(e, 'close'):
                    e.close()
            episode_idx += wave_size

        log_data = dict()
        log_data['test_mean_score'] = np.mean(all_returns)
        log_data['mean_returns'] = np.mean(all_returns)
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(np.mean(all_returns))
        self.logger_util_test10.record(np.mean(all_returns))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        return log_data

    def _run_serial(self, policy: BasePolicy, use_cm=False, distill2mean=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_returns = []

        time1 = time.time()
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            actual_step_count = 0
            episode_reward  = 0
            time_start = time.time()
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
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                    predict_kwargs = _filter_supported_kwargs(
                        policy.predict_action,
                        {
                            'deterministic': True,
                            'use_cm': use_cm,
                            'distill2mean': distill2mean,
                        }
                    )
                    action_dict = policy.predict_action(obs_dict_input, **predict_kwargs)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                episode_reward = episode_reward + reward
                actual_step_count += 1
            time_end = time.time()
            print('action frequency: ', actual_step_count/(time_end - time_start))
            all_returns.append(episode_reward)
        time2 = time.time()
        print('eval time (40 traj.):', time2 - time1)   

        # log
        log_data = dict()

        log_data['test_mean_score'] = np.mean(all_returns)
        log_data['mean_returns'] = np.mean(all_returns)

        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_returns)}", 'green')

        self.logger_util_test.record(np.mean(all_returns))
        self.logger_util_test10.record(np.mean(all_returns))
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
    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=False, distill2mean=False, eval_env_num=1):
        if eval_env_num <= 1:
            return self._idql_run_serial(policy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=use_cm, distill2mean=distill2mean)

        device = policy.device
        all_returns = []

        episode_idx = 0
        while episode_idx < self.eval_episodes:
            wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
            envs = [self.make_env(record_video=(j == 0)) for j in range(wave_size)]
            obs_list = [envs[j].reset() for j in range(wave_size)]
            policy.reset()

            active = [True] * wave_size
            ep_rewards = [0.0] * wave_size

            while any(active):
                active_indices = [j for j in range(wave_size) if active[j]]
                active_obs = [obs_list[j] for j in active_indices]
                obs_batched = self._stack_obs(active_obs, device)

                with torch.no_grad():
                    sample_kwargs = _filter_supported_kwargs(
                        policy.sample_action,
                        {'dynamics': dynamics, 'first_action': first_action,
                         'get_np': get_np, 'use_gae': use_gae, 'iql': iql,
                         'Q': Q, 'repeat_num': repeat_num,
                         'use_cm': use_cm, 'distill2mean': distill2mean})
                    action_dict = policy.sample_action(obs_batched, **sample_kwargs)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())
                actions = np_action_dict['action']

                for k, j in enumerate(active_indices):
                    obs, reward, done, info = envs[j].step(actions[k])
                    ep_rewards[j] += reward
                    if np.all(done):
                        active[j] = False
                    else:
                        obs_list[j] = obs

            for j in range(wave_size):
                all_returns.append(ep_rewards[j])

            for e in envs:
                if hasattr(e, 'close'):
                    e.close()
            episode_idx += wave_size

        log_data = dict()
        log_data['test_mean_score'] = np.mean(all_returns)
        log_data['mean_returns'] = np.mean(all_returns)
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(np.mean(all_returns))
        self.logger_util_test10.record(np.mean(all_returns))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        return log_data

    def _idql_run_serial(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=False, distill2mean=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_returns = []

        time1 = time.time()
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            actual_step_count = 0
            episode_reward  = 0
            time_start = time.time()
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
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = (obs_dict['image']).unsqueeze(0).to(torch.float)
                    sample_kwargs = _filter_supported_kwargs(
                        policy.sample_action,
                        {
                            'dynamics': dynamics,
                            'first_action': first_action,
                            'get_np': get_np,
                            'use_gae': use_gae,
                            'iql': iql,
                            'Q': Q,
                            'repeat_num': repeat_num,
                            'use_cm': use_cm,
                            'distill2mean': distill2mean,
                        }
                    )
                    action_dict = policy.sample_action(obs_dict_input, **sample_kwargs)
            
                    
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                episode_reward = episode_reward + reward
                actual_step_count += 1
            time_end = time.time()
            print('action frequency: ', actual_step_count/(time_end - time_start))
            all_returns.append(episode_reward)
        time2 = time.time()
        print('eval time (40 traj.):', time2 - time1)   

        # log
        log_data = dict()

        log_data['test_mean_score'] = np.mean(all_returns)
        log_data['mean_returns'] = np.mean(all_returns)

        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        cprint(f"mean_returns pre: {np.mean(all_returns)}", 'green')

        self.logger_util_test.record(np.mean(all_returns))
        self.logger_util_test10.record(np.mean(all_returns))
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
