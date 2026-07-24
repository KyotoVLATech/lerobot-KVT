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
                 record_video=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self._device = device
        self._use_point_crop = use_point_crop
        self._num_points = num_points
        self.record_video = record_video

        def env_fn(task_name, record_video=False):
            inner_env = MetaWorldEnv(
                task_name=task_name, device=device,
                use_point_crop=use_point_crop, num_points=num_points, rgb_size=84)
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

    def make_env(self, record_video=True):
        inner_env = MetaWorldEnv(task_name=self.task_name, device=self._device,
                                 use_point_crop=self._use_point_crop,
                                 num_points=self._num_points, rgb_size=84)
        if record_video:
            inner_env = SimpleVideoRecordingWrapper(inner_env)
        return MultiStepWrapper(
            inner_env,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
            reward_agg_method='sum',
        )

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
        all_traj_rewards = []
        all_success_rates = []

        episode_idx = 0
        videos_wandb = None
        while episode_idx < self.eval_episodes:
            wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
            envs = [self.make_env(record_video=(self.record_video and j == 0)) for j in range(wave_size)]
            obs_list = [envs[j].reset() for j in range(wave_size)]
            policy.reset()

            active = [True] * wave_size
            ep_rewards = [0.0] * wave_size
            ep_success = [False] * wave_size

            while any(active):
                active_indices = [j for j in range(wave_size) if active[j]]
                active_obs = [obs_list[j] for j in active_indices]
                obs_batched = self._stack_obs(active_obs, device)

                with torch.no_grad():
                    action_dict = policy.predict_action(
                        obs_batched, deterministic=True,
                        use_cm=use_cm, distill2mean=distill2mean)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())
                actions = np_action_dict['action']

                for k, j in enumerate(active_indices):
                    obs, reward, done, info = envs[j].step(actions[k])
                    ep_rewards[j] += reward
                    done = np.all(done)
                    ep_success[j] = ep_success[j] or max(info['success'])
                    if done:
                        active[j] = False
                    else:
                        obs_list[j] = obs

            for j in range(wave_size):
                all_traj_rewards.append(ep_rewards[j])
                all_success_rates.append(ep_success[j])

            if self.record_video and episode_idx == 0 and hasattr(envs[0], 'env') and hasattr(envs[0].env, 'get_video'):
                videos = envs[0].env.get_video()
                if len(videos.shape) == 5:
                    videos = videos[:, 0]
                videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")

            for e in envs:
                if hasattr(e, 'close'):
                    e.close()
            episode_idx += wave_size

        log_data = dict()
        log_data['mean_returns'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        if videos_wandb is not None:
            log_data['sim_video_eval'] = videos_wandb
        return log_data

    def _run_serial(self, policy: BasePolicy, use_cm=False, distill2mean=False):
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
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                    
            
                # device_transfer
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
        

        if self.record_video:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data['sim_video_eval'] = videos_wandb

            _ = env.reset()
            videos = None

        return log_data
        
    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=False, distill2mean=False, eval_env_num=1):
        if eval_env_num <= 1:
            return self._idql_run_serial(policy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=use_cm, distill2mean=distill2mean)

        device = policy.device
        all_traj_rewards = []
        all_success_rates = []

        episode_idx = 0
        videos_wandb = None
        while episode_idx < self.eval_episodes:
            wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
            envs = [self.make_env(record_video=(self.record_video and j == 0)) for j in range(wave_size)]
            obs_list = [envs[j].reset() for j in range(wave_size)]
            policy.reset()

            active = [True] * wave_size
            ep_rewards = [0.0] * wave_size
            ep_success = [False] * wave_size

            while any(active):
                active_indices = [j for j in range(wave_size) if active[j]]
                active_obs = [obs_list[j] for j in active_indices]
                obs_batched = self._stack_obs(active_obs, device)

                with torch.no_grad():
                    action_dict = policy.sample_action(
                        obs_batched, dynamics=dynamics,
                        first_action=first_action, get_np=get_np,
                        use_gae=use_gae, iql=iql, Q=Q,
                        repeat_num=repeat_num)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())
                actions = np_action_dict['action']

                for k, j in enumerate(active_indices):
                    obs, reward, done, info = envs[j].step(actions[k])
                    ep_rewards[j] += reward
                    done = np.all(done)
                    ep_success[j] = ep_success[j] or max(info['success'])
                    if done:
                        active[j] = False
                    else:
                        obs_list[j] = obs

            for j in range(wave_size):
                all_traj_rewards.append(ep_rewards[j])
                all_success_rates.append(ep_success[j])

            if self.record_video and episode_idx == 0 and hasattr(envs[0], 'env') and hasattr(envs[0].env, 'get_video'):
                videos = envs[0].env.get_video()
                if len(videos.shape) == 5:
                    videos = videos[:, 0]
                videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")

            for e in envs:
                if hasattr(e, 'close'):
                    e.close()
            episode_idx += wave_size

        log_data = dict()
        log_data['mean_returns'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        if videos_wandb is not None:
            log_data['sim_video_eval'] = videos_wandb
        return log_data

    def _idql_run_serial(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=False, distill2mean=False):
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
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    obs_dict_input['image'] = (obs_dict['image']).unsqueeze(0).to(torch.float)
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
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if False:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data

