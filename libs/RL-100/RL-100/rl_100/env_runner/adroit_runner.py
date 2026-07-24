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


def _make_adroit_env(task_name, use_point_crop, n_obs_steps, n_action_steps,
                     max_steps, reward_agg_method, gamma, record_video,
                     with_pointcloud=True):
    inner_env = MujocoPointcloudWrapperAdroit(
        env=AdroitEnv(env_name=task_name, use_point_cloud=with_pointcloud),
        env_name='adroit_' + task_name,
        use_point_crop=use_point_crop,
        with_pointcloud=with_pointcloud,
    )
    if record_video:
        inner_env = SimpleVideoRecordingWrapper(inner_env)
    return MultiStepWrapper(
        inner_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method=reward_agg_method,
        gamma=gamma,
    )


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
                 env_num=1,
                 with_pointcloud=True,
                 gamma=0.99,
                 eval_seed=0,
                 record_video=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        if 'pen' in task_name:
            self.success_threshold = 20
        else:
            self.success_threshold = 25

        steps_per_render = max(10 // fps, 1)

        self._use_point_crop = use_point_crop
        self._with_pointcloud = with_pointcloud
        self._gamma = gamma
        self.eval_episodes = eval_episodes

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.eval_seed = eval_seed
        self.record_video = record_video
        self.env = self.make_env(record_video=self.record_video)

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def _build_env(self, record_video=True, reward_agg_method='sum', gamma=None):
        return _make_adroit_env(
            task_name=self.task_name,
            use_point_crop=self._use_point_crop,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_steps=self.max_steps,
            reward_agg_method=reward_agg_method,
            gamma=self._gamma if gamma is None else gamma,
            record_video=record_video,
            with_pointcloud=self._with_pointcloud,
        )

    def make_env(self, record_video=True, reward_agg_method='sum', gamma=None):
        return self._build_env(
            record_video=record_video,
            reward_agg_method=reward_agg_method,
            gamma=gamma,
        )

    def make_env_fn(self, record_video=True, reward_agg_method='sum', gamma=None):
        task_name = self.task_name
        use_point_crop = self._use_point_crop
        n_obs_steps = self.n_obs_steps
        n_action_steps = self.n_action_steps
        max_steps = self.max_steps
        with_pointcloud = self._with_pointcloud
        gamma = self._gamma if gamma is None else gamma

        def _init():
            return _make_adroit_env(
                task_name=task_name,
                use_point_crop=use_point_crop,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                record_video=record_video,
                reward_agg_method=reward_agg_method,
                gamma=gamma,
                with_pointcloud=with_pointcloud,
            )
        return _init

    def make_subproc_vec_env(self, env_num, record_video_first=False,
                             reward_agg_method='sum', gamma=None, start_method='spawn'):
        env_fns = [
            self.make_env_fn(
                record_video=(record_video_first and i == 0),
                reward_agg_method=reward_agg_method,
                gamma=gamma,
            )
            for i in range(env_num)
        ]
        return SubprocVecEnv(env_fns, start_method)

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

    def _seed_eval_episode(self, episode_idx, env=None):
        seed = int(self.eval_seed + episode_idx)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        target_env = env if env is not None else self.env
        target_env.seed(seed)

    def _stack_obs(self, obs_list, device):
        """Stack list of single-env obs dicts into batched obs dict."""
        batched = {}
        for key in ['point_cloud', 'agent_pos', 'image']:
            if key in obs_list[0]:
                batched[key] = torch.from_numpy(
                    np.stack([obs[key] for obs in obs_list], axis=0)
                ).to(device=device, dtype=torch.float)
        return batched

    def _stack_vec_obs(self, obs_dict, device):
        batched = {}
        for key in ['point_cloud', 'agent_pos', 'image']:
            if key in obs_dict:
                batched[key] = torch.from_numpy(obs_dict[key]).to(
                    device=device, dtype=torch.float)
        return batched

    def _seed_eval_vec_env(self, base_seed, vec_env):
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        vec_env.seed(base_seed)

    def run(self, policy: BasePolicy, use_cm=False, distill2mean=False, eval_env_num=1):
        device = policy.device
        dtype = policy.dtype

        if eval_env_num <= 1:
            return self._run_serial(policy, use_cm=use_cm, distill2mean=distill2mean)

        all_goal_achieved = []
        all_returns = []
        hard_success = 0

        rng_state = self._save_rng_states()
        try:
            episode_idx = 0
            videos_wandb = None
            while episode_idx < self.eval_episodes:
                wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
                torch.cuda.empty_cache()
                vec_env = self.make_subproc_vec_env(
                    wave_size,
                    record_video_first=(self.record_video and episode_idx == 0),
                    reward_agg_method='sum',
                    gamma=self._gamma,
                )
                self._seed_eval_vec_env(self.eval_seed + episode_idx, vec_env)

                obs = vec_env.reset()
                policy.reset()

                first_done = np.zeros(wave_size, dtype=bool)
                ep_rewards = np.zeros(wave_size, dtype=np.float64)
                ep_goals = np.zeros(wave_size, dtype=np.float64)
                first_done_returns = np.zeros(wave_size, dtype=np.float64)
                first_done_goals = np.zeros(wave_size, dtype=np.float64)

                while not np.all(first_done):
                    obs_batched = self._stack_vec_obs(obs, device)
                    with torch.no_grad():
                        action_dict = policy.predict_action(
                            obs_batched, deterministic=True,
                            use_cm=use_cm, distill2mean=distill2mean)
                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().cpu().numpy())
                    obs, reward, done, info = vec_env.step(np_action_dict['action'])
                    ep_rewards += reward
                    for j, entry in enumerate(info):
                        ep_goals[j] += np.sum(entry['goal_achieved'])
                        if (not first_done[j]) and done[j]:
                            first_done[j] = True
                            first_done_returns[j] = ep_rewards[j]
                            first_done_goals[j] = ep_goals[j]

                for j in range(wave_size):
                    all_returns.append(first_done_returns[j])
                    all_goal_achieved.append(first_done_goals[j])
                    if first_done_goals[j] > self.success_threshold:
                        hard_success += 1

                if self.record_video and episode_idx == 0:
                    videos = vec_env.env_method('get_video', indices=0)[0]
                    if len(videos.shape) == 5:
                        videos = videos[:, 0]
                    videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")

                vec_env.close()
                episode_idx += wave_size
        finally:
            self._restore_rng_states(rng_state)

        log_data = dict()
        all_success_rates = hard_success / self.eval_episodes
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates
        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_returns)
        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        if videos_wandb is not None:
            log_data['sim_video_eval'] = videos_wandb
        return log_data

    def _run_serial(self, policy: BasePolicy, use_cm=False, distill2mean=False):
        """Original serial eval path."""
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_returns = []
        hard_success = 0


        rng_state = self._save_rng_states()
        try:
            for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                # self._seed_eval_episode(episode_idx)

                # start rollout
                obs = env.reset()
                policy.reset()

                done = False
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                time_start = time.time()
                while not done:
                    np_obs_dict = obs
                    obs_dict = dict_apply(np_obs_dict,
                                          lambda x: torch.from_numpy(x).to(
                                              device=device))
                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                        action_dict = policy.predict_action(
                            obs_dict_input,
                            deterministic=True,
                            use_cm=use_cm,
                            distill2mean=distill2mean,
                        )

                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    num_goal_achieved += np.sum(info['goal_achieved'])
                    done = np.all(done)
                    actual_step_count += 1
                time_end = time.time()
                print('action frequency: ', actual_step_count/(time_end - time_start))
                all_returns.append(episode_reward)

                if num_goal_achieved > self.success_threshold:
                    hard_success += 1
                all_goal_achieved.append(num_goal_achieved)
        finally:
            self._restore_rng_states(rng_state)

        # log
        log_data = dict()
        all_success_rates = hard_success / self.eval_episodes

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates

        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_returns)
        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        if self.record_video:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data['sim_video_eval'] = videos_wandb

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
        all_goal_achieved = []
        all_returns = []
        hard_success = 0

        rng_state = self._save_rng_states()
        try:
            episode_idx = 0
            videos_wandb = None
            while episode_idx < self.eval_episodes:
                wave_size = min(eval_env_num, self.eval_episodes - episode_idx)
                torch.cuda.empty_cache()
                vec_env = self.make_subproc_vec_env(
                    wave_size,
                    record_video_first=(self.record_video and episode_idx == 0),
                    reward_agg_method='sum',
                    gamma=self._gamma,
                )
                self._seed_eval_vec_env(self.eval_seed + episode_idx, vec_env)

                obs = vec_env.reset()
                policy.reset()

                first_done = np.zeros(wave_size, dtype=bool)
                ep_rewards = np.zeros(wave_size, dtype=np.float64)
                ep_goals = np.zeros(wave_size, dtype=np.float64)
                first_done_returns = np.zeros(wave_size, dtype=np.float64)
                first_done_goals = np.zeros(wave_size, dtype=np.float64)

                while not np.all(first_done):
                    obs_batched = self._stack_vec_obs(obs, device)
                    with torch.no_grad():
                        action_dict = policy.sample_action(
                            obs_batched, dynamics=dynamics,
                            first_action=first_action, get_np=get_np,
                            use_gae=use_gae, iql=iql, Q=Q,
                            repeat_num=repeat_num)
                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().cpu().numpy())
                    obs, reward, done, info = vec_env.step(np_action_dict['action'])
                    ep_rewards += reward
                    for j, entry in enumerate(info):
                        ep_goals[j] += np.sum(entry['goal_achieved'])
                        if (not first_done[j]) and done[j]:
                            first_done[j] = True
                            first_done_returns[j] = ep_rewards[j]
                            first_done_goals[j] = ep_goals[j]

                for j in range(wave_size):
                    all_returns.append(first_done_returns[j])
                    all_goal_achieved.append(first_done_goals[j])
                    if first_done_goals[j] > self.success_threshold:
                        hard_success += 1

                if self.record_video and episode_idx == 0:
                    videos = vec_env.env_method('get_video', indices=0)[0]
                    if len(videos.shape) == 5:
                        videos = videos[:, 0]
                    videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")

                vec_env.close()
                episode_idx += wave_size
        finally:
            self._restore_rng_states(rng_state)

        log_data = dict()
        all_success_rates = hard_success / self.eval_episodes
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates
        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_returns)
        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        if videos_wandb is not None:
            log_data['sim_video_eval'] = videos_wandb
        return log_data

    def _idql_run_serial(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, use_cm=False, distill2mean=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        all_goal_achieved = []
        all_success_rates = []
        all_returns = []
        hard_success = 0
        rng_state = self._save_rng_states()
        try:
            for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                # self._seed_eval_episode(episode_idx)
                # start rollout
                obs = env.reset()
                policy.reset()

                done = False
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                time_start = time.time()
                while not done:
                    np_obs_dict = dict(obs)
                    obs_dict = dict_apply(np_obs_dict,
                                          lambda x: torch.from_numpy(x).to(
                                              device=device))
                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                        obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)

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
                    action = np_action_dict['action'].squeeze(0)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    num_goal_achieved += np.sum(info['goal_achieved'])
                    done = np.all(done)
                    actual_step_count += 1
                time_end = time.time()
                print('action frequency: ', actual_step_count/(time_end - time_start))
                if num_goal_achieved > self.success_threshold:
                    hard_success += 1
                all_returns.append(episode_reward)
                all_goal_achieved.append(num_goal_achieved)
        finally:
            self._restore_rng_states(rng_state)

        # log
        log_data = dict()
        
        all_success_rates = hard_success / self.eval_episodes
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = all_success_rates

        log_data['test_mean_score'] = all_success_rates
        log_data['mean_returns'] = np.mean(all_returns)

        cprint(f"test_mean_score: {all_success_rates}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
        self.logger_util_test.record(all_success_rates)
        self.logger_util_test10.record(all_success_rates)
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        if self.record_video and hasattr(env.env, 'get_video'):
            videos = env.env.get_video()
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
