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
import copy
import torch
import numpy as np
import h5py
import tqdm
import os
from copy import deepcopy
from rl_100.env import FrankaPourEnv
class Pour(BaseRunner):
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
                 fake_env=False,
                 num_points=512,
                 state_shape=6,
                 ):
        super().__init__(output_dir)
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.fake_env = fake_env
        self.eval_episodes = eval_episodes
        self.task_name = task_name
        self.state_shape = state_shape
        if self.fake_env:
            self.env = None
        else:
            self.env = MultiStepWrapper(
                FrankaPourEnv(num_point_cloud=num_points),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
                )

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
    def run1(self, policy: BasePolicy, use_cm = False, distill2mean=False, traj_path=None):  
        if self. fake_env:
            all_goal_achieved = np.random.randint(90, 100)
            all_success_rates = np.random.randint(90, 100)
            all_returns = 100
            log_data = {}
            log_data['mean_n_goal_achieved'] = np.mean(all_success_rates)
            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
            return log_data
        else:
            device = policy.device
            dtype = policy.dtype
            env = self.env

            all_goal_achieved = []
            all_success_rates = []
            all_returns = []
            hard_success = 0

            for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
                # start rollout
                obs = env.reset()
                policy.reset()

                done = False
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                # import pdb; pdb.set_trace()
                time_start = time.time()
                is_success = False
                pre_reward = -1
                while not done:
                    # create obs dict
                    time_action = time.time()
                    np_obs_dict = obs
                    # device transfer
                    # import pdb; pdb.set_trace()
                    obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).to(
                                            device=device))
                    # run policy
                    with torch.no_grad():
                        obs_dict_input = {}  # flush unused keys
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        if self.state_shape == 2:
                            obs_dict_input['agent_pos'] = obs_dict['agent_xy'].unsqueeze(0).to(torch.float)
                        else:
                            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                        # import pdb; pdb.set_trace()
                        action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                        
                
                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    # step env
                    obs, reward, done, info = env.step(action)
                    if reward is None:
                        reward = pre_reward
                    episode_reward += reward
                    pre_reward = reward
                    if info['is_success'].any() and is_success == False:
                        is_success = True
                        hard_success += 1
                    actual_step_count += 1
                    print(1 / (time.time() - time_action))
                time_end = time.time()
                print('action frequency: ', actual_step_count/(time_end - time_start))
                all_returns.append(episode_reward)
                time.sleep(4)
            # log
            log_data = dict()

            all_success_rates = hard_success / self.eval_episodes

            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

            _ = env.reset()
            del env

            return log_data
    def run(self, policy: BasePolicy, data_collect = False, use_cm = False, distill2mean=False, traj_path = None): 
        if data_collect:
            deterministic = False
        else:
            deterministic = True

        if self. fake_env:
            all_goal_achieved = np.random.randint(90, 100)
            all_success_rates = np.random.randint(90, 100)
            all_returns = 100
            log_data = {}
            log_data['mean_n_goal_achieved'] = np.mean(all_success_rates)
            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
            return log_data
        else:
            device = policy.device
            dtype = policy.dtype
            env = self.env

            all_goal_achieved = []
            all_success_rates = []
            all_returns = []
            hard_success = 0
            from datetime import datetime

            file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # 使用 'a' 模式打开，方便追加写入数据
            for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                        desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                        leave=False, mininterval=self.tqdm_interval_sec):
                
                # 每个 episode 初始化数据存储列表
                point_cloud_arrays = []
                state_arrays = []
                action_arrays = []
                depth_arrays = []
                depth_scale_arrays = []
                
                next_point_cloud_arrays = []
                next_state_arrays = []   
                next_action_arrays = []
                next_depth_arrays = []
                next_depth_scale_arrays = []
                
                reward_arrays = []
                done_arrays = []
                timeout_arrays = []
                is_success_arrays = []
                tag_pos_arrays = []
                
                # 开始 rollout
                obs = env.reset()
                policy.reset()
                
                done = False 
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                total_count = 0
                time_start = time.time()
                is_success = False
                pre_reward = -1
                
                while not done:
                    # 保存当前状态信息
                    time_action = time.time()
                    np_obs_dict = obs
                    if self.state_shape == 2:
                        state_arrays.append(obs['agent_xy'][-1])
                    else:
                        state_arrays.append(obs['agent_pos'][-1])
                    point_cloud_arrays.append(obs['point_cloud'][-1])
                    depth_arrays.append(obs['depth'][-1])
                    depth_scale_arrays.append(obs['depth_scale'][-1])
                    
                    # 将 numpy 转换为 torch tensor 并移动到指定设备
                    obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).to(device=device))
                    
                    # 构造策略输入字典（增加 batch 维度）
                    with torch.no_grad():
                        obs_dict_input = {
                            'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        }
                        if self.state_shape == 2:
                            obs_dict_input['agent_pos'] = obs_dict['agent_xy'].unsqueeze(0).to(torch.float)
                        else:
                            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                        action_dict = policy.predict_action(obs_dict_input, deterministic=deterministic, use_cm=use_cm, distill2mean=distill2mean)
                    
                    # 将动作从 tensor 转换为 numpy 并保存
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    action_arrays.append(action)
                    
                    # 执行环境一步
                    obs, reward, done, info = env.step(action)
                    if self.state_shape == 2:
                        next_state_arrays.append(obs['agent_xy'][-1])
                    else:
                        next_state_arrays.append(obs['agent_pos'][-1])
                    next_point_cloud_arrays.append(obs['point_cloud'][-1])
                    next_depth_arrays.append(obs['depth'][-1])
                    next_depth_scale_arrays.append(obs['depth_scale'][-1])
                    
                    done_arrays.append(done)
                    timeout_arrays.append(info['timeout'][-1])
                    is_success_arrays.append(info['is_success'][-1])
                    tag_pos_arrays.append(info['tag_pos'][-1])
                    if reward is None:
                        reward = pre_reward
                    episode_reward += reward
                    reward_arrays.append(reward)
                    pre_reward = reward
                    
                    if info['is_success'].any() and not is_success:
                        is_success = True
                        hard_success += 1  # 确保 hard_success 已定义
                        
                    actual_step_count += 1
                    print(1 / (time.time() - time_action))
                # time.sleep(3)
                # 结束回合时，再对最后一步做一次动作预测（保证 next_action_arrays 对齐）
                obs_dict = dict_apply(obs,
                                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    }
                    if self.state_shape == 2:
                        obs_dict_input['agent_pos'] = obs_dict['agent_xy'].unsqueeze(0).to(torch.float)
                    else:
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                
                # 处理 next_action_arrays，使用 deepcopy 避免引用问题，然后调整索引
                next_action_arrays = copy.deepcopy(action_arrays)
                next_action_arrays.append(action)
                next_action_arrays = next_action_arrays[1:]
                
                time_end = time.time()
                print('action frequency: ', actual_step_count / (time_end - time_start))
                all_returns.append(episode_reward)
                time.sleep(4)
                
                # 将本 episode 的数据保存到 HDF5 文件中
                # import pdb; pdb.set_trace()
                # if data_collect:
                hdf5_filename = os.path.join(traj_path, 'eval_episode_{}_{}.h5'.format(str(file_time), episode_idx))
                # tag_pos_path =os.path.join(traj_path, 'demo_{episode_idx}.npy')
                # np.save(tag_pos_path, info['tag_pos'][-1],  allow_pickle=True)
                with h5py.File(hdf5_filename, 'w') as hdf5_file:
                    hdf5_file.create_dataset('point_cloud', data=np.array(point_cloud_arrays))
                    hdf5_file.create_dataset('depth', data=np.array(depth_arrays))
                    hdf5_file.create_dataset('depth_scale', data=np.array(depth_scale_arrays))
                    
                    hdf5_file.create_dataset('state', data=np.array(state_arrays))
                    
                    hdf5_file.create_dataset('action', data=np.array(action_arrays))
                    hdf5_file.create_dataset('next_point_cloud', data=np.array(next_point_cloud_arrays))
                    hdf5_file.create_dataset('next_depth', data=np.array(next_depth_arrays))
                    hdf5_file.create_dataset('next_depth_scale', data=np.array(next_depth_scale_arrays))
                    hdf5_file.create_dataset('next_state', data=np.array(next_state_arrays))
                    hdf5_file.create_dataset('next_action', data=np.array(next_action_arrays))
                    hdf5_file.create_dataset('reward', data=np.array(reward_arrays))
                    hdf5_file.create_dataset('done', data=np.array(done_arrays))
                    hdf5_file.create_dataset('timeout', data=np.array(timeout_arrays))
                    hdf5_file.create_dataset('is_success', data=np.array(is_success_arrays))
                    # import pdb; pdb.set_trace()
                    # hdf5_file.create_dataset('tag_pos', data=tag_pos_arrays)
                    print('data save in {}'.format(hdf5_filename))
            # log
            log_data = dict()

            all_success_rates = hard_success / self.eval_episodes

            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

            _ = env.reset()
            del env

            return log_data
    
    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, traj_path = None, data_collect = None, use_cm=False, distill2mean=False):
        if data_collect:
            deterministic = False
        else:
            deterministic = True

        if self. fake_env:
            all_goal_achieved = np.random.randint(90, 100)
            all_success_rates = np.random.randint(90, 100)
            all_returns = 100
            log_data = {}
            log_data['mean_n_goal_achieved'] = np.mean(all_success_rates)
            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
            return log_data
        else:
            device = policy.device
            dtype = policy.dtype
            env = self.env

            all_goal_achieved = []
            all_success_rates = []
            all_returns = []
            hard_success = 0
            from datetime import datetime

            file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # 使用 'a' 模式打开，方便追加写入数据
            for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                        desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                        leave=False, mininterval=self.tqdm_interval_sec):
                
                # 每个 episode 初始化数据存储列表
                point_cloud_arrays = []
                state_arrays = []
                action_arrays = []
                depth_arrays = []
                depth_scale_arrays = []
                
                next_point_cloud_arrays = []
                next_state_arrays = []   
                next_action_arrays = []
                next_depth_arrays = []
                next_depth_scale_arrays = []
                
                reward_arrays = []
                done_arrays = []
                timeout_arrays = []
                is_success_arrays = []
                tag_pos_arrays = []
                
                # 开始 rollout
                obs = env.reset()
                policy.reset()
                
                done = False 
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                total_count = 0
                time_start = time.time()
                is_success = False
                pre_reward = -1
                
                while not done:
                    # 保存当前状态信息
                    time_action = time.time()
                    np_obs_dict = obs
                    if self.state_shape == 2:
                        state_arrays.append(obs['agent_xy'][-1])
                    else:
                        state_arrays.append(obs['agent_pos'][-1])
                    point_cloud_arrays.append(obs['point_cloud'][-1])
                    depth_arrays.append(obs['depth'][-1])
                    depth_scale_arrays.append(obs['depth_scale'][-1])
                    
                    # 将 numpy 转换为 torch tensor 并移动到指定设备
                    obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).to(device=device))
                    
                    # 构造策略输入字典（增加 batch 维度）
                    with torch.no_grad():
                        obs_dict_input = {
                            'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        }
                        if self.state_shape == 2:
                            obs_dict_input['agent_pos'] = obs_dict['agent_xy'].unsqueeze(0).to(torch.float)
                        else:
                            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                        
                        action_dict = policy.sample_action(obs_dict_input, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae = use_gae, iql=iql, Q=Q, repeat_num=repeat_num \
                                                           , use_cm=use_cm, distill2mean=distill2mean)                    
                    # 将动作从 tensor 转换为 numpy 并保存
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    action_arrays.append(action)
                    
                    # 执行环境一步
                    obs, reward, done, info = env.step(action)
                    if self.state_shape == 2:
                        next_state_arrays.append(obs['agent_xy'][-1])
                    else:
                        next_state_arrays.append(obs['agent_pos'][-1])
                    next_point_cloud_arrays.append(obs['point_cloud'][-1])
                    next_depth_arrays.append(obs['depth'][-1])
                    next_depth_scale_arrays.append(obs['depth_scale'][-1])
                    
                    done_arrays.append(done)
                    timeout_arrays.append(info['timeout'][-1])
                    is_success_arrays.append(info['is_success'][-1])
                    tag_pos_arrays.append(info['tag_pos'][-1])
                    if reward is None:
                        reward = pre_reward
                    episode_reward += reward
                    reward_arrays.append(reward)
                    pre_reward = reward
                    
                    if info['is_success'].any() and not is_success:
                        is_success = True
                        hard_success += 1  # 确保 hard_success 已定义
                        
                    actual_step_count += 1
                    print(1 / (time.time() - time_action))

                # 结束回合时，再对最后一步做一次动作预测（保证 next_action_arrays 对齐）
                obs_dict = dict_apply(obs,
                                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    }
                    if self.state_shape == 2:
                        obs_dict_input['agent_pos'] = obs_dict['agent_xy'].unsqueeze(0).to(torch.float)
                    else:
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                
                # 处理 next_action_arrays，使用 deepcopy 避免引用问题，然后调整索引
                next_action_arrays = copy.deepcopy(action_arrays)
                next_action_arrays.append(action)
                next_action_arrays = next_action_arrays[1:]
                
                time_end = time.time()
                print('action frequency: ', actual_step_count / (time_end - time_start))
                all_returns.append(episode_reward)
                time.sleep(4)
                
                # 将本 episode 的数据保存到 HDF5 文件中
                # import pdb; pdb.set_trace()
                # if data_collect:
                hdf5_filename = os.path.join(traj_path, 'eval_episode_{}_{}.h5'.format(str(file_time), episode_idx))
                # tag_pos_path =os.path.join(traj_path, 'demo_{episode_idx}.npy')
                # np.save(tag_pos_path, info['tag_pos'][-1],  allow_pickle=True)
                with h5py.File(hdf5_filename, 'w') as hdf5_file:
                    hdf5_file.create_dataset('point_cloud', data=np.array(point_cloud_arrays))
                    hdf5_file.create_dataset('depth', data=np.array(depth_arrays))
                    hdf5_file.create_dataset('depth_scale', data=np.array(depth_scale_arrays))
                    
                    hdf5_file.create_dataset('state', data=np.array(state_arrays))
                    
                    hdf5_file.create_dataset('action', data=np.array(action_arrays))
                    hdf5_file.create_dataset('next_point_cloud', data=np.array(next_point_cloud_arrays))
                    hdf5_file.create_dataset('next_depth', data=np.array(next_depth_arrays))
                    hdf5_file.create_dataset('next_depth_scale', data=np.array(next_depth_scale_arrays))
                    hdf5_file.create_dataset('next_state', data=np.array(next_state_arrays))
                    hdf5_file.create_dataset('next_action', data=np.array(next_action_arrays))
                    hdf5_file.create_dataset('reward', data=np.array(reward_arrays))
                    hdf5_file.create_dataset('done', data=np.array(done_arrays))
                    hdf5_file.create_dataset('timeout', data=np.array(timeout_arrays))
                    hdf5_file.create_dataset('is_success', data=np.array(is_success_arrays))
                    # import pdb; pdb.set_trace()
                    # hdf5_file.create_dataset('tag_pos', data=tag_pos_arrays)
                    print('data save in {}'.format(hdf5_filename))
            # log
            log_data = dict()

            all_success_rates = hard_success / self.eval_episodes

            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

            _ = env.reset()
            del env

            return log_data