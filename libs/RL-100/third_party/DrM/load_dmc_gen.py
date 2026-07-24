import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import dmc

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import re

DRM_ROOT = Path(__file__).resolve().parent

import mj_envs
from mjrl.utils.gym_env import GymEnv
from rrl_local.rrl_utils import make_basic_env, make_dir
from adroit import AdroitEnv
import matplotlib.pyplot as plt
import argparse
import os
import torch
import utils
from termcolor import cprint
from PIL import Image
import zarr
from copy import deepcopy
import numpy as np
from numpy import ndarray
from rl_100.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
from tqdm import tqdm

from rl_100.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
torch.backends.cudnn.benchmark = True

def render_camera(sim, camera_name="top"):
    img = sim.render(84, 84, camera_name=camera_name)
    return img

def render_high_res(sim, camera_name="top"):
    img = sim.render(1024, 1024, camera_name=camera_name)
    return img

def compute_return(reward: ndarray, not_done: ndarray, gamma: float == 0.99
    ) -> ndarray:
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_

def make_agent(obs_spec, action_spec, cfg):
    # import pdb; pdb.set_trace()
    # cfg.obs_shape = obs_spec.shape
    cfg.obs_shape = obs_spec['pixels'].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            wandb.init(project="DrM",
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # import pdb; pdb.set_trace()

                            
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            self._nstep,
            self._discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action.reshape(1, -1))
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        print(f'eval episode: {episode}, total_reward: {total_reward}')
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
    def eval_checkpoint(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs  = self.eval_env.reset()
            done = False
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not done:
                obs['image'] = np.concatenate(obs['image'], axis=0)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs['image'],
                                            self.global_step,
                                            eval_mode=True)
                obs, reward, done, _ = self.eval_env.step(action.reshape(1, -1))
                self.video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        print(f'========================================= eval episode: {episode}, total_reward: {total_reward / episode} =========================================')

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                self.save_snapshot(self.global_frame)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
                )
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, global_frame = None):
        if global_frame != None:
            snapshot = self.work_dir / 'snapshot_{}.pt'.format(global_frame)
        else:
            snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, path = None):
        if path != None:
            snapshot = path
        else:
            snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def gen_demo(self, num_episodes = 10, data_property='medium', pt_name=None):
        local_path = DRM_ROOT / 'exp_local'
        if pt_name != None:
            save_dir = local_path / 'data1' / f'dmc_{self.cfg.task_name}_{pt_name}_{data_property}.zarr'
        else:
            save_dir = local_path / 'data1' / f'dmc_{self.cfg.task_name}_{data_property}.zarr'
        if os.path.exists(save_dir):
            cprint('Data already exists at {}'.format(save_dir), 'red')
            cprint("If you want to overwrite, delete the existing directory first.", "red")
            cprint("Do you want to overwrite? (y/n)", "red")
            # user_input = input()
            user_input = 'y'
            if user_input == 'y':
                cprint('Overwriting {}'.format(save_dir), 'red')
                os.system('rm -rf {}'.format(save_dir))
            else:
                cprint('Exiting', 'red')
                return
        os.makedirs(save_dir, exist_ok=True)
        
        # cprint('Loaded expert ckpt from {}'.format(args.expert_ckpt_path), 'green')
        total_count = 0
        img_arrays = []
        point_cloud_arrays = []
        depth_arrays = []

        next_img_arrays = []
        next_point_cloud_arrays = []
        next_depth_arrays = []
        next_state_arrays = []   

        state_arrays = []
        action_arrays = []
        next_action_arrays = []
        reward_arrays = []
        done_arrays = []
        timeout_arrays = []
        episode_ends_arrays = []
        
        all_total_rewards = []
        # loop over episodes
        minimal_episode_length = 200
        episode_idx = 0
        while episode_idx < num_episodes:
            env = self.eval_env

            obs = env.reset()
            # input_obs_visual = time_step.observation # (3n,84,84), unit8
            # input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

            total_reward = 0.
            n_goal_achieved_total = 0.
            step_count = 0
            
            img_arrays_sub = []
            point_cloud_arrays_sub = []
            depth_arrays_sub = []

            next_img_arrays_sub = []
            next_point_cloud_arrays_sub = []
            next_depth_arrays_sub = []
            next_state_arrays_sub = []

            state_arrays_sub = []
            action_arrays_sub = []
            reward_arrays_sub = []
            done_arrays_sub = []
            timeout_arrays_sub = []
            total_count_sub = 0
            done = False

            while not done or step_count < minimal_episode_length:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    cond_img = np.concatenate(obs['image'], axis=0)
                    
                    action = self.agent.act(cond_img,
                                                self.global_step,
                                                eval_mode=True)
                    

                            
                    # save data
                    total_count_sub += 1
                    img_arrays_sub.append(obs['image'][-1])
                    state_arrays_sub.append(obs['agent_pos'][-1])
                    action_arrays_sub.append(action)
                    point_cloud_arrays_sub.append(obs['point_cloud'][-1])
                    depth_arrays_sub.append(obs['depth'][-1])

                

                next_obs, reward, done, _ = self.eval_env.step(action.reshape(1, -1))
                done_arrays_sub.append(done)
                reward_arrays_sub.append(reward)
                

                next_img_arrays_sub.append(next_obs['image'][-1])
                next_point_cloud_arrays_sub.append(next_obs['point_cloud'][-1])
                next_depth_arrays_sub.append(next_obs['depth'][-1])  
                next_state_arrays_sub.append(next_obs['agent_pos'][-1])

                obs = next_obs
                total_reward += reward
                step_count += 1
                if step_count < minimal_episode_length:
                    timeout_arrays_sub.append(False)
                else:
                    timeout_arrays_sub.append(True)
            with torch.no_grad(), utils.eval_mode(self.agent):
                cond_img = np.concatenate(obs['image'], axis=0)
                last_next_action = self.agent.act(cond_img,
                                            self.global_step,
                                            eval_mode=True) 
            next_action_arrays_sub = deepcopy(action_arrays_sub)   
                
            next_action_arrays_sub.append(last_next_action)
            next_action_arrays_sub = next_action_arrays_sub[1:]

            if False:
                cprint(f"Episode {episode_idx} has {n_goal_achieved_total} goals achieved and {total_reward} reward. Discarding.", 'red')
            else:
                total_count += total_count_sub
                episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
                img_arrays.extend(deepcopy(img_arrays_sub))
                point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
                depth_arrays.extend(deepcopy(depth_arrays_sub))
                next_img_arrays.extend(deepcopy(next_img_arrays_sub))
                next_point_cloud_arrays.extend(deepcopy(next_point_cloud_arrays_sub))
                next_depth_arrays.extend(deepcopy(next_depth_arrays_sub))
                next_state_arrays.extend(deepcopy(next_state_arrays_sub))

                state_arrays.extend(deepcopy(state_arrays_sub))
                action_arrays.extend(deepcopy(action_arrays_sub))
                next_action_arrays.extend(deepcopy(next_action_arrays_sub))
                reward_arrays.extend(deepcopy(reward_arrays_sub))
                done_arrays.extend(deepcopy(done_arrays_sub))
                timeout_arrays.extend(deepcopy(timeout_arrays_sub))
                print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
                all_total_rewards.append(total_reward)
                episode_idx += 1
        print('Mean reward: {}'.format(np.mean(all_total_rewards)))
        # import pdb; pdb.set_trace()
        # tracemalloc.stop()
        ###############################
        # save data
        ###############################
        # create zarr file
        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        # save img, state, action arrays into data, and episode ends arrays into meta
        img_arrays = np.stack(img_arrays, axis=0)
        next_img_arrays = np.stack(next_img_arrays, axis=0)
        if img_arrays.shape[1] == 3: # make channel last
            img_arrays = np.transpose(img_arrays, (0,2,3,1))
            next_img_arrays = np.transpose(next_img_arrays, (0,2,3,1))
        state_arrays = np.stack(state_arrays, axis=0)
        point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
        depth_arrays = np.stack(depth_arrays, axis=0)
        action_arrays = np.stack(action_arrays, axis=0)

        next_state_arrays = np.stack(next_state_arrays, axis=0)
        next_point_cloud_arrays = np.stack(next_point_cloud_arrays, axis=0)
        next_depth_arrays = np.stack(next_depth_arrays, axis=0)
        next_action_arrays = np.stack(next_action_arrays, axis=0)
        
        reward_arrays = np.array(reward_arrays).reshape(action_arrays.shape[0], -1)
        done_arrays = np.array(done_arrays).reshape(action_arrays.shape[0], -1)
        timeout_arrays = np.array(timeout_arrays).reshape(action_arrays.shape[0], -1)
        episode_ends_arrays = np.array(episode_ends_arrays)
        # print(done_arrays)

        # import pdb
        # pdb.set_trace()

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
        state_chunk_size = (100, state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
        action_chunk_size = (100, action_arrays.shape[1])
        reward_chunk_size = (100, reward_arrays.shape[1])

        done_chunk_size = (100, done_arrays.shape[1])
        timeout_chunk_size = (100, timeout_arrays.shape[1])
        # compute return for each episode
        not_done_arrays =  1. - (done_arrays | timeout_arrays)
        done_timeout_arrays = done_arrays | timeout_arrays  
        done_indices = np.where(done_timeout_arrays.flatten())[0]
        return_arrays = compute_return(reward_arrays, not_done_arrays, 0.99)
        return_chunk_size = (100, return_arrays.shape[1])

        
        # import pdb
        # pdb.set_trace()
        zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

        zarr_data.create_dataset('next_img', data=next_img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_state', data=next_state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_point_cloud', data=next_point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_depth', data=next_depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_action', data=next_action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('return', data=return_arrays, chunks=timeout_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('reward', data=reward_arrays, chunks=reward_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('done', data=done_arrays, chunks=done_chunk_size, dtype='bool', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('timeout', data=timeout_arrays, chunks=timeout_chunk_size, dtype='bool', overwrite=True, compressor=compressor)

        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
        
        
        # print shape
        cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
        cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
        cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')

        cprint(f'next_img shape: {next_img_arrays.shape}, range: [{np.min(next_img_arrays)}, {np.max(next_img_arrays)}]', 'green')
        cprint(f'next_point_cloud shape: {next_point_cloud_arrays.shape}, range: [{np.min(next_point_cloud_arrays)}, {np.max(next_point_cloud_arrays)}]', 'green')
        cprint(f'next_depth shape: {next_depth_arrays.shape}, range: [{np.min(next_depth_arrays)}, {np.max(next_depth_arrays)}]', 'green')
        cprint(f'next_state shape: {next_state_arrays.shape}, range: [{np.min(next_state_arrays)}, {np.max(next_state_arrays)}]', 'green')
        cprint(f'next_action shape: {next_action_arrays.shape}, range: [{np.min(next_action_arrays)}, {np.max(next_action_arrays)}]', 'green')
        
        cprint(f'reward shape: {reward_arrays.shape}, range: [{np.min(reward_arrays)}, {np.max(reward_arrays)}]', 'green')
        cprint(f'done shape: {done_arrays.shape}, range: [{np.min(done_arrays)}, {np.max(done_arrays)}]', 'green')
        cprint(f'timeout shape: {timeout_arrays.shape}, range: [{np.min(timeout_arrays)}, {np.max(timeout_arrays)}]', 'green')
        cprint(f'return shape: {return_arrays.shape}, range: [{np.min(return_arrays)}, {np.max(return_arrays)}]', 'green')
        cprint(f'Saved zarr file to {save_dir}', 'green')
        
        cprint(f'Saved zarr file to {save_dir}', 'green')
        
        # clean up
        del img_arrays, state_arrays, point_cloud_arrays, action_arrays, reward_arrays, done_arrays, timeout_arrays, episode_ends_arrays
        del zarr_root, zarr_data, zarr_meta
        # del env, expert_agent
@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs, resume: bool = True, num_frames = 390000):   
    from load_dmc import Workspace as W
    root_dir = Path.cwd()
    workspace = Workspace(cfgs)
    if resume:
        if num_frames != None:
            folder = DRM_ROOT / 'exp_local' / cfgs.task_name
            pt_files = [f for f in os.listdir(folder) if f.endswith('.pt')]
            # snapshot = DRM_ROOT / 'exp_local' / cfgs.task_name / f'snapshot_{num_frames}.pt'
            # snapshot = Path(snapshot)
            for pt_name in pt_files:
                snapshot = folder / pt_name
                print(f'resuming: {snapshot}')
                workspace.load_snapshot(snapshot)
                workspace.gen_demo(num_episodes=3, data_property='medium', pt_name=pt_name)
        else:
            snapshot = root_dir / 'snapshot.pt'
        if os.path.exists(snapshot):
            print(f'resuming: {snapshot}')
            workspace.load_snapshot(snapshot)
            # workspace.eval_checkpoint()
            workspace.gen_demo(num_episodes=30, data_property='medium')
            return workspace
        else:
            print(f'no snapshot found at {snapshot}')
            workspace.train()   
    else:
        workspace.train()

if __name__ == '__main__':
    main()
