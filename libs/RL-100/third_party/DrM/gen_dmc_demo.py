# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cheetah_run', help='environment to run')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--expert_ckpt_path', type=str, default=None, help='path to expert ckpt')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--gamma', type=float, default=0.99, help='image size')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    parser.add_argument('--data_property', type=str, default='expert', help='data')
    args = parser.parse_args()
    return args


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

def main():
    args = parse_args()
    # load env
    from omegaconf import OmegaConf
    from load_dmc import load_policy_env
    cfg = OmegaConf.load('cfgs/config.yaml')
    cfg.task = args.env_name
    cfg.task_name = args.env_name
    # cfg.agent = 'drm'
    dmc_workspace = load_policy_env(cfg)

    import pdb; pdb.set_trace()
    num_episodes = args.num_episodes
    save_dir = os.path.join(args.root_dir, 'dmc_'+args.env_name+'_{}.zarr'.format(args.data_property))
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
    minimal_episode_length = 100
    episode_idx = 0
    while episode_idx < num_episodes:
        env = dmc_workspace.eval_env

        obs = env.reset()
        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

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
        

        while (not time_step.last()) or step_count < minimal_episode_length:
            with torch.no_grad(), utils.eval_mode(expert_agent):
                input_obs_visual = time_step.observation
                input_obs_sensor = time_step.observation_sensor
                # cam: top, vil_camera, fixed
                # vrl3_input = render_camera(env.env._env.sim, camera_name="top").transpose(2,0,1).copy() # (3,84,84)
                    
                action = expert_agent.act(obs=input_obs_visual, step=0,
                                        eval_mode=False, 
                                        obs_sensor=input_obs_sensor) # (28,) float32
                
                if args.not_use_multi_view:
                    input_obs_visual = input_obs_visual[:3] # (3,84,84)
                

                        
                # save data
                total_count_sub += 1
                img_arrays_sub.append(input_obs_visual)
                state_arrays_sub.append(input_obs_sensor)
                action_arrays_sub.append(action)
                point_cloud_arrays_sub.append(time_step.observation_pointcloud)
                depth_arrays_sub.append(time_step.observation_depth)
            if time_step.last():
                done_arrays_sub.append(True)
            else:
                done_arrays_sub.append(False)

            time_step = env.step(action)
            reward_arrays_sub.append(time_step.reward)
            
            # next_obs
            next_input_obs_visual = time_step.observation
            if args.not_use_multi_view:
                next_input_obs_visual = next_input_obs_visual[:3] # (3,84,84)
            next_img_arrays_sub.append(next_input_obs_visual)
            next_point_cloud_arrays_sub.append(time_step.observation_pointcloud)
            next_depth_arrays_sub.append(time_step.observation_depth)
            next_state_arrays_sub.append(time_step.observation_sensor)

            obs = time_step.observation # np array, (3,84,84)
            obs = obs[:3] if obs.shape[0] > 3 else obs # (3,84,84)
            n_goal_achieved_total += time_step.n_goal_achieved
            total_reward += time_step.reward
            step_count += 1
            if step_count < minimal_episode_length:
                timeout_arrays_sub.append(False)
            else:
                timeout_arrays_sub.append(True)
        with torch.no_grad(), utils.eval_mode(expert_agent):
            last_next_action = expert_agent.act(obs=time_step.observation, step=0,
                                            eval_mode=True, 
                                            obs_sensor=time_step.observation_sensor) # (28,) float32  
        next_action_arrays_sub = deepcopy(action_arrays_sub)   
            
        next_action_arrays_sub.append(last_next_action)
        next_action_arrays_sub = next_action_arrays_sub[1:]

        if n_goal_achieved_total < 1.:
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
    import pdb; pdb.set_trace()
    return_arrays = compute_return(reward_arrays, not_done_arrays, args.gamma)
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
    del env, expert_agent
    
    
if __name__ == '__main__':
    main()


