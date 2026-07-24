import torch
import numpy as np
from rl_100.common.pytorch_util import dict_apply

class ReplayBuffer:
    def __init__(self, args, shape_info, device, env_num, wo_visual=False, steps_per_update=None):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        # image HWC -> CHW (sync with online_buffer.py)
        if shape_info['obs']['image'][-1] == 3:
            shape_info['obs']['image'] = (
                shape_info['obs']['image'][0],
                shape_info['obs']['image'][-1],
                shape_info['obs']['image'][1],
                shape_info['obs']['image'][2],
            )
        self.shape_info = shape_info
        self.env_num = env_num
        self.args = args
        self.device = device
        self.wo_visual = wo_visual
        self.steps_per_update = steps_per_update if steps_per_update is not None else args.batch_size

    def reset(self):
        wo_visual = self.wo_visual
        shape_info = self.shape_info
        env_num = self.env_num
        args = self.args
        spu = self.steps_per_update
        if not wo_visual:
            self.point_cloud = np.zeros((spu, env_num, *shape_info['obs']['point_cloud']))
            self.image = np.zeros((spu, env_num, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((spu, env_num, *shape_info['obs']['imagin_robot']))
        self.agent_pos = np.zeros((spu, env_num, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((spu, env_num, args.num_inference_steps + 1, *shape_info['action']))
        self.a_logprob = np.zeros((spu, env_num, args.num_inference_steps, *shape_info['action']))

        if not wo_visual:
            self.next_point_cloud = np.zeros((spu, env_num, *shape_info['obs']['point_cloud']))
            self.next_image = np.zeros((spu, env_num, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((spu, env_num, *shape_info['obs']['imagin_robot']))

        self.next_agent_pos = np.zeros((spu, env_num, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((spu, env_num, 1))
        self.done = np.zeros((spu, env_num, 1))
        self.dw = np.zeros((spu, env_num, 1))
        self.count = 0

    def store(self, obs, action, a_logprob, reward, next_obs, done, dw):
        reward = reward.reshape(-1, 1)
        done = np.array(done).reshape(-1, 1)
        dw = np.array(dw).reshape(-1, 1)
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            self.image[self.count] = obs['image']
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.a_logprob[self.count] = a_logprob
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            self.next_image[self.count] = next_obs['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.done[self.count] = done
        self.dw[self.count] = dw
        self.count += 1
    def flatten(self):
        total_size = self.args.batch_size * self.env_num
        if not self.wo_visual:
            self.point_cloud = self.point_cloud.reshape(total_size, *self.point_cloud.shape[2:])
            self.image = self.image.reshape(total_size, *self.image.shape[2:])
            if self.use_imagin_robot:
                self.imagin_robot = self.imagin_robot.reshape(total_size, *self.imagin_robot.shape[2:])
        self.agent_pos = self.agent_pos.reshape(total_size, *self.agent_pos.shape[2:])
        self.action = self.action.reshape(total_size, *self.action.shape[2:])
        self.a_logprob = self.a_logprob.reshape(total_size, *self.a_logprob.shape[2:])
        if not self.wo_visual:
            self.next_point_cloud = self.next_point_cloud.reshape(total_size, *self.next_point_cloud.shape[2:])
            self.next_image = self.next_image.reshape(total_size, *self.next_image.shape[2:])
            if self.use_imagin_robot:
                self.next_imagin_robot = self.next_imagin_robot.reshape(total_size, *self.next_imagin_robot.shape[2:])
        self.next_agent_pos = self.next_agent_pos.reshape(total_size, *self.next_agent_pos.shape[2:])
        self.reward = self.reward.reshape(total_size, *self.reward.shape[2:])
        self.done = self.done.reshape(total_size, *self.done.shape[2:])
        self.dw = self.dw.reshape(total_size, *self.dw.shape[2:])
        self.count = total_size

    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        if not self.wo_visual:
            point_cloud = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            image = torch.FloatTensor(self.image[ind]).to(self.device)
            if self.use_imagin_robot:
                imagin_robot = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
        agent_pos = torch.FloatTensor(self.agent_pos[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)

        if not self.wo_visual:
            if self.use_imagin_robot:
                obs = {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                'image': image, # T, 84, 84, 3
                'imagin_robot': imagin_robot, # T, 96, 7
            }
            else:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action}

    def numpy_to_dict(self):
        if not self.wo_visual:
            if self.use_imagin_robot:
                return {
                    'point_cloud': self.point_cloud,
                    'img': self.image,
                    'imagin_robot': self.imagin_robot,
                    'state': self.agent_pos,
                    'action': self.action,
                    'a_logprob': self.a_logprob,
                    'reward': self.reward,
                    'next_point_cloud': self.next_point_cloud,
                    'next_img': self.next_image,
                    'next_imagin_robot': self.next_imagin_robot,
                    'next_state': self.next_agent_pos,
                    'done': self.done,
                    'dw': self.dw
                }
            else:
                return {
                    'point_cloud': self.point_cloud,
                    'img': self.image,
                    'state': self.agent_pos,
                    'action': self.action,
                    'a_logprob': self.a_logprob,
                    'reward': self.reward,
                    'next_point_cloud': self.next_point_cloud,
                    'next_img': self.next_image,
                    'next_state': self.next_agent_pos,
                    'done': self.done,
                    'dw': self.dw
                }
        else:
            return {
                'state': self.agent_pos,
                'action': self.action,
                'a_logprob': self.a_logprob,
                'reward': self.reward,
                'next_state': self.next_agent_pos,
                'done': self.done,
                'dw': self.dw
            }

    
    def numpy_to_tensor(self):
        if not self.wo_visual:
            point_cloud = torch.tensor(self.point_cloud, dtype=torch.float).to(self.device)
            image = torch.tensor(self.image, dtype=torch.float).to(self.device)
            if self.use_imagin_robot:
                imagin_robot = torch.tensor(self.imagin_robot, dtype=torch.float).to(self.device)
        agent_pos = torch.tensor(self.agent_pos, dtype=torch.float).to(self.device)
        action = torch.tensor(self.action, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        reward = torch.tensor(self.reward, dtype=torch.float).to(self.device)
        if not self.wo_visual:
            next_point_cloud = torch.tensor(self.next_point_cloud, dtype=torch.float).to(self.device)
            next_image = torch.tensor(self.next_image, dtype=torch.float).to(self.device)
            if self.use_imagin_robot:
                next_imagin_robot = torch.tensor(self.next_imagin_robot, dtype=torch.float).to(self.device)
        next_agent_pos = torch.tensor(self.next_agent_pos, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        if not self.wo_visual:
            if self.use_imagin_robot:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                    'imagin_robot': imagin_robot, # T, 96, 7
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                    'imagin_robot': next_imagin_robot, # T, 96, 7
                }
            else:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }

        return obs, action, a_logprob, reward, next_obs, dw, done

    def numpy_to_tensor_vec(self):
        """Return tensors with shape (steps_per_update, env_num, *) — no flatten."""
        n = self.count
        if not self.wo_visual:
            point_cloud = torch.tensor(self.point_cloud[:n], dtype=torch.float).to(self.device)
            image = torch.tensor(self.image[:n], dtype=torch.float).to(self.device)
            if self.use_imagin_robot:
                imagin_robot = torch.tensor(self.imagin_robot[:n], dtype=torch.float).to(self.device)
        agent_pos = torch.tensor(self.agent_pos[:n], dtype=torch.float).to(self.device)
        action = torch.tensor(self.action[:n], dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob[:n], dtype=torch.float).to(self.device)
        reward = torch.tensor(self.reward[:n], dtype=torch.float).to(self.device)
        if not self.wo_visual:
            next_point_cloud = torch.tensor(self.next_point_cloud[:n], dtype=torch.float).to(self.device)
            next_image = torch.tensor(self.next_image[:n], dtype=torch.float).to(self.device)
            if self.use_imagin_robot:
                next_imagin_robot = torch.tensor(self.next_imagin_robot[:n], dtype=torch.float).to(self.device)
        next_agent_pos = torch.tensor(self.next_agent_pos[:n], dtype=torch.float).to(self.device)
        done = torch.tensor(self.done[:n], dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw[:n], dtype=torch.float).to(self.device)
        if not self.wo_visual:
            if self.use_imagin_robot:
                obs = {'point_cloud': point_cloud, 'agent_pos': agent_pos,
                       'image': image, 'imagin_robot': imagin_robot}
                next_obs = {'point_cloud': next_point_cloud, 'agent_pos': next_agent_pos,
                            'image': next_image, 'imagin_robot': next_imagin_robot}
            else:
                obs = {'point_cloud': point_cloud, 'agent_pos': agent_pos, 'image': image}
                next_obs = {'point_cloud': next_point_cloud, 'agent_pos': next_agent_pos, 'image': next_image}
        else:
            obs = {'agent_pos': agent_pos}
            next_obs = {'agent_pos': next_agent_pos}
        return obs, action, a_logprob, reward, next_obs, dw, done

class IqlBuffer:
    def __init__(self, offline_data, args, shape_info,  device, wo_visual=False):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        self.wo_visual = wo_visual
        self.offline_data = offline_data
        if not wo_visual:   
            self.point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            self.image = np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))

        self.agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((args.capacity,  *shape_info['action']))
        if not self.wo_visual:
            self.next_point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            self.next_image = np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))
        self.next_agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((args.capacity, 1))
        self.not_done = np.zeros((args.capacity, 1))
        self.count = 0
        self.capacity = args.capacity
        self.device = device
        self.full = False
    def store(self, obs, action, reward, next_obs, done):
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            self.image[self.count] = obs['image']
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            self.next_image[self.count] = next_obs['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.not_done[self.count] = 1 - done
        self.count = (self.count + 1) % self.capacity
        self.full = self.full or self.count == 0

    def initial_with_dataset(self, dataset):
        dataset = dict_apply(dataset, lambda x: x.cpu().numpy())
        data_size = dataset['action'].shape[0]
        if not self.wo_visual:
            self.point_cloud[:data_size] = dataset['obs']['point_cloud']
            self.image[:data_size] = dataset['obs']['image']
            if self.use_imagin_robot:
                self.imagin_robot[:data_size] = dataset['obs']['imagin_robot']
        self.agent_pos[:data_size] = dataset['obs']['agent_pos']
        self.action[:data_size] = dataset['action']
        self.reward[:data_size] = dataset['reward'].squeeze(1)
        if not self.wo_visual:
            self.next_point_cloud[:data_size] = dataset['next_obs']['point_cloud']
            self.next_image[:data_size] = dataset['next_obs']['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[:data_size] = dataset['next_obs']['imagin_robot']
        self.next_agent_pos[:data_size] = dataset['next_obs']['agent_pos']
        self.not_done[:data_size] = dataset['not_done'].squeeze(1)
        self.count = data_size

    def merge(self, online_batch, offline_batch):
        if not self.wo_visual:
            point_cloud = torch.cat([online_batch['obs']['point_cloud'], offline_batch['obs']['point_cloud']], dim=0)
            image = torch.cat([online_batch['obs']['image'], offline_batch['obs']['image']], dim=0)
            if self.use_imagin_robot:
                imagin_robot = torch.cat([online_batch['obs']['imagin_robot'], offline_batch['obs']['imagin_robot']], dim=0)
        agent_pos = torch.cat([online_batch['obs']['agent_pos'], offline_batch['obs']['agent_pos']], dim=0)
        action = torch.cat([online_batch['action'], offline_batch['action']], dim=0)
        if not self.wo_visual:
            next_point_cloud = torch.cat([online_batch['next_obs']['point_cloud'], offline_batch['next_obs']['point_cloud']], dim=0)
            next_image = torch.cat([online_batch['next_obs']['image'], offline_batch['next_obs']['image']], dim=0)
            if self.use_imagin_robot:
                next_imagin_robot = torch.cat([online_batch['next_obs']['imagin_robot'], offline_batch['next_obs']['imagin_robot']], dim=0)
        next_agent_pos = torch.cat([online_batch['next_obs']['agent_pos'], offline_batch['next_obs']['agent_pos']], dim=0)
        reward = torch.cat([online_batch['reward'], offline_batch['reward'].squeeze(1)], dim=0)
        not_done = torch.cat([online_batch['not_done'], offline_batch['not_done'].squeeze(1)], dim=0)
        if not self.wo_visual:
            if self.use_imagin_robot:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                    'imagin_robot': imagin_robot, # T, 96, 7
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                    'imagin_robot': next_imagin_robot, # T, 96, 7
                }
            else:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}



    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        if not self.wo_visual:
            point_cloud = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            image = torch.FloatTensor(self.image[ind]).to(self.device)
            if self.use_imagin_robot:
                imagin_robot = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
        agent_pos = torch.FloatTensor(self.agent_pos[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        if not self.wo_visual:

            next_point_cloud = torch.FloatTensor(self.next_point_cloud[ind]).to(self.device)
            next_image = torch.FloatTensor(self.next_image[ind]).to(self.device)
            if self.use_imagin_robot:
                next_imagin_robot = torch.FloatTensor(self.next_imagin_robot[ind]).to(self.device)
        next_agent_pos = torch.FloatTensor(self.next_agent_pos[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[ind]).to(self.device)

        if not self.wo_visual:  
            if self.use_imagin_robot:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                    'imagin_robot': imagin_robot, # T, 96, 7
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                    'imagin_robot': next_imagin_robot, # T, 96, 7
                }
            else:

                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}
