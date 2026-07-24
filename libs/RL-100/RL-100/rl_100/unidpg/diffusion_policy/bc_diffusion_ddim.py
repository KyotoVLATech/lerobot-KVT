# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler

from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob_dpok import DDIMSchedulerExtended
from rl_100.unidpg.diffusion_policy.diffusers_patch.lcm_scheduler import LCMScheduler, LCMSchedulerExtended
from rl_100.unidpg.diffusion_policy.model import MLP
from rl_100.unidpg.diffusion_policy.helpers import Losses
import pdb
class Diffusion_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=2e-4,
                 num_inference_steps = 5,
                 is_timesteps = False,
                 ratio = 5, 
                 loss_type='l2',
                 is_iql = True,
                 num_train_timesteps = 100,
                 scheduler = 'ddim',
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        if scheduler == 'ddim':
            self.scheduler = DDIMSchedulerExtended(num_train_timesteps=num_train_timesteps)
            self.scheduler.config.steps_offset = int(1)
        elif scheduler == 'ddpm':
            self.scheduler = DDPMSchedulerExtended(num_train_timesteps=num_train_timesteps)
        elif scheduler == 'lcm':
            if num_train_timesteps < 50:
                original_inference_steps = num_inference_steps
            self.scheduler = LCMSchedulerExtended(num_train_timesteps=num_train_timesteps, original_inference_steps=original_inference_steps)
        else:
            raise ValueError(f"Unsupported scheduler {scheduler}")
        if is_timesteps:
            self.scheduler.config.num_train_timesteps = num_inference_steps * ratio
        self.scheduler.config.steps_offset = int(1)
        self.num_inference_steps = num_inference_steps

        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = Losses[loss_type]()
        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device
        self.is_iql = is_iql
    
    def train_align(self, policy, replay_buffer, optimizer, fix_encoder, iterations, batch_size=512, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

        for step in tqdm(range(int(iterations)), desc='duffusion bc updating ......'): 
            # Sample replay buffer / batch
            batch = replay_buffer.sample(batch_size)
            batch['action'] = batch['action'][:, -1]
            # sample noise that we'll add to the action
            loss, _ = policy.compute_loss(batch, fix_encoder)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, _, _, _, _, _, _ = replay_buffer.sample(batch_size)
            # sample noise that we'll add to the action
            noise = torch.randn_like(action, device=self.device)

            # sample a random timestep for each action
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()

            # add noise to the clean action according to the noise magnitude at each timestep
            noisy_action = self.scheduler.add_noise(action, noise, timestep)
            # pdb.set_trace()
            # predict the noise residual
            pred_action = self.model(noisy_action, timestep, state)
            # pdb.set_trace()
            pred_type = self.scheduler.config.prediction_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = action
            elif pred_type == 'v_prediction':
                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
                # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
                # sigma = self.noise_scheduler.sigmas[timesteps]
                # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
                self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
                self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
                alpha_t, sigma_t = self.noise_scheduler.alpha_t[self.scheduler.timesteps], self.noise_scheduler.sigma_t[self.scheduler.timesteps]
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
                v_t = alpha_t * noise - sigma_t * action
                target = v_t
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")

            loss = self.loss_fn(pred_action, target, 1.0)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric

    def sample_action(self, state, get_np = True, iql = None, Q = None, implicit_policy = False, repeat_num = 100, critic_objective = 'expectile'):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        model = self.model
        scheduler = self.scheduler

        if get_np:
            state = torch.repeat_interleave(state, repeats=repeat_num, dim=0)  
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        # set x_T
        x = torch.randn(shape, device=self.device)
        # set inference timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            # pdb.set_trace()
            # 1. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(x.shape[0])

            model_output = model(x, timesteps, state)
            # pdb.set_trace()
            # compute previous step action: x_t -> x_t-1
            x = scheduler.step(model_output, t, x, ).prev_sample
        
        if get_np:
            if implicit_policy:
                adv = iql.get_advantage(state, x)
                if critic_objective == 'expectile':
                    tau_weights = torch.where(adv > 0, iql._omega, 1 - iql._omega).flatten()
                    # pdb.set_trace()
                    probabilities = tau_weights / tau_weights.sum()
                    sample_idx = torch.multinomial(probabilities, 1)
                    action = x[sample_idx]
                elif critic_objective == 'quantile':
                    tau_weights = torch.where(adv > 0, iql._omega, 1 - iql._omega).flatten()
                    tau_weights = tau_weights.flatten() / adv
                    probabilities = tau_weights / tau_weights.sum()
                    sample_idx = torch.multinomial(probabilities, 1)
                    action = x[sample_idx]
                elif critic_objective == 'exponential':
                    weights = iql._omega * torch.abs(adv * iql._omega) / torch.abs(adv)
                    probabilities = weights.flatten() / weights.sum()
                    sample_idx = torch.multinomial(probabilities, 1)
                    action = x[sample_idx]
                else:
                    raise ValueError(f'Invalid critic objective: {self.critic_objective}')
                return action.cpu().data.numpy().flatten()
            else:
                if self.is_iql:
                    q_value = iql.minQ(state, x).flatten()
                else:
                    q_value = Q(state, x).flatten()
                idx = torch.multinomial(F.softmax(q_value), 1)
                return x[idx].cpu().data.numpy().flatten()
        else:
            return x

    def all_step_logprob(self, state):
        model = self.model
        scheduler = self.scheduler

        
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        # set x_T
        x = torch.randn(shape, device=self.device)
        # set inference timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        all_x, all_next_x, all_logprob = [], [], []
        for t in scheduler.timesteps:
            # pdb.set_trace()
            # 1. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(x.shape[0])

            model_output = model(x, timesteps, state)
            # pdb.set_trace()
            # compute previous step action: x_t -> x_t-1
            all_x.append(x)
            x, log_prob = scheduler.step_logprob(model_output, timesteps, x)
            x = x.prev_sample
            # pdb.set_trace()
            all_next_x.append(x); all_logprob.append(log_prob.unsqueeze(1))
        # x = x.clamp_(-self.max_action, self.max_action)
        return all_x, all_next_x, all_logprob

    def sample_with_logprob(self, state):
        model = self.model
        scheduler = self.scheduler

        
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        # set x_T
        x = torch.randn(shape, device=self.device)
        # set inference timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        all_x, all_next_x, all_logprob = [], [], []
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # pdb.set_trace()
        for t in scheduler.timesteps:
            # pdb.set_trace()
            # 1. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(x.shape[0])

            model_output = model(x, timesteps, state)
            # pdb.set_trace()
            # compute previous step action: x_t -> x_t-1
            all_x.append(x)
            x, log_prob = ddim_step_with_logprob(scheduler, model_output, timesteps, x, eta = 1)
            all_next_x.append(x); all_logprob.append(log_prob.unsqueeze(1))
            # pdb.set_trace()
        return all_x, all_next_x, all_logprob

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.model.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.model.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.model.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.model.load_state_dict(torch.load(f'{dir}/actor.pth'))
        print('model parameters loaded from {}'.format(f'{dir}/actor.pth'))

