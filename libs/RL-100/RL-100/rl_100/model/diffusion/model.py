# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import cprint
from rl_100.model.common.mlp import MLP, ResidualMLP
from rl_100.unidpg.diffusion_policy.helpers import SinusoidalPosEmb


class MLPResNetBlock(nn.Module):
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.fc_residual = nn.Linear(features, features) if features != features else None
        
        self.layer_norm = nn.LayerNorm(features) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        if self.fc_residual is not None:
            residual = self.fc_residual(residual)
        
        return residual + x

# 定义MLPResNet
class MLPResNet(nn.Module):
    def __init__(self, state_dim, action_dim, depth=3, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, act='mish', 
                 t_dim=16, n_action_steps=1):
        super(MLPResNet, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU()
            print("Using ReLU activation in MLP decoder")
        elif act == 'mish':
            self.act = nn.Mish()
            print("Using Mish activation in MLP decoder")
        else:
            raise ValueError("Activation function not supported")
        self.num_blocks = depth
        self.n_action_steps = n_action_steps
        self.single_action_dim = action_dim
        action_dim = action_dim * n_action_steps
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = self.act
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + action_dim + t_dim
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=self.activations, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate) for _ in range(depth)])
        self.fc_final = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        # import pdb; pdb.set_trace()        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t = self.time_mlp(timesteps)
        # Fix tensor shape mismatch - flatten sample properly
        if len(sample.shape) == 3:
            # sample is [batch_size, horizon, action_dim], flatten to [batch_size, horizon*action_dim] 
            sample_flat = sample.view(sample.shape[0], -1)
        else:
            # sample is already [batch_size, action_dim]
            sample_flat = sample.squeeze(1)
        x = torch.cat([sample_flat, t, global_cond], dim=1)
        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.activations(x)
        x = self.fc_final(x)
        x = x.view(sample.shape[0], self.n_action_steps, self.single_action_dim)
        return x
# 定义MLPResNet
class MLPResNet_T(nn.Module):
    def __init__(self, obs_feature_dim, action_dim, predict_r = False, use_action_embed = False, depth=3, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, act='mish', 
                 t_dim=16, chunk_as_single_action=False, n_action_steps=1, action_embed_dim=None, action_embed_layer_norm=False):
        super(MLPResNet_T, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU()
            print("Using ReLU activation in MLP decoder")
        elif act == 'mish':
            self.act = nn.Mish()
            print("Using Mish activation in MLP decoder")
        else:
            raise ValueError("Activation function not supported")
        
        # Action embedding dimension (default to obs_feature_dim if not specified)
        self.action_embed_dim = action_embed_dim if action_embed_dim is not None else obs_feature_dim
        
        # Calculate cond_dim based on action embedding
        if use_action_embed:
            # Whole chunk action embedded to single latent
            cond_dim = self.action_embed_dim + obs_feature_dim
        else:
            # Raw action input
            if chunk_as_single_action:
                cond_dim = action_dim * n_action_steps + obs_feature_dim
            else:
                cond_dim = action_dim + obs_feature_dim
        
        target_dim = obs_feature_dim + predict_r
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.num_blocks = depth
        self.out_dim = target_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = self.act
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim), 
        )
        input_dim = cond_dim + target_dim + t_dim
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=self.activations, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate) for _ in range(depth)])
        self.fc_final = nn.Linear(hidden_dim, target_dim)
        self.use_action_embed = use_action_embed
        self.chunk_as_single_action = chunk_as_single_action
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        
        if use_action_embed:
            # Embed entire chunk action to single latent
            input_action_dim = action_dim * n_action_steps if chunk_as_single_action else action_dim
            action_encoder_layers = [nn.Linear(input_action_dim, self.action_embed_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(self.action_embed_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)
            print(f"Action encoder: [{input_action_dim}] -> [{self.action_embed_dim}]")
        
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        # import pdb; pdb.set_trace()        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t = self.time_mlp(timesteps)
        # import pdb; pdb.set_trace()
        x = torch.cat([sample, t, global_cond], dim=1)
        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.activations(x)
        x = self.fc_final(x)
        return x

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 device,
                 depth=2,
                 t_dim=16,
                 act='mish',
                 use_layer_norm=False,
                 ):

        super(MLP, self).__init__()
        self.device = device
        if act == 'relu':
            self.act = nn.ReLU()
            print("Using ReLU activation in MLP decoder")
        elif act == 'mish':
            self.act = nn.Mish()
            print("Using Mish activation in MLP decoder")
        else:
            raise ValueError("Activation function not supported")
        if use_layer_norm:
            cprint("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! using layer norm in MLP policy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", 'green')
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            self.act,
            nn.Linear(t_dim * 2, t_dim),
        )
        self.use_layer_norm = use_layer_norm
        input_dim = state_dim + action_dim + t_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(), self.act]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim)) if use_layer_norm else nn.Identity(),
            layers.append(self.act)
        self.mid_layer = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        # import pdb; pdb.set_trace()        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t = self.time_mlp(timesteps)
        # Fix tensor shape mismatch - flatten sample properly
        if len(sample.shape) == 3:
            # sample is [batch_size, horizon, action_dim], flatten to [batch_size, horizon*action_dim] 
            sample_flat = sample.view(sample.shape[0], -1)
        else:
            # sample is already [batch_size, action_dim]
            sample_flat = sample.squeeze(1)
        x = torch.cat([sample_flat, t, global_cond], dim=1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        return x.unsqueeze(1)


class ViTMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        action_dim,
        state_dim,
        action_steps,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()

        # diffusion
        input_dim = (
            time_dim + action_dim * action_steps + state_dim
        )
        self.input_dim = input_dim
        output_dim = action_dim * action_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        sample,
        timestep,
        local_cond,
        global_cond,
        **kwargs,
    ):
        """
        sample: (B, Ta, Da)
        timestep: (B,) or int, diffusion step
        global_cond: feature of cond, (B, cond_dim)
        TODO long term: more flexible handling of cond
        """
        # append time and cond
        B, Ta, Da = sample.shape
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        # time = timestep.view(B, 1)
        time_emb = self.time_embedding(timesteps).view(B, self.time_dim)
        sample = sample.view(B, -1)
        sample = torch.cat([sample, time_emb, global_cond], dim=-1)

        # mlp
        out = self.mlp_mean(sample)
        return out.view(B, Ta, Da)
