# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16,
                 width=64):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, width),
                                       nn.Mish(),
                                       nn.Linear(width, width),
                                       nn.Mish(),
                                       nn.Linear(width, width),
                                       nn.Mish())

        self.final_layer = nn.Linear(width, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)




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
    def __init__(self, state_dim, out_dim, num_blocks=3, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, activations=nn.Mish(), 
                 t_dim=16,):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + out_dim + t_dim
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=activations, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate) for _ in range(num_blocks)])
        self.fc_final = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.activations(x)
        x = self.fc_final(x)
        return x
