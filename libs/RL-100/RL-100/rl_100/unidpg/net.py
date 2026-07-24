import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

from rl_100.model.action_ae import ActionChunkEncoder, ActionChunkDecoder


def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    #x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = 'relu',
    final_activation: str = None,
    use_layer_norm: bool = False,
) -> torch.nn.modules.container.Sequential:


    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()

    layers = [nn.Linear(input_dim, hidden_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act_f)
    if depth -1 > 0:
        for _ in range(depth -1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_f)

    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers

    return nn.Sequential(*layers)

# def MLP(
#     input_dim: int,
#     hidden_dim: int,
#     depth: int,
#     output_dim: int,
#     final_activation: str = None
# ) -> torch.nn.modules.container.Sequential:

#     layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
#     for _ in range(depth -1):
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         layers.append(nn.ReLU())
#     layers.append(nn.Linear(hidden_dim, output_dim))
#     if final_activation == 'relu':
#         layers.append(nn.ReLU())
#     elif final_activation == 'tanh':
#         layers.append(nn.Tanh())
#     else:
#         layers = layers
#     return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, obs_encoder, state_dim: int, hidden_dim: int, depth: int, n_obs_steps: int = 1, fix_encoder: bool = False
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim
        self._net = MLP(state_dim, hidden_dim, depth - 1, 1)
        if obs_encoder != None:
            self._obs_encoder = obs_encoder
            if fix_encoder:
                self._obs_encoder.eval()
                for param in self._obs_encoder.parameters():
                    param.requires_grad = False
        else:
            self._obs_encoder = None
    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(state, dict):
            batch_size = state['agent_pos'].shape[0]
        else:
            batch_size = state.shape[0]
        if self._obs_encoder != None and isinstance(state, dict):# do not share encoder, i.e., s - obs. else s - feature
            state = self._obs_encoder(state).reshape(-1, self.state_dim) 
        else:
            state = state.reshape(-1, self.state_dim)  # cat the obs feature from n_obs_steps 
        return self._net(state)



class QMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, use_action_embed, obs_encoder, state_dim: int, feature_dim: int, action_dim: int, hidden_dim: int, depth:int, fix_encoder: bool = False,
        use_conv_action_embed: bool = False, single_action_dim: int = None, n_action_steps: int = 16,
        conv_hidden_dims: list = [128, 256], conv_latent_cz: int = 32, conv_kernel_size: int = 5, conv_n_groups: int = 8,
        q_layer_norm: bool = False, action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_action_embed = use_action_embed
        self.use_conv_action_embed = use_conv_action_embed
        self.state_dim = state_dim
        self.action_dim = action_dim
        # raw-action LayerNorm only applies to the raw concat path
        self.use_action_scale_norm = bool(
            action_scale_norm and (not use_action_embed) and (not use_conv_action_embed)
        )

        if use_conv_action_embed:
            assert single_action_dim is not None
            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim, hidden_dims=conv_hidden_dims,
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
            )
            self._conv_action_decoder = ActionChunkDecoder(
                action_dim=single_action_dim, hidden_dims=list(reversed(conv_hidden_dims)),
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
                target_len=n_action_steps,
            )
            # Compute actual conv output dim via dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = nn.LayerNorm(conv_out_dim) if action_embed_layer_norm else nn.Identity()
            self._net = MLP((state_dim + conv_out_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
        else:
            action_encoder_layers = [nn.Linear(action_dim, feature_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(feature_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)
            if self.use_action_embed:
                self._net = MLP((state_dim + feature_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
            else:
                if self.use_action_scale_norm:
                    self._action_scale_layer_norm = nn.LayerNorm(action_dim)
                self._net = MLP((state_dim + action_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)

        if obs_encoder != None:
            self._obs_encoder = obs_encoder
            if fix_encoder:
                self._obs_encoder.eval()
                for param in self._obs_encoder.parameters():
                    param.requires_grad = False
        else:
            self._obs_encoder = None

    def encode_action(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_embed, recon) for conv mode, (action_embed, None) otherwise."""
        if self.use_conv_action_embed:
            # a: [B, H*Da] -> [B, H, Da]
            B = a.shape[0]
            Da = self._conv_action_encoder.action_dim
            a_chunk = a.reshape(B, -1, Da)
            z = self._conv_action_encoder(a_chunk)  # [B, Tz, Cz]
            recon = self._conv_action_decoder(z)     # [B, H, Da]
            return self._conv_action_layer_norm(z.reshape(B, -1)), recon.reshape(B, -1)
        elif self.use_action_embed:
            return self._action_encoder(a.reshape(-1, self.action_dim)), None
        else:
            a_flat = a.reshape(-1, self.action_dim)
            if self.use_action_scale_norm:
                a_flat = self._action_scale_layer_norm(a_flat)
            return a_flat, None

    def compute_action_recon_loss(self, a: torch.Tensor) -> torch.Tensor:
        """Conv action AE reconstruction loss. a: [B, H*Da] flattened."""
        if not self.use_conv_action_embed:
            return torch.tensor(0.0, device=a.device)
        a_flat = a.reshape(a.shape[0], -1)
        _, a_recon = self.encode_action(a_flat)
        return torch.nn.functional.mse_loss(a_recon, a_flat)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor, return_action_recon_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(s, dict):
            batch_size = s['agent_pos'].shape[0]
        else:
            batch_size = s.shape[0]
        if self._obs_encoder != None:
            s = self._obs_encoder(s).reshape(-1, self.state_dim)
        a_embed, a_recon = self.encode_action(a)
        sa = torch.cat([s, a_embed], dim=1)
        q = self._net(sa)
        if return_action_recon_loss:
            if self.use_conv_action_embed and a_recon is not None:
                action_recon_loss = torch.nn.functional.mse_loss(a_recon, a.reshape(a.shape[0], -1))
            else:
                action_recon_loss = torch.tensor(0.0, device=a.device)
            return q, action_recon_loss
        return q

class DoubleQMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, use_action_embed, obs_encoder, state_dim: int, feature_dim: int, action_dim: int, hidden_dim: int, depth:int, fix_encoder: bool = False,
        use_conv_action_embed: bool = False, single_action_dim: int = None, n_action_steps: int = 16,
        conv_hidden_dims: list = [128, 256], conv_latent_cz: int = 32, conv_kernel_size: int = 5, conv_n_groups: int = 8,
        q_layer_norm: bool = False, action_embed_layer_norm: bool = False,
        action_scale_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_action_embed = use_action_embed
        self.use_conv_action_embed = use_conv_action_embed
        self.action_dim = action_dim
        self.state_dim = state_dim
        # raw-action LayerNorm only applies to the raw concat path
        self.use_action_scale_norm = bool(
            action_scale_norm and (not use_action_embed) and (not use_conv_action_embed)
        )

        if use_conv_action_embed:
            assert single_action_dim is not None
            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim, hidden_dims=conv_hidden_dims,
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
            )
            self._conv_action_decoder = ActionChunkDecoder(
                action_dim=single_action_dim, hidden_dims=list(reversed(conv_hidden_dims)),
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
                target_len=n_action_steps,
            )
            # Compute actual conv output dim via dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = nn.LayerNorm(conv_out_dim) if action_embed_layer_norm else nn.Identity()
            self._net1 = MLP((state_dim + conv_out_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
            self._net2 = MLP((state_dim + conv_out_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
        else:
            action_encoder_layers = [nn.Linear(action_dim, feature_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(feature_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)
            if self.use_action_embed:
                self._net1 = MLP((state_dim + feature_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
                self._net2 = MLP((state_dim + feature_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
            else:
                if self.use_action_scale_norm:
                    self._action_scale_layer_norm = nn.LayerNorm(action_dim)
                self._net1 = MLP((state_dim + action_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)
                self._net2 = MLP((state_dim + action_dim), hidden_dim, depth - 1, 1, use_layer_norm=q_layer_norm)

        if obs_encoder != None:
            self._obs_encoder = obs_encoder
            if fix_encoder:
                self._obs_encoder.eval()
                for param in self._obs_encoder.parameters():
                    param.requires_grad = False
        else:
            self._obs_encoder = None

    def encode_action(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_embed, recon) for conv mode, (action_embed, None) otherwise."""
        if self.use_conv_action_embed:
            B = a.shape[0]
            Da = self._conv_action_encoder.action_dim
            a_chunk = a.reshape(B, -1, Da)
            z = self._conv_action_encoder(a_chunk)
            recon = self._conv_action_decoder(z)
            return self._conv_action_layer_norm(z.reshape(B, -1)), recon.reshape(B, -1)
        elif self.use_action_embed:
            return self._action_encoder(a.reshape(-1, self.action_dim)), None
        else:
            a_flat = a.reshape(-1, self.action_dim)
            if self.use_action_scale_norm:
                a_flat = self._action_scale_layer_norm(a_flat)
            return a_flat, None

    def compute_action_recon_loss(self, a: torch.Tensor) -> torch.Tensor:
        """Conv action AE reconstruction loss. a: [B, H*Da] flattened."""
        if not self.use_conv_action_embed:
            return torch.tensor(0.0, device=a.device)
        a_flat = a.reshape(a.shape[0], -1)
        _, a_recon = self.encode_action(a_flat)
        return torch.nn.functional.mse_loss(a_recon, a_flat)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor, return_action_recon_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(s, dict):
            batch_size = s['agent_pos'].shape[0]
        else:
            batch_size = s.shape[0]
        if self._obs_encoder != None:
            s = self._obs_encoder(s).reshape(-1, self.state_dim)
        a_embed, a_recon = self.encode_action(a)
        sa = torch.cat([s, a_embed], dim=1)
        q1, q2 = self._net1(sa), self._net2(sa)
        if return_action_recon_loss:
            if self.use_conv_action_embed and a_recon is not None:
                action_recon_loss = torch.nn.functional.mse_loss(a_recon, a.reshape(a.shape[0], -1))
            else:
                action_recon_loss = torch.tensor(0.0, device=a.device)
            return q1, q2, action_recon_loss
        return q1, q2

class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, pi_activation_f = 'relu'
    ) -> None:
        super().__init__()
        if pi_activation_f == 'relu':
            print('using relu as activation function!!!')
        elif pi_activation_f == 'tanh':
            print('using tanh as activation function!!!') 
        self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim), pi_activation_f, 'tanh')
        self._log_std_bound = (-5., 0.)
        for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist
    
    def predict(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist, mu, std
