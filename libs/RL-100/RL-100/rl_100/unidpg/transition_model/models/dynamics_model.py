import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from rl_100.unidpg.transition_model.models.nets import EnsembleLinear
from rl_100.model.action_ae import ActionChunkEncoder, ActionChunkDecoder


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu",
        cfg = None
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self._device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)
        module_list = []

        action_embed_layer_norm = getattr(getattr(cfg, "dynamics", None), "action_embed_layer_norm", False)
        action_scale_norm = getattr(getattr(cfg, "dynamics", None), "action_scale_norm", False)
        self.use_conv_action_embed = getattr(cfg, 'use_conv_action_embed', False)
        # raw-action LayerNorm only applies to raw concat path (no embed, no conv embed)
        self.use_action_scale_norm = bool(
            action_scale_norm and (not self.use_conv_action_embed) and (not cfg.use_action_embed)
        )
        if self.use_conv_action_embed:
            n_action_steps = getattr(cfg, 'n_action_steps', 16)
            single_action_dim = action_dim // n_action_steps
            conv_latent_cz = getattr(cfg, 'conv_latent_cz', 32)
            conv_hidden_dims = list(getattr(cfg, 'conv_hidden_dims', [128, 256]))
            conv_kernel_size = getattr(cfg, 'conv_kernel_size', 5)
            conv_n_groups = getattr(cfg, 'conv_n_groups', 8)
            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim, hidden_dims=conv_hidden_dims,
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
            )
            self._conv_action_decoder = ActionChunkDecoder(
                action_dim=single_action_dim, hidden_dims=list(reversed(conv_hidden_dims)),
                latent_cz=conv_latent_cz, kernel_size=conv_kernel_size, n_groups=conv_n_groups,
                target_len=n_action_steps,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = nn.LayerNorm(conv_out_dim) if action_embed_layer_norm else nn.Identity()
            hidden_dims = [obs_dim + conv_out_dim] + list(hidden_dims)
            self._single_action_dim = single_action_dim
            self._n_action_steps = n_action_steps
        elif cfg.use_action_embed:
            action_encoder_layers = [nn.Linear(action_dim, int(obs_dim))]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(int(obs_dim)))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)
            hidden_dims = [obs_dim+int(obs_dim)] + list(hidden_dims)
        else:
            if self.use_action_scale_norm:
                self._action_scale_layer_norm = nn.LayerNorm(action_dim)
            hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self._device)

    def forward(self, obs_action, targets=None, action_chunk=None,
                logvar_loss_coef=0.01, action_recon_beta=0.5):
        """
        Unified forward for both inference and conv AE training.

        Inference mode (default): obs_action is pre-concatenated [obs, action_embed].
            Returns (mean, logvar).
        Conv AE training mode: pass targets and action_chunk.
            Returns scalar loss (dynamics + recon).
        """
        if action_chunk is not None and targets is not None:
            # Conv AE training path: encode -> dynamics -> recon, all in one forward
            batch_size = action_chunk.shape[0]
            z = self._conv_action_encoder(action_chunk)
            action_embed = self._conv_action_layer_norm(z.reshape(batch_size, -1))
            obs_act = torch.cat([obs_action, action_embed], dim=-1)
            output = obs_act
            for layer in self.backbones:
                output = self.activation(layer(output))
            mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (torch.pow(mean - targets, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.get_decay_loss()
            loss = loss + logvar_loss_coef * self.max_logvar.sum() - logvar_loss_coef * self.min_logvar.sum()
            a_recon = self._conv_action_decoder(z)
            recon_loss = F.mse_loss(a_recon, action_chunk)
            loss = loss + action_recon_beta * recon_loss
            return loss

        # Inference path (unchanged)
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self._device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
