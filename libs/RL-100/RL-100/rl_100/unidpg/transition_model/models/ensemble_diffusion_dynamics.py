"""
Ensemble Diffusion Dynamics Model

This module wraps the LDDM (Latent Diffusion Dynamics Model) to be compatible 
with the EnsembleDynamics_batch interface used in ensemble_dynamics_for_batch.py.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from copy import deepcopy
from rl_100.model.action_ae import ActionChunkEncoder


class EnsembleDiffusionDynamicsModel(nn.Module):
    """
    Ensemble wrapper for Diffusion Dynamics Model.
    
    This class wraps LDDM to provide the same interface as EnsembleDynamicsModel,
    making it compatible with ensemble_dynamics_for_batch.py.
    
    Key differences from MLP ensemble:
    - Uses diffusion model for prediction (stochastic by nature)
    - Can use multiple diffusion models for ensemble, or single model with multiple samples
    - Returns (mean, logvar) format for compatibility, with fixed/learned logvar
    """
    is_diffusion_dynamics = True
    
    def __init__(
        self,
        lddm_model: nn.Module,  # LDDM instance
        obs_dim: int,
        action_dim: int,
        num_ensemble: int = 7,
        num_elites: int = 5,
        with_reward: bool = True,
        device: str = "cpu",
        use_true_ensemble: bool = False,  # If True, create multiple LDDM models
        cfg = None
    ) -> None:
        super().__init__()
        
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self._device = torch.device(device)
        self.use_true_ensemble = use_true_ensemble
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if use_true_ensemble:
            # Create multiple LDDM models
            self.models = nn.ModuleList([deepcopy(lddm_model) for _ in range(num_ensemble)])
            self.model = self.models[0]  # Keep reference for compatibility
        else:
            # Use single model, sample multiple times for ensemble effect
            self.model = lddm_model
            self.models = None
        
        # Action encoder: use the one from inner LDDM.model (MLPResNet_T) if available
        # Otherwise create a new one for compatibility
        self._setup_action_encoder(lddm_model, cfg, action_dim, obs_dim)
        
        # Learnable logvar bounds (for compatibility)
        output_dim = obs_dim + (1 if with_reward else 0)
        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(output_dim) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(output_dim) * -10, requires_grad=True)
        )
        
        # Elite indices
        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )
        
        # For saving/loading best models
        self._saved_state_dicts = {}
        
        self.to(self._device)
    
    def _setup_action_encoder(self, lddm_model, cfg, action_dim, obs_dim):
        """
        Setup action encoder by reusing the one from inner LDDM model if available.
        The inner LDDM uses MLPResNet_T which has _action_encoder defined.
        """
        self.use_conv_action_embed = bool(getattr(cfg, 'use_conv_action_embed', False)) if cfg is not None else False
        if self.use_conv_action_embed:
            action_embed_layer_norm = getattr(getattr(cfg, "dynamics", None), "action_embed_layer_norm", False)
            n_action_steps = getattr(cfg, 'n_action_steps', 16)
            single_action_dim = action_dim // n_action_steps
            conv_latent_cz = getattr(cfg, 'conv_latent_cz', 32)
            conv_hidden_dims = list(getattr(cfg, 'conv_hidden_dims', [128, 256]))
            conv_kernel_size = getattr(cfg, 'conv_kernel_size', 5)
            conv_n_groups = getattr(cfg, 'conv_n_groups', 8)
            self._conv_action_encoder = ActionChunkEncoder(
                action_dim=single_action_dim,
                hidden_dims=conv_hidden_dims,
                latent_cz=conv_latent_cz,
                kernel_size=conv_kernel_size,
                n_groups=conv_n_groups,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, n_action_steps, single_action_dim)
                conv_out_dim = self._conv_action_encoder(dummy).reshape(1, -1).shape[-1]
            self._conv_action_layer_norm = nn.LayerNorm(conv_out_dim) if action_embed_layer_norm else nn.Identity()
            self._single_action_dim = single_action_dim
            self._n_action_steps = n_action_steps
            self._action_encoder = None
            print(f"Created conv action encoder in EnsembleDiffusionDynamicsModel: [{action_dim}] -> [{conv_out_dim}]")
            return

        inner_encoder = None
        
        # Try to find _action_encoder in the nested structure: LDDM.model (MLPResNet_T)
        if hasattr(lddm_model, 'model') and hasattr(lddm_model.model, '_action_encoder'):
            inner_encoder = lddm_model.model._action_encoder
        elif hasattr(lddm_model, '_action_encoder'):
            inner_encoder = lddm_model._action_encoder
        
        if inner_encoder is not None:
            # Reuse the existing action encoder from inner model
            self._action_encoder = inner_encoder
            print("Using action encoder from inner LDDM model")
        elif cfg is not None and getattr(cfg, 'use_action_embed', False):
            # Create a new one if inner model doesn't have it but config requires it
            action_embed_layer_norm = getattr(getattr(cfg, "dynamics", None), "action_embed_layer_norm", False)
            action_encoder_layers = [nn.Linear(action_dim, obs_dim)]
            if action_embed_layer_norm:
                action_encoder_layers.append(nn.LayerNorm(obs_dim))
            action_encoder_layers.append(nn.ReLU())
            self._action_encoder = nn.Sequential(*action_encoder_layers)
            print("Created new action encoder in EnsembleDiffusionDynamicsModel")
        else:
            # No action encoder needed
            self._action_encoder = None
    
    def set_device(self, device):
        """Set the device for the model."""
        self._device = torch.device(device)
        self.to(self._device)
        if self.use_true_ensemble:
            for model in self.models:
                if hasattr(model, 'set_device'):
                    model.set_device(device)
        else:
            if hasattr(self.model, 'set_device'):
                self.model.set_device(device)
    
    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with EnsembleDynamicsModel interface.
        
        Args:
            obs_action: Input tensor of shape [B, obs_dim + action_dim]
            
        Returns:
            mean: [num_ensemble, B, output_dim] - Predictions from each ensemble member
            logvar: [num_ensemble, B, output_dim] - Log variance estimates
        """
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self._device)
        batch_size = obs_action.shape[0]
        output_dim = self.obs_dim + (1 if self._with_reward else 0)
        
        if self.use_true_ensemble:
            # Use multiple models
            means = []
            for model in self.models:
                if hasattr(model, 'model'):
                    # LDDM wrapper
                    output = model(obs_action)  # [B, output_dim]
                else:
                    output = model(obs_action)
                means.append(output)
            mean = torch.stack(means, dim=0)  # [num_ensemble, B, output_dim]
        else:
            # Single model, sample multiple times
            means = []
            for _ in range(self.num_ensemble):
                output = self.model(obs_action)  # [B, output_dim]
                means.append(output)
            mean = torch.stack(means, dim=0)  # [num_ensemble, B, output_dim]
        
        # Compute logvar from ensemble variance or use fixed value
        if self.num_ensemble > 1:
            # Use empirical variance from ensemble
            variance = torch.var(mean, dim=0, keepdim=True).expand(self.num_ensemble, -1, -1)
            logvar = torch.log(variance + 1e-6)
        else:
            # Fixed logvar
            logvar = torch.zeros_like(mean)
        
        # Clamp logvar
        logvar = self._soft_clamp(logvar, self.min_logvar, self.max_logvar)
        
        return mean, logvar
    
    def _soft_clamp(self, x: torch.Tensor, _min: torch.Tensor, _max: torch.Tensor) -> torch.Tensor:
        """Soft clamping to maintain gradients."""
        x = _max - F.softplus(_max - x)
        x = _min + F.softplus(x - _min)
        return x
    
    def compute_loss(self, condition_batch: torch.Tensor, targets_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion loss for training.
        
        Args:
            condition_batch: Input conditions [B, obs_dim + action_dim]
            targets_batch: Target outputs [B, output_dim]
            
        Returns:
            loss: Scalar loss value
        """
        if self.use_true_ensemble:
            # Train all models
            total_loss = 0
            for model in self.models:
                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(condition_batch, targets_batch)
                else:
                    # Fallback: MSE loss with model forward
                    pred = model(condition_batch)
                    loss = F.mse_loss(pred, targets_batch)
                total_loss += loss
            return total_loss / len(self.models)
        else:
            if hasattr(self.model, 'compute_loss'):
                return self.model.compute_loss(condition_batch, targets_batch)
            else:
                pred = self.model(condition_batch)
                return F.mse_loss(pred, targets_batch)
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        """Randomly select elite model indices for each batch element."""
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    
    def set_elites(self, indexes: List[int]) -> None:
        """Set the elite model indices."""
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def load_save(self) -> None:
        """Load saved state dicts for all models."""
        if self.use_true_ensemble:
            for i, model in enumerate(self.models):
                if i in self._saved_state_dicts:
                    model.load_state_dict(self._saved_state_dicts[i])
        else:
            if 0 in self._saved_state_dicts:
                self.model.load_state_dict(self._saved_state_dicts[0])
    
    def update_save(self, indexes: List[int]) -> None:
        """Save state dicts for specified model indices."""
        if self.use_true_ensemble:
            for i in indexes:
                if i < len(self.models):
                    self._saved_state_dicts[i] = deepcopy(self.models[i].state_dict())
        else:
            # For single model, just save the current state
            if 0 in indexes or len(indexes) == 0:
                self._saved_state_dicts[0] = deepcopy(self.model.state_dict())
    
    def get_decay_loss(self) -> torch.Tensor:
        """Get weight decay loss (returns 0 for diffusion model)."""
        # Diffusion models typically don't use explicit weight decay loss
        return torch.tensor(0.0, device=self._device)
