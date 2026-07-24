####################### for latent dynamics model traning; input: latent obs & action; output: next latent obs ####################
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
from copy import deepcopy
from rl_100.unidpg.critic import IQL_Q_V_no
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.policy.base_policy import BasePolicy
from rl_100.model.diffusion.conditional_unet1d import ConditionalUnet1D
from rl_100.model.diffusion.simple_conditional_unet1d import SampleConditionalUnet1D
from rl_100.model.diffusion.mask_generator import LowdimMaskGenerator
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.model_util import print_params
from rl_100.model.vision.pointnet_extractor import DP3Encoder, iDP3Encoder, DP3EncoderReconVIB
from rl_100.model.diffusion.model import MLP, MLPResNet, MLPResNet_T
from rl_100.model.vision.torch_layers import PointNetImaginationExtractorGP
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
class LDDM(BasePolicy):
    def __init__(self, 
            shape_meta: Dict[str, int],
            cm_noise_scheduler: DDPMScheduler, # DDIM or CM scheduler
            ddim_noise_scheduler: DDPMScheduler,
            scheduler_type: str,
            num_inference_steps=None,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            model='dp3', # dp3, sample dp3, mlp
            mlp_policy_depth=2,
            act='mish',
            policy_layer_norm=True,
            encoder_output_dim=128, # latent obs feature dim
            use_action_embed=False, # whether use action embedding
            predict_r= False, # whether predict reward
            chunk_as_single_action=False,
            n_action_steps=1,
            action_embed_dim=None, # explicit action embed dim (useful when obs_feature_dim != single-step dim)
            action_embed_layer_norm=False,
            **kwargs):
        super().__init__()
        """
        latent dynamics diffusion model for next obs prediction.
        """
        global_cond_dim = encoder_output_dim
        self.encoder_output_dim = encoder_output_dim
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        self.kwargs = kwargs
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        if model == 'sample_dp3':
            print("Sample DP3 used for the denoising model")
            model = SampleConditionalUnet1D(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )
        elif model == 'dp3':
            model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            )
        elif model == 'mlp':
            model = MLP(
                state_dim=global_cond_dim,
                action_dim=action_dim,
                hidden_dim=diffusion_step_embed_dim,
                depth=mlp_policy_depth,
                device='cuda',
                t_dim=16,
                act=act,
                use_layer_norm=policy_layer_norm,
            )
        elif model == 'skipnet':
            model = MLPResNet_T(
                obs_feature_dim=encoder_output_dim,
                action_dim=action_dim,
                predict_r=predict_r,
                use_action_embed=use_action_embed,
                hidden_dim=diffusion_step_embed_dim,
                depth=mlp_policy_depth,
                t_dim=16,
                act=act,
                use_layer_norm=policy_layer_norm,
                chunk_as_single_action=chunk_as_single_action,
                n_action_steps=n_action_steps,
                action_embed_dim=action_embed_dim,
                action_embed_layer_norm=action_embed_layer_norm,
            )
        self.model = model
        self.target_dim = model.target_dim
        self.cond_dim = model.cond_dim
        if scheduler_type == 'cm':
            self.noise_scheduler = cm_noise_scheduler
        elif scheduler_type == 'ddim':
            self.noise_scheduler = ddim_noise_scheduler
        else:
            raise ValueError(f"Unsupported scheduler type {scheduler_type}")
        
        self.noise_scheduler_pc = copy.deepcopy(self.noise_scheduler)
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=self.target_dim,
        #     obs_dim=0,
        #     max_n_obs_steps=1,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        print_params(self)

    def set_device(self, device):
        """
        Set the device for the model and noise scheduler.
        """
        self._device = device
        self.model.to(device)

    def conditional_sample(self, 
            condition_data, condition_mask, online=False,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            if online:
                trajectory = scheduler.step_mean(
                    model_output, t, trajectory, ).prev_sample
            else:
                trajectory = scheduler.step(
                    model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   
        return trajectory
    
    def forward(self, condition_batch: torch.Tensor, deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        condition_batch: (obs_features, action) # shape: batch_size, feature_dim + action_dim
        Returns: next_obs_features, reward?
        """
        B = condition_batch.shape[0]# batch size, n obs step 
        global_cond = condition_batch # (B, T, cond_dim)
        local_cond = None # no local condition for latent dynamics model
        # build input
        device = self._device
        dtype = self.dtype

        cond_data = torch.zeros(size=(B, self.target_dim), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # run sampling
        outputs = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            online=deterministic,
            **self.kwargs) # (batch_size, horizon, feature_dim)
        
        return outputs  # (B, feature_dim)

    def compute_loss(self, condition_batch, targets_batch):
        """
        condition_batch: (obs_features, action) # shape: batch_size, feature_dim + action_dim
        targets_batch: (next_obs_features, reward?) # shape: batch_size, feature_dim+1? depends on whether use reward signal.
        """
        # handle different ways of passing observation
        local_cond = None
        global_cond = condition_batch
        trajectory = targets_batch
        cond_data = trajectory

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self._device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self._device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
