"""
Flow matching scheduler extension for RL fine-tuning.

This module extends the official diffusers
`FlowMatchEulerDiscreteScheduler` with the transition log-probability
interfaces required by the existing BPPO / PPO code paths.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    denoised: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None


class FlowMatchSchedulerExtended(FlowMatchEulerDiscreteScheduler):
    """
    Extend the official flow-matching Euler scheduler with:
    - stochastic SDE/CPS transitions for RL rollouts
    - per-element reverse and forward log-prob computation
    - training helpers used by RL1003D BC loss
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        sde_type: str = "sde",
        noise_level: float = 0.7,
        num_inference_steps: int = 10,
        clip_std_min: float = 0.0,
        clip_std_max: float = None,
        sigma_safe_max: float = 0.9,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=invert_sigmas,
            shift_terminal=shift_terminal,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
        )
        self.sde_type = sde_type
        self.noise_level = noise_level
        self.default_num_inference_steps = num_inference_steps
        self.clip_std_min = clip_std_min
        self.clip_std_max = clip_std_max
        self.sigma_safe_max = sigma_safe_max
        self.cps_logprob_mode = "pseudo"  # "pseudo" or "gaussian"; set via rl100_3d.py from config
        self.sde_window_size = 0  # 0 = all stochastic; k > 0 = first k steps stochastic, rest deterministic
        self.flow_noise_on_final_step = False  # whether to inject noise on the last denoising step

        # Save the N-point training sigma grid created by parent __init__ (before
        # set_timesteps overwrites self.sigmas with the inference grid).
        # Grid is descending: index 0 = highest sigma, index N-1 = lowest.
        # Apply the same sigma clamp as set_timesteps to keep consistency.
        self._train_sigmas = torch.clamp(self.sigmas.clone(), max=1.0 - 1e-4)
        self._train_timesteps_int = (
            self._train_sigmas * self.config.num_train_timesteps
        ).long().clamp(0, self.config.num_train_timesteps - 1)

    def is_stochastic_step(self, step_index=None):
        """Return True if the given step should use stochastic SDE/CPS transitions."""
        idx = step_index if step_index is not None else self.step_index
        if self.sde_window_size == 0:
            return True
        return idx < self.sde_window_size

    def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
        """Override parent to clamp sigma_max < 1.0, keeping UNet timesteps in [0, N-1]."""
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        # Clamp sigma_max < 1.0 so that timestep = floor(sigma * N) stays in [0, N-1]
        self.sigmas = torch.clamp(self.sigmas, max=1.0 - 1e-4)
        # Recompute timesteps from clamped sigmas (exclude terminal sigma=0)
        self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)

    def _training_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Map training-time `t in [0, 1]` to the scheduler sigma parameterization."""
        sigma = t.float()
        if self.config.use_dynamic_shifting:
            raise NotImplementedError(
                "Dynamic shifting is not supported in flow BC training helper. "
                "Use a non-dynamic FlowMatch scheduler configuration."
            )
        sigma = self.config.shift * sigma / (1 + (self.config.shift - 1) * sigma)
        if self.config.shift_terminal:
            # Match the inference scheduler's terminal shift stretch in the common scalar case.
            terminal = torch.as_tensor(self.config.shift_terminal, device=sigma.device, dtype=sigma.dtype)
            one = torch.ones_like(sigma)
            sigma = terminal + (one - terminal) * sigma
        if self.config.invert_sigmas:
            sigma = 1.0 - sigma
        return sigma

    @staticmethod
    def _scheduler_timestep(timestep):
        """Collapse batch-expanded timesteps to the scalar form expected by diffusers scheduler internals."""
        if torch.is_tensor(timestep):
            if timestep.ndim == 0:
                return timestep
            return timestep.reshape(-1)[0]
        return timestep

    def add_noise(self, original_samples, noise, timestep_indices):
        """Flow matching forward process using discrete training sigma grid.

        Args:
            timestep_indices: integer tensor with values in [0, num_train_timesteps - 1].
                Index 0 = highest sigma (most noise), index N-1 = lowest sigma.
        """
        sigma = self._train_sigmas.to(device=original_samples.device, dtype=original_samples.dtype)[timestep_indices]
        sigma = sigma.view(-1, *([1] * (len(original_samples.shape) - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    def get_training_target(self, original_samples, noise):
        """Velocity target for the selected flow parameterization."""
        return noise - original_samples

    def get_training_noisy_sample(self, original_samples, noise, timestep_indices):
        """
        Return the noisy sample, velocity target, and model timestep embedding
        used by BC training, using discrete sigma values from the training grid.

        Args:
            timestep_indices: integer tensor with values in [0, num_train_timesteps - 1].
                Index 0 = highest sigma (most noise), index N-1 = lowest sigma.
                The grid is the same N-point sigma schedule built by the parent
                constructor (with shift applied), so training and inference share
                exactly the same sigma ↔ timestep mapping.

        Returns:
            noisy_samples: (1 - sigma) * x_0 + sigma * eps
            target: velocity target (eps - x_0)
            model_timesteps: integer timesteps for UNet embedding, looked up from
                the same grid (no quantization error)
        """
        device = original_samples.device
        sigma = self._train_sigmas.to(device=device, dtype=original_samples.dtype)[timestep_indices]
        model_timesteps = self._train_timesteps_int.to(device=device)[timestep_indices]
        sigma = sigma.view(-1, *([1] * (len(original_samples.shape) - 1)))
        noisy_samples = (1 - sigma) * original_samples + sigma * noise
        target = self.get_training_target(original_samples, noise)
        return noisy_samples, target, model_timesteps

    def _compute_step(self, model_output, sample, prev_sample=None, generator=None):
        """Shared transition computation using the official scheduler sigma grid.
        Uses stochastic SDE/CPS for steps within sde_window_size, deterministic ODE otherwise.
        """
        model_output = model_output.float()
        sample = sample.float()

        step_index = min(self.step_index, len(self.sigmas) - 2)
        next_index = min(step_index + 1, len(self.sigmas) - 1)
        sigma = self.sigmas[step_index].to(sample.device)
        sigma_next = self.sigmas[next_index].to(sample.device)

        expand_shape = [1] * (sample.ndim - 1)
        sigma = sigma.view(*expand_shape)
        sigma_next = sigma_next.view(*expand_shape)
        dt = sigma_next - sigma

        if not self.is_stochastic_step(step_index):
            # Deterministic ODE (Euler) step: x_{t+dt} = x_t + v * dt
            prev_sample_mean = sample + model_output * dt
            std = torch.zeros_like(sigma)
            if prev_sample is None:
                prev_sample = prev_sample_mean
            else:
                prev_sample = prev_sample.float()
            return prev_sample, prev_sample_mean, std

        if self.sde_type == "sde":
            # Clamp sigma to avoid enormous std_dev_t from sqrt(sigma/(1-sigma))
            # near sigma=1. Uses a fixed cap instead of NFE-dependent sigmas[1].
            sigma_safe = torch.clamp(sigma, max=self.sigma_safe_max)
            std_dev_t = torch.sqrt(sigma_safe / (1 - sigma_safe)) * self.noise_level
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * sigma_safe) * dt)
                + model_output * (1 + std_dev_t**2 * (1 - sigma_safe) / (2 * sigma_safe)) * dt
            )
            std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-10))
        elif self.sde_type == "cps":
            std_dev_t = sigma_next * math.sin(self.noise_level * math.pi / 2)
            pred_x0 = sample - sigma * model_output
            pred_x1 = sample + model_output * (1 - sigma)
            prev_sample_mean = pred_x0 * (1 - sigma_next) + pred_x1 * torch.sqrt(
                torch.clamp(sigma_next**2 - std_dev_t**2, min=0.0)
            )
            std = std_dev_t
        else:
            raise ValueError(f"Unknown sde_type: {self.sde_type}")

        if self.clip_std_min > 0:
            std = torch.clamp(std, min=self.clip_std_min)
        if self.clip_std_max is not None:
            std = torch.clamp(std, max=self.clip_std_max)

        if prev_sample is None:
            is_final = self.step_index >= len(self.sigmas) - 2
            should_add_noise = not is_final or self.flow_noise_on_final_step
            if should_add_noise:
                noise = randn_tensor(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype,
                )
                prev_sample = prev_sample_mean + std * noise
            else:
                prev_sample = prev_sample_mean
        else:
            prev_sample = prev_sample.float()

        return prev_sample, prev_sample_mean, std

    @staticmethod
    def _compute_logprob(next_sample, prev_sample_mean, std, sde_type: str,
                         cps_logprob_mode: str = "pseudo"):
        if sde_type == "cps":
            if cps_logprob_mode not in {"pseudo", "gaussian"}:
                raise ValueError(
                    f"Invalid flow_cps_logprob_mode='{cps_logprob_mode}'. "
                    "Must be 'pseudo' or 'gaussian'."
                )
            if cps_logprob_mode == "gaussian":
                # Plan B: true Gaussian log-prob for CPS (experimental)
                std_safe = torch.clamp(std, min=1e-12)
                return (
                    -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_safe**2)
                    - torch.log(std_safe)
                    - 0.5 * math.log(2 * math.pi)
                )
            else:
                # Plan A (default): pseudo-log-prob scaled by 1/action_dim
                log_prob = -((next_sample.detach() - prev_sample_mean) ** 2)
                action_dim = next_sample.shape[-1]
                return log_prob / action_dim
        # SDE: unchanged Gaussian log-prob
        std_safe = torch.clamp(std, min=1e-12)
        return (
            -((next_sample.detach() - prev_sample_mean) ** 2) / (2 * std_safe**2)
            - torch.log(std_safe)
            - 0.5 * math.log(2 * math.pi)
        )

    def step_mean(
        self,
        model_output: torch.FloatTensor,
        timestep,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        prev_sample=None,
        step_index=None,
        eta=1.0,
    ):
        """Deterministic mean step for evaluation.
        CPS: uses CPS mean formula (matches rollout trajectory without noise).
        SDE / default: ODE Euler x_{t+dt} = x_t + v * dt.
        """
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step_mean")
        if self.step_index is None or step_index is not None:
            if step_index is not None:
                self._step_index = step_index
            else:
                self._init_step_index(self._scheduler_timestep(timestep))

        model_output = model_output.float()
        sample = sample.float()
        idx = min(self.step_index, len(self.sigmas) - 2)
        sigma = self.sigmas[idx].to(sample.device)
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)].to(sample.device)

        if self.sde_type == "cps":
            # CPS deterministic eval: use CPS mean formula (same as rollout but without noise)
            # This ensures eval trajectory matches rollout mean trajectory.
            expand_shape = [1] * (sample.ndim - 1)
            sigma = sigma.view(*expand_shape)
            sigma_next = sigma_next.view(*expand_shape)
            std_dev_t = sigma_next * math.sin(self.noise_level * math.pi / 2)
            pred_x0 = sample - sigma * model_output
            pred_x1 = sample + model_output * (1 - sigma)
            prev_sample_mean = pred_x0 * (1 - sigma_next) + pred_x1 * torch.sqrt(
                torch.clamp(sigma_next**2 - std_dev_t**2, min=0.0)
            )
        else:
            # SDE / default: ODE Euler (unchanged)
            dt = sigma_next - sigma
            prev_sample_mean = sample + model_output * dt

        self._step_index += 1
        if return_dict:
            return FlowMatchSchedulerOutput(prev_sample=prev_sample_mean, denoised=prev_sample_mean)
        return (prev_sample_mean, prev_sample_mean)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        eta=1.0,
    ):
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step")
        if self.step_index is None:
            self._init_step_index(self._scheduler_timestep(timestep))

        prev_sample, _, _ = self._compute_step(model_output, sample, generator=generator)
        self._step_index += 1
        if return_dict:
            return FlowMatchSchedulerOutput(prev_sample=prev_sample, denoised=prev_sample)
        return (prev_sample,)

    def _check_sde_window_for_logprob(self):
        """Block log-prob computation when sde_window_size > 0.
        Mixed SDE/ODE mode has undefined density semantics for PPO/BPPO ratios.
        Use sde_window_size > 0 only for inference/evaluation, not RL training.
        """
        if self.sde_window_size > 0:
            raise RuntimeError(
                f"flow_sde_window_size={self.sde_window_size} is not compatible with "
                "log-prob based RL training (PPO/BPPO). Deterministic ODE suffix steps "
                "have no valid probability density. Set flow_sde_window_size=0 for RL, "
                "or use sde_window_size > 0 only for inference/evaluation."
            )

    def step_logprob(
        self,
        model_output: torch.FloatTensor,
        timestep,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        prev_sample=None,
        step_index=None,
        eta=1.0,
    ):
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step_logprob")
        self._check_sde_window_for_logprob()
        if step_index is not None:
            self._step_index = step_index
        else:
            self._init_step_index(self._scheduler_timestep(timestep))

        prev_sample_out, prev_sample_mean, std = self._compute_step(
            model_output, sample, prev_sample=prev_sample, generator=generator
        )
        log_prob = self._compute_logprob(prev_sample_out, prev_sample_mean, std, self.sde_type, self.cps_logprob_mode)
        self._step_index += 1
        if return_dict:
            return FlowMatchSchedulerOutput(prev_sample=prev_sample_out, denoised=prev_sample_out), log_prob
        return (prev_sample_out, prev_sample_out, log_prob)

    def step_forward_logprob(
        self,
        model_output: torch.FloatTensor,
        timestep,
        sample: torch.FloatTensor,
        next_sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        prev_sample=None,
        step_index=None,
        eta=1.0,
        return_debug: bool = False,
    ):
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step_forward_logprob")
        self._check_sde_window_for_logprob()
        if step_index is not None:
            self._step_index = step_index
        else:
            self._init_step_index(self._scheduler_timestep(timestep))

        _, prev_sample_mean, std = self._compute_step(model_output, sample, generator=generator)
        log_prob = self._compute_logprob(next_sample.float(), prev_sample_mean, std, self.sde_type, self.cps_logprob_mode)
        self._step_index += 1
        if return_debug:
            return log_prob, {'mean': prev_sample_mean, 'std': std}
        return log_prob

    def step_forward_logprob_with_entropy(
        self,
        model_output: torch.FloatTensor,
        timestep,
        sample: torch.FloatTensor,
        next_sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        prev_sample=None,
        step_index=None,
        eta=1.0,
    ):
        if self.num_inference_steps is None:
            raise ValueError("Must call set_timesteps before step_forward_logprob_with_entropy")
        self._check_sde_window_for_logprob()
        if step_index is not None:
            self._step_index = step_index
        else:
            self._init_step_index(self._scheduler_timestep(timestep))

        _, prev_sample_mean, std = self._compute_step(model_output, sample, generator=generator)
        log_prob = self._compute_logprob(next_sample.float(), prev_sample_mean, std, self.sde_type, self.cps_logprob_mode)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * std**2)
        entropy = entropy.expand_as(sample)
        self._step_index += 1
        return log_prob, entropy
