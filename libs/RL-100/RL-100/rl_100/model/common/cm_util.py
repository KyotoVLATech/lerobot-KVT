
import torch
import numpy as np
@torch.no_grad()
def update_ema(target_params, source_params, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        # pred_original_sample = (
    #       sample - beta_prod_t ** (0.5) * model_output
    #   ) / alpha_prod_t ** (0.5)

        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev