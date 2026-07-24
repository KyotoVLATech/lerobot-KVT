import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
import torch.nn.functional as F
from rl_100.common.pytorch_util import dict_apply
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = 'relu',
    final_activation: str = None
) -> torch.nn.modules.container.Sequential:


    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()

    layers = [nn.Linear(input_dim, hidden_dim), act_f]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_f)

    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers

    return nn.Sequential(*layers)



class ValueReluMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'relu', 'relu')
    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        state = state['agent_pos']
        return self._net(state)
    
class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'relu')
    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(s, dict):
            s = s['agent_pos']
        return self._net(s)

class ValueLearner_online:
    _device: torch.device
    _value: ValueReluMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
        self, 
        args,
        value_lr: float, 
        batch_size: int,
        normalizer,
        n_obs_steps, 
        use_pc_color,
        share_encoder=False,
        feature_dim=None
    ) -> None:
        super().__init__()
        self._device = args.device
        if share_encoder:
            args.state_dim = feature_dim
        self._value = ValueMLP(args).to(args.device)
        # import pdb; pdb.set_trace()
        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=value_lr,
            )
        self._batch_size = batch_size
        self.normalizer = normalizer
        self.n_obs_steps = n_obs_steps
        self.use_pc_color = use_pc_color
        self.share_encoder = share_encoder

    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, batch: dict
    ) -> float:

        nobs = self.normalizer.normalize(batch['obs'])
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).squeeze(1))

        nobs_features, Return = this_nobs, batch['return'][:, self.n_obs_steps - 1]
        value_loss = F.mse_loss(self._value(nobs_features), Return)

        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')
    