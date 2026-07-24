import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from rl_100.unidpg.transition_model.dynamics.base_dynamics import BaseDynamics
from rl_100.unidpg.transition_model.utils.scaler import StandardScaler
from rl_100.unidpg.transition_model.utils.logger import Logger

from rl_100.common.pytorch_util import dict_apply
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        # scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        env,
        normalizer,
        encoder,
        cfg,
        gamma: float = 0.99,
        lamda: float = 0.95,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        # self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.env = env
        self.normalizer = normalizer
        self.encoder = encoder
        self.cfg = cfg
        self.n_obs_steps = cfg.n_obs_steps
        self.use_pc_color = cfg.policy.use_pc_color
        self.predict_delta = cfg.dynamics.predict_delta
        self.lamda = lamda
        self.gamma = gamma
    @ torch.no_grad()
    def step(
        self,
        nobs_features: torch.tensor,
        action: torch.tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        # obs_act = np.concatenate([obs, action], axis=-1)
        # obs_act = self.scaler.transform(obs_act)
        # nobs_features = self.obs2latent(obs)
        if self.cfg.use_action_embed:
            action = self.model._action_encoder(action)
        nobs_features, action = nobs_features.cpu().data.numpy(), action.cpu().data.numpy()
        obs_act = np.concatenate([nobs_features, action], axis=-1)
        # obs_act = torch.cat((nobs_features, action), dim = -1) # B, n_obs*feature_dim; B, action_dim

        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += nobs_features
        std = np.sqrt(np.exp(logvar))
        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]

        terminal = self.terminal_fn(nobs_features, action, next_obs, self.env)
        info = {}
        info["raw_reward"] = reward
        self.cnt = 0
        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    @ torch.no_grad()
    def multi_step(
        self,
        nobs_features: torch.tensor,
        nactions: torch.tensor,
        reward_strategy: str = "sum",
        discount: float = 1.0,
        Return: float = 0.0,
        Qs: List[torch.tensor] = None,
        Q: Callable[[torch.tensor, torch.tensor], torch.tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        n_step_actions = nactions.shape[1]
        all_obs_features, rewards, terminals, infos = [], [], [], []
        rewards = 0
        batch_size = nactions.shape[0]
        nobs_features = nobs_features.reshape(batch_size, -1)   
        for i in range(n_step_actions):
            if Q != None:
                Qs.append(Q(nobs_features, nactions[:, i]))
            next_obs, reward, terminal, info = self.step(nobs_features, nactions[:, i])
            if reward_strategy == "sum":
                rewards += reward
            Return += discount * reward
            discount *= self.gamma
            nobs_features = torch.from_numpy(next_obs).to(self.model.device)
        return next_obs, rewards, terminal, info, Return, Qs, discount

    @ torch.no_grad()
    def multi_step_evaluation(
        self,
        nobs_features: torch.tensor,
        nactions: torch.tensor,
        Q: Callable[[torch.tensor, torch.tensor], torch.tensor],
        state_dict: Dict = None,
        use_gae: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        n_step_actions = nactions.shape[1]
        all_obs_features, rewards, terminals, infos, gae_advantages, Qs = [], [], [], [], [], []
        G, discount = 0, 1
        if self.cfg.online:
            if self.cfg.ppo.iql_ft:
                iql_input = state_dict
            else:
                iql_input = nobs_features
        else:
            iql_input = nobs_features
        if n_step_actions > 1:
            raise NotImplementedError # just for 

        for i in range(n_step_actions):
            
            Qs.append(Q(iql_input, nactions[:, i]))
            all_obs_features.append(nobs_features)
            next_obs, reward, terminal, info = self.step(nobs_features, nactions[:, i])
            G += discount * reward
            discount *= self.gamma
            rewards.append(reward); terminals.append(terminal); infos.append(info)
            nobs_features = torch.from_numpy(next_obs).to(self.model.device)
        if use_gae:
            deltas, gae = Qs, 0
            for delta in reversed(deltas):
                gae = delta + self.gamma * self.lamda * gae
                gae_advantages.insert(0, gae)
        G += discount * Qs[-1].cpu().numpy()
        if use_gae:
            return all_obs_features, rewards, terminals, infos, G.squeeze(), torch.stack(gae_advantages).squeeze(1)
        else:
            return all_obs_features, rewards, terminals, infos, torch.from_numpy(G).squeeze(), None

    def obs2latent(self, nobs):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['agent_pos'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.encoder(this_nobs).reshape(batch_size, -1)
        return nobs_features

    
    @ torch.no_grad()
    def compute_model_uncertainty(self, obs: torch.tensor, action: np.ndarray, uncertainty_mode="aleatoric") -> np.ndarray:
        # obs_act = np.concatenate([obs, action], axis=-1)
        # obs_act = self.scaler.transform(obs_act)
        if self.cfg.dynamics.fix_encoder:
            nobs_features = obs
        else:
            nobs_features = self.obs2latent(obs)
        if self.cfg.use_action_embed:
            action = self.model._action_encoder(action)
        obs_act = torch.cat((nobs_features, action), dim = -1) # B, n_obs*feature_dim; B, action_dim


        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += nobs_features
        std = np.sqrt(np.exp(logvar))

        if uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        elif uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean[:, :, :-1]
            next_obs_mean = np.mean(next_obses_mean, axis=0)
            diff = next_obses_mean - next_obs_mean
            penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        else:
            raise ValueError
        
        penalty = np.expand_dims(penalty, 1).astype(np.float32)

        return self._penalty_coef * penalty
    
    @ torch.no_grad()
    def predict_next_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        # get latent obs from encoder
        if self.cfg.dynamics.fix_encoder:
            nobs_features = obs
        else:
            nobs_features = self.obs2latent(obs)
        if self.cfg.use_action_embed:
            action = self.model._action_encoder(action)
        obs_act = torch.cat((nobs_features, action), dim = -1) # B, n_obs*feature_dim; B, action_dim

        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:

        nobs_features = self.obs2latent(data['obs']).detach()
        next_nobs_features = self.obs2latent(data['next_obs']).detach()

        if self.predict_delta:
            targets = next_nobs_features - nobs_features
        else:
            targets = next_nobs_features
        if self.cfg.action_norm:
            actions = self.normalizer['action'].normalize(data['action']) # TODO: Is needing to normalize actions?
        else:
            actions = data['action']
        rewards = data["reward"][:, self.n_obs_steps - 1]
        action = actions[:, self.n_obs_steps - 1]
        if self.cfg.use_action_embed:
            action = self.model._action_encoder(action)
        inputs = torch.cat((nobs_features, action), dim = -1) # B, n_obs*feature_dim; B, action_dim
        targets = torch.cat((targets, rewards), dim=-1) #B, n_obs*feature_dim; B, 1
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 15,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)

        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 100)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        # normalize data;
        # self.scaler.fit(train_inputs)
        # train_inputs = self.scaler.transform(train_inputs)
        # holdout_inputs = self.scaler.transform(holdout_inputs)
        # -----------------------------normalized before
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = torch.from_numpy(np.random.randint(train_size, size=[self.model.num_ensemble, train_size]))
        
        # def shuffle_rows(arr):
        #     idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        #     return arr[np.arange(arr.shape[0])[:, None], idxes]
        def shuffle_rows(arr):
            shape = arr.size()
            idxes = torch.argsort(torch.rand(shape), dim=-1)
            row_indices = torch.arange(shape[0]).unsqueeze(1)
            return arr[row_indices, idxes]
        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            print('epoch {}'.format(epoch))
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            # import pdb
            # pdb.set_trace()
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    @ torch.no_grad()
    def rollout(self,   
        policy: nn.Module,
        Q: nn.Module,
        iql: nn.Module,
        batch: dict,
        rollout_length: int,
        is_iql: bool = True,
        use_gae: bool = False,
        first_action: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if is_iql:
            q_eval = iql.minQ
        else:
            q_eval = Q
        num_transitions = 0
        rewards_arr = np.array([])
        total_q = np.array([])
        rollout_transitions = defaultdict(list)

        # nobs_features = self.obs2latent(init_obss)
        # rollout
        # observations = init_obss
        length = 0
        Return, discount, Qs = 0, 1, []
        # get latent obs from encoder
        batch_size = batch['obs']['agent_pos'].shape[0]
        if self.cfg.dynamics.fix_encoder:
            nobs_features = policy.obs2latent(batch['obs']) 
        else:
            nobs_features = self.obs2latent(batch['obs'])

        for _ in range(int(rollout_length/self.cfg.n_action_steps)):

            actions = policy.sample_action(nobs_features, get_np = False, batch_size=batch_size)    
            if not self.cfg.no_pre_action:
                actions = actions[:, self.cfg.n_obs_steps - 1:]
            next_nobs_features, rewards, terminal, info, Return, Qs, discount = self.multi_step(nobs_features, actions[:, :self.cfg.n_action_steps], \
                                                                                        discount=discount, Return=Return, Qs=Qs, Q=q_eval)
            rewards_arr = np.append(rewards_arr, rewards.flatten())
            nobs_features = torch.from_numpy(next_nobs_features).to(self.model.device)
        if use_gae:
            deltas, gae = Qs, 0
            gae_advantages = []
            for delta in reversed(deltas):
                gae = delta + self.gamma * self.lamda * gae
                gae_advantages.insert(0, gae)
            gae_advantages = torch.stack(gae_advantages) # batch_size, rollout_length
            if first_action:
                q_evaluation = torch.mean(gae_advantages[:, 0])
            else:
                q_evaluation = torch.mean(gae_advantages)
        else:
            q_evaluation = torch.mean(torch.stack(Qs))
    
        return q_evaluation, rewards_arr.mean()
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        # Handle DDP model saving
        model_state_dict = self.model.state_dict()
        new_model_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('module.'):
                new_model_state_dict[k[7:]] = v
            else:
                new_model_state_dict[k] = v
        torch.save(new_model_state_dict, os.path.join(save_path, "dynamics.pth"))
        
        if not self.cfg.dynamics.fix_encoder:
            # Handle DDP encoder saving
            enc_state_dict = self.encoder.state_dict()
            new_enc_state_dict = {}
            for k, v in enc_state_dict.items():
                if k.startswith('module.'):
                    new_enc_state_dict[k[7:]] = v
                else:
                    new_enc_state_dict[k] = v
            torch.save(new_enc_state_dict, os.path.join(save_path, "dynamics_encoder.pth"))
        # self.scaler.save_scaler(save_path)
        print('dynamics model saved in {}'.format(str(save_path)))

    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        if not self.cfg.dynamics.fix_encoder:
            self.encoder.load_state_dict(torch.load(os.path.join(load_path, "dynamics_encoder.pth"), map_location=self.model.device))
        # self.scaler.load_scaler(load_path)
        print('dynamics model loaded from {}'.format(str(load_path)))
