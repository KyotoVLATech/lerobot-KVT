#!/usr/bin/env python3
"""Memory-conscious, synchronous EXPO-FT for Iloha and a pretrained X-VLA.

Uses the ordinary EXPO-FT algorithm, not RTC. Frozen Florence features are
cached on CPU; only soft prompts (or optionally the action transformer),
a Gaussian edit actor, twin critics and entropy temperature are optimized.
The robot connects using iloha_rl.py --observation_format xvla.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint

LOG = logging.getLogger(__name__)
ACTION_DIM = 14


def pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.frombuffer(obj[b"data"], dtype=obj[b"dtype"]).reshape(obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def stack_cache(items, device):
    return {key: torch.cat([item[key] for item in items]).to(device) for key in items[0]}


class XVLAAdapter:
    """Keep the saved evaluation processors; bypass only the frozen VLM on replay."""

    def __init__(self, args):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.processor import NormalizerProcessorStep

        self.device = torch.device(args.device)
        cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        if cfg.type != "xvla" or cfg.action_feature.shape != (ACTION_DIM,):
            raise ValueError("Requires an Iloha XVLA checkpoint with 14D absolute ALOHA actions")
        if cfg.action_mode not in ("auto", "joint"):
            raise ValueError("Only auto/joint action spaces are supported; ee6d is not an Iloha joint policy")
        cfg.device = args.device
        cfg.dtype = "bfloat16" if self.device.type == "cuda" else "float32"
        self.policy = XVLAPolicy.from_pretrained(args.policy_path, config=cfg)
        self.model = self.policy.model
        self.horizon = args.replan_steps
        if not 1 <= self.horizon <= cfg.chunk_size:
            raise ValueError("replan_steps must be between 1 and checkpoint chunk_size")
        # This is the same saved processor route used by iloha_eval.py.
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=args.policy_path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        self.normalizers = [s for s in self.pre.steps if isinstance(s, NormalizerProcessorStep)]
        self.policy.requires_grad_(False)
        for name, param in self.model.transformer.named_parameters():
            if args.train_scope == "transformer" or "soft_prompt_hub" in name:
                param.requires_grad_(True)
                # FP32 optimizer/master parameters avoid BF16 tiny-update rounding.
                param.data = param.data.float()
        self.trainable = [p for p in self.policy.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError("Checkpoint has no trainable soft prompts")
        self.optimizer = torch.optim.AdamW(self.trainable, lr=args.vla_lr)
        self.reward_weighted_imitation = not args.imitate_failures
        if not args.no_gradient_checkpointing:
            # Wrap forward only, preserving state_dict names for normal evaluation.
            for block in self.model.transformer.blocks:
                original = block.forward

                def forward(x, original=original):
                    if torch.is_grad_enabled():
                        return checkpoint(original, x, use_reentrant=False)
                    return original(x)

                block.forward = forward
        self.policy.eval()
        LOG.info("X-VLA: %s trainable parameters / %s total", sum(p.numel() for p in self.trainable),
                 sum(p.numel() for p in self.policy.parameters()))

    def amp(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    @torch.no_grad()
    def encode(self, observation):
        if "observation.state" not in observation:
            raise ValueError("Start iloha_rl.py with --observation_format xvla")
        raw = {}
        for key, value in observation.items():
            if key.startswith("observation."):
                tensor = torch.from_numpy(np.array(value, copy=True))
                if key.startswith("observation.images."):
                    if tensor.ndim != 3 or tensor.shape[-1] != 3:
                        raise ValueError(f"Expected HWC RGB image for {key}")
                    tensor = tensor.float().div(255).permute(2, 0, 1).contiguous()
                raw[key] = tensor.unsqueeze(0).to(self.device)
        raw["task"] = observation.get("task", "")
        raw["robot_type"] = "iloha"
        batch = self.pre(raw)
        missing = set(self.policy.config.image_features) - set(batch)
        if missing:
            raise ValueError(f"Missing checkpoint cameras after saved rename map: {sorted(missing)}")
        inputs = self.policy._build_model_inputs(batch)
        with self.amp():
            enc = self.model.forward_vlm(inputs["input_ids"],
                                         inputs["image_input"].to(self.model._get_target_dtype()),
                                         inputs["image_mask"])
        cache = {**enc, "proprio": inputs["proprio"], "domain_id": inputs["domain_id"]}
        # Separate pooling preserves main and wrist information for the small RL networks.
        aux = enc["aux_visual_inputs"]
        aux_mean = aux.mean(1) if aux.shape[1] else torch.zeros_like(enc["vlm_features"].mean(1))
        cache["rl_features"] = torch.cat([enc["vlm_features"].mean(1), aux_mean,
                                           inputs["proprio"]], dim=-1).float()
        return {k: v.detach().cpu() for k, v in cache.items()}

    @torch.no_grad()
    def sample(self, caches):
        c = stack_cache(caches, self.device)
        model = self.model
        dtype = model._get_target_dtype()
        noise = torch.randn(len(caches), model.chunk_size, model.dim_action, device=self.device, dtype=dtype)
        action = torch.zeros_like(noise)
        self.policy.eval()
        steps = max(1, self.policy.config.num_denoising_steps)
        with self.amp():
            for i in range(steps, 0, -1):
                t = torch.full((len(caches),), i / steps, device=self.device, dtype=dtype)
                noisy = noise * t[:, None, None] + action * (1 - t[:, None, None])
                proprio, noisy = model.action_space.preprocess(c["proprio"].to(dtype), noisy)
                action = model.transformer(domain_id=c["domain_id"], action_with_noise=noisy, t=t,
                                           proprio=proprio, vlm_features=c["vlm_features"],
                                           aux_visual_inputs=c["aux_visual_inputs"])
            action = model.action_space.postprocess(action).float()[..., :ACTION_DIM]
        return self.post(action).to(self.device).float()[:, :self.horizon]

    def imitate(self, transitions):
        """Microbatch one, accumulate gradients; never supervise unexecuted padding."""
        from lerobot.types import TransitionKey

        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        weights = [t.episode_reward if self.reward_weighted_imitation and t.episode_reward is not None
                   else 1.0 for t in transitions]
        weight_sum = sum(weights)
        for transition, weight in zip(transitions, weights, strict=True):
            c = stack_cache([transition.observation], self.device)
            action = transition.action.unsqueeze(0).to(self.device)
            normalized = {TransitionKey.ACTION: action}
            for step in self.normalizers:
                normalized = step(normalized)
            targets = self.policy._prepare_action_targets({"action": normalized[TransitionKey.ACTION]})
            dtype = self.model._get_target_dtype()
            targets = targets.to(dtype)
            t = torch.rand(1, device=self.device, dtype=dtype) * (1 - 1e-5)
            noisy = torch.randn_like(targets) * t[:, None, None] + targets * (1 - t[:, None, None])
            proprio, noisy = self.model.action_space.preprocess(c["proprio"].to(dtype), noisy)
            with self.amp():
                pred = self.model.transformer(domain_id=c["domain_id"], action_with_noise=noisy, t=t,
                                              proprio=proprio, vlm_features=c["vlm_features"],
                                              aux_visual_inputs=c["aux_visual_inputs"])
                valid = transition.steps
                losses = self.model.action_space.compute_loss(pred[:, :valid].float(), targets[:, :valid].float())
                loss = sum(losses.values()) * weight / max(weight_sum, 1e-8)
            loss.backward()
            loss_total += loss.detach().item()
        nn.utils.clip_grad_norm_(self.trainable, 1.0)
        self.optimizer.step()
        return loss_total

    def export(self, path):
        self.policy.save_pretrained(path)
        self.pre.save_pretrained(path)
        self.post.save_pretrained(path)


def mlp(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class EditActor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(feature_dim + action_dim, action_dim * 2, hidden)

    def sample(self, features, base, scale):
        mean, log_std = self.net(torch.cat([features, base], -1)).chunk(2, -1)
        dist = Normal(mean, log_std.clamp(-5, 1).exp())
        z = dist.rsample()
        edit = z.tanh()
        log_prob = (dist.log_prob(z) - torch.log(1 - edit.square() + 1e-6) - scale.log()).sum(-1)
        return base + edit * scale, log_prob


class TwinCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden):
        super().__init__()
        self.q1 = mlp(feature_dim + action_dim, 1, hidden)
        self.q2 = mlp(feature_dim + action_dim, 1, hidden)

    def forward(self, features, action):
        x = torch.cat([features, action], -1)
        return torch.stack([self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)])


@dataclass
class Transition:
    observation: dict
    action: torch.Tensor
    next_observation: dict
    reward: float
    bootstrap: float
    steps: int
    success: bool = False
    episode_reward: float | None = None


def make_transition(observation, executed, next_observation, rewards, done, horizon, gamma):
    """Chunk return with per-step discount and actual post-safety hardware actions."""
    if len(executed) == 0 or len(executed) != len(rewards) or len(executed) > horizon:
        raise ValueError("Invalid executed chunk/reward lengths")
    action = np.asarray(executed, dtype=np.float32)
    if action.shape != (len(executed), ACTION_DIM) or not np.isfinite(action).all():
        raise ValueError("Hardware feedback must contain finite 14D actions")
    if len(action) < horizon:
        action = np.concatenate([action, np.repeat(action[-1:], horizon - len(action), axis=0)])
    return Transition(observation, torch.from_numpy(action.copy()), next_observation,
                      sum(gamma**i * r for i, r in enumerate(rewards)),
                      0.0 if done else gamma**len(executed), len(executed))


class EXPOLearner:
    """Ordinary EXPO: Q-ranked base/edit proposals, off-policy RL, VLA imitation."""

    def __init__(self, adapter, feature_dim, args):
        self.adapter, self.args = adapter, args
        self.device = adapter.device
        self.action_dim = adapter.horizon * ACTION_DIM
        self.edit = EditActor(feature_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic = TwinCritic(feature_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.edit_opt = torch.optim.Adam(self.edit.parameters(), lr=args.rl_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.rl_lr)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(args.initial_alpha), device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.rl_lr)
        # Physical ALOHA units; joint radians and normalized gripper openness.
        scale = [args.edit_scale] * ACTION_DIM
        scale[6] = scale[13] = args.gripper_edit_scale
        self.scale = torch.tensor(scale * adapter.horizon, device=self.device)
        self.target_entropy = -self.action_dim / 2 + self.scale.log().sum().item()
        self.replay = deque(maxlen=args.replay_capacity)
        self.updates = 0
        self.episodes = 0

    def constrain(self, actions):
        chunk = actions.reshape(-1, self.adapter.horizon, ACTION_DIM)
        # Use an out-of-place mask so actor gradients remain well-defined.
        mask = torch.zeros_like(chunk, dtype=torch.bool)
        mask[..., [6, 13]] = True
        return torch.where(mask, chunk.clamp(0, 1), chunk).flatten(1)

    @torch.no_grad()
    def choose(self, caches, edits=True):
        features = stack_cache(caches, self.device)["rl_features"].float()
        # VLA candidates are generated sequentially to bound peak VRAM.
        bases = [self.adapter.sample(caches).flatten(1) for _ in range(self.args.base_candidates)]
        candidates = list(bases)
        if edits:
            for i in range(self.args.edit_candidates):
                edited, _ = self.edit.sample(features, bases[i % len(bases)], self.scale)
                candidates.append(self.constrain(edited))
        proposals = torch.stack(candidates, 1)
        count = proposals.shape[1]
        qs = self.target(features.repeat_interleave(count, 0), proposals.flatten(0, 1)).amin(0)
        best = qs.reshape(len(caches), count).argmax(-1)
        return proposals[torch.arange(len(caches), device=self.device), best].reshape(
            len(caches), self.adapter.horizon, ACTION_DIM)

    def update(self):
        batch = random.sample(list(self.replay), self.args.batch_size)
        features = stack_cache([t.observation for t in batch], self.device)["rl_features"].float()
        actions = torch.stack([t.action for t in batch]).to(self.device).flatten(1)
        rewards = torch.tensor([t.reward for t in batch], device=self.device)
        bootstrap = torch.tensor([t.bootstrap for t in batch], device=self.device)
        with torch.no_grad():
            target_q = rewards.clone()
            live = [i for i, t in enumerate(batch) if t.bootstrap > 0]
            if live:
                next_caches = [batch[i].next_observation for i in live]
                next_actions = self.choose(next_caches).flatten(1)
                next_features = stack_cache(next_caches, self.device)["rl_features"].float()
                target_q[live] += bootstrap[live] * self.target(next_features, next_actions).amin(0)
        critic_loss = (self.critic(features, actions) - target_q).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Match libs/expo-ft: edit actor conditions on replay actions and optimizes mean Q.
        self.critic.requires_grad_(False)
        edited, log_prob = self.edit.sample(features, actions, self.scale)
        edited = self.constrain(edited)
        edit_loss = (self.log_alpha.exp().detach() * log_prob - self.critic(features, edited).mean(0)).mean()
        self.edit_opt.zero_grad(set_to_none=True)
        edit_loss.backward()
        nn.utils.clip_grad_norm_(self.edit.parameters(), 1.0)
        self.edit_opt.step()
        self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-12, 2)
            for target, source in zip(self.target.parameters(), self.critic.parameters(), strict=True):
                target.lerp_(source, self.args.tau)
        self.updates += 1
        metrics = {"critic_loss": critic_loss.item(), "edit_loss": edit_loss.item(),
                   "alpha": self.log_alpha.exp().item(), "updates": self.updates}
        if self.updates % self.args.vla_update_every == 0:
            pool = [t for t in self.replay if t.success or (t.episode_reward or 0) > 0
                    or self.args.imitate_failures]
            if pool:
                metrics["vla_loss"] = self.adapter.imitate(random.sample(pool, min(len(pool), self.args.vla_batch_size)))
        return metrics

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.adapter.export(directory / "pretrained_model")
        # Cached observations are bounded CPU tensors and contain no raw videos.
        state = {"edit": self.edit.state_dict(), "critic": self.critic.state_dict(),
                 "target": self.target.state_dict(), "log_alpha": self.log_alpha.detach(),
                 "edit_opt": self.edit_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                 "alpha_opt": self.alpha_opt.state_dict(), "vla_opt": self.adapter.optimizer.state_dict(),
                 "vla_trainable": {n: p.detach().cpu() for n, p in self.adapter.policy.named_parameters()
                                   if p.requires_grad},
                 "updates": self.updates, "episodes": self.episodes, "args": vars(self.args),
                 "replay": [{"observation": t.observation, "action": t.action,
                             "next_observation": t.next_observation, "reward": t.reward,
                             "bootstrap": t.bootstrap, "steps": t.steps, "success": t.success,
                             "episode_reward": t.episode_reward}
                            for t in self.replay], "random_state": random.getstate(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}
        torch.save(state, directory / "learner.pt")
        (directory / "run_config.json").write_text(json.dumps(vars(self.args), indent=2), encoding="utf-8")

    def restore(self, directory):
        state = torch.load(directory / "learner.pt", map_location="cpu", weights_only=True)
        for key in ("replan_steps", "train_scope", "hidden_dim", "edit_scale", "gripper_edit_scale"):
            if state["args"][key] != getattr(self.args, key):
                raise ValueError(f"Resume config mismatch: {key}")
        for name in ("edit", "critic", "target", "edit_opt", "critic_opt", "alpha_opt"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
            for name, param in self.adapter.policy.named_parameters():
                if param.requires_grad:
                    param.copy_(state["vla_trainable"][name])
        self.adapter.optimizer.load_state_dict(state["vla_opt"])
        # map_location must not leave cached replay tensors resident on the GPU.
        for item in state["replay"]:
            t = Transition(**item)
            t.observation = {k: v.cpu() for k, v in t.observation.items()}
            t.next_observation = {k: v.cpu() for k, v in t.next_observation.items()}
            t.action = t.action.cpu()
            self.replay.append(t)
        self.updates, self.episodes = state["updates"], state["episodes"]
        random.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


async def rpc(websocket, operation, **kwargs):
    await websocket.send(msgpack.packb({"operation": operation, **kwargs}, default=pack_array))
    reply = msgpack.unpackb(await websocket.recv(), object_hook=unpack_array)
    if reply.get("status") != "success":
        raise RuntimeError(f"Robot operation {operation}: {reply.get('message', reply)}")
    return reply


async def compute(function, *args):
    """Finish GPU mutations before cancellation starts checkpoint serialization."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def collect_episode(websocket, adapter, learner, args):
    response = await rpc(websocket, "reset")
    cache = await compute(adapter.encode, response["observation"])
    episode = []
    done, success = False, False
    rating = 0.0
    while not done:
        warm = len(learner.replay) >= args.warmup_chunks
        proposal = await compute(learner.choose, [cache], warm)
        chunk = proposal[0].detach().cpu().numpy()
        executed, rewards = [], []
        for action in chunk:
            # Check before inference-delayed actions and again after each control step.
            info = await rpc(websocket, "get_info_for_step")
            if info["done"]:
                done, success = True, bool(info["success"])
                rating = float(info["reward"])
                if rewards:
                    rewards[-1] = float(info["reward"])
                elif episode:
                    last = episode[-1]
                    last.reward += args.discount ** (last.steps - 1) * float(info["reward"])
                    last.bootstrap = 0.0
                break
            start = time.monotonic()
            reply = await rpc(websocket, "step", action=action)
            executed.append(reply["action"])
            await asyncio.sleep(max(0, 1 / args.control_hz - (time.monotonic() - start)))
            info = await rpc(websocket, "get_info_for_step")
            rewards.append(float(info["reward"]))
            done, success = bool(info["done"]), bool(info["success"])
            if done:
                rating = float(info["reward"])
                break
        if executed:
            next_cache = cache
            if not done:
                response = await rpc(websocket, "get_observation")
                next_cache = await compute(adapter.encode, response["observation"])
                # The observation RPC also polls the verdict; keep a terminal event.
                if response["done"]:
                    done, success = True, bool(response["success"])
                    rating = float(response["reward"])
                    rewards[-1] = float(response["reward"])
            episode.append(make_transition(cache, executed, next_cache, rewards, done,
                                           adapter.horizon, args.discount))
            cache = next_cache
    for transition in episode:
        transition.success = success
        transition.episode_reward = rating
    learner.replay.extend(episode)
    learner.episodes += 1
    return {"episode": learner.episodes, "reward": rating, "chunks": len(episode),
            "steps": sum(t.steps for t in episode), "replay": len(learner.replay)}


async def serve(args):
    from websockets.asyncio.server import serve as websocket_serve

    adapter = XVLAAdapter(args)
    learner = None
    connected = False
    completed = asyncio.Event()

    async def handler(websocket):
        nonlocal learner, connected
        if connected:
            await websocket.close(code=1013, reason="A robot is already connected")
            return
        connected = True
        try:
            await rpc(websocket, "create_env")
            observation = await rpc(websocket, "get_observation")
            cache = await compute(adapter.encode, observation["observation"])
            if learner is None:
                learner = EXPOLearner(adapter, cache["rl_features"].shape[-1], args)
                if args.resume:
                    learner.restore(Path(args.resume))
            while args.episodes == 0 or learner.episodes < args.episodes:
                metrics = await collect_episode(websocket, adapter, learner, args)
                # Standby before any learning/export pause; robot side retains reset gating.
                await rpc(websocket, "standby")
                LOG.info("Rollout %s", json.dumps(metrics))
                if len(learner.replay) >= max(args.warmup_chunks, args.batch_size):
                    for _ in range(args.updates_per_episode):
                        metrics = await compute(learner.update)
                    LOG.info("Train %s", json.dumps(metrics))
                if learner.episodes % args.save_every == 0:
                    await compute(learner.save, Path(args.output_dir) / f"episode_{learner.episodes:06d}")
            await websocket.close(code=1000, reason="Training complete")
        except Exception:
            LOG.exception("X-VLA session stopped; incomplete episode is excluded from replay")
            # The client also stops motors on a broken connection.
            try:
                await rpc(websocket, "standby")
            except Exception:
                LOG.warning("Unable to request standby after session error")
        finally:
            try:
                if learner is not None:
                    await compute(learner.save, Path(args.output_dir) / "latest")
            finally:
                connected = False
                if learner is not None and args.episodes > 0 and learner.episodes >= args.episodes:
                    completed.set()

    LOG.info("Normal EXPO-FT listening on ws://%s:%d", args.bind, args.port)
    async with websocket_serve(handler, args.bind, args.port, compression=None, max_size=None,
                               ping_interval=None, ping_timeout=None):
        await completed.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True, help="Local trained XVLA pretrained_model directory")
    parser.add_argument("--output_dir", default="outputs/rl/xvla_expo_ft")
    parser.add_argument("--resume", help="Trusted local episode checkpoint directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--replan_steps", type=int, default=8)
    parser.add_argument("--control_hz", type=float, default=30)
    parser.add_argument("--episodes", type=int, default=0, help="0: unlimited; total including resumed episodes")
    parser.add_argument("--train_scope", choices=["soft-prompts", "transformer"], default="soft-prompts")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--vla_lr", type=float, default=1e-5)
    parser.add_argument("--rl_lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vla_batch_size", type=int, default=1)
    parser.add_argument("--vla_update_every", type=int, default=4)
    parser.add_argument("--updates_per_episode", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--replay_capacity", type=int, default=256, help="CPU cached feature chunks")
    parser.add_argument("--warmup_chunks", type=int, default=32)
    parser.add_argument("--base_candidates", type=int, default=2)
    parser.add_argument("--edit_candidates", type=int, default=2)
    parser.add_argument("--edit_scale", type=float, default=0.03, help="Joint edit bound in ALOHA radians")
    parser.add_argument("--gripper_edit_scale", type=float, default=0.03)
    parser.add_argument("--initial_alpha", type=float, default=0.001)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--imitate_failures", action="store_true",
                        help="Imitate zero-rated episodes too; default: reward-weighted positive-rated episodes")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    positive = ("replan_steps", "control_hz", "vla_lr", "rl_lr", "batch_size", "vla_batch_size",
                "vla_update_every", "updates_per_episode", "hidden_dim", "replay_capacity",
                "base_candidates", "edit_candidates", "edit_scale", "gripper_edit_scale", "initial_alpha", "save_every")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        parser.error("Learning rates, dimensions, counts, frequencies and edit scales must be finite and positive")
    if args.episodes < 0 or args.warmup_chunks < 0 or not 0 < args.discount <= 1 or not 0 < args.tau <= 1:
        parser.error("Invalid episode/warmup count, discount or tau")
    if args.replay_capacity < max(args.batch_size, args.warmup_chunks):
        parser.error("Replay capacity must cover batch_size and warmup_chunks")
    if args.resume:
        args.policy_path = str(Path(args.resume) / "pretrained_model")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_args()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for debugging")
    try:
        asyncio.run(serve(arguments))
    except KeyboardInterrupt:
        LOG.info("Stopped")
