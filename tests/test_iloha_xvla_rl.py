"""Hardware-free regressions for chunk credit, actual actions and EXPO updates."""

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iloha_xvla_rl", ROOT / "iloha_xvla_rl.py")
rl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rl
spec.loader.exec_module(rl)


def cache():
    return {"rl_features": torch.ones(1, 4)}


class FakeAdapter:
    device = torch.device("cpu")
    horizon = 2

    def __init__(self):
        self.policy = nn.Linear(4, 4)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.imitated = []

    def encode(self, observation):
        return cache()

    def sample(self, caches):
        return torch.full((len(caches), self.horizon, 14), 0.5)

    def imitate(self, transitions):
        self.imitated.extend(transitions)
        return 0.0

    def export(self, path):
        path.mkdir(parents=True, exist_ok=True)


def learner():
    args = rl.parse_args(["--policy_path", "unused", "--device", "cpu", "--batch_size", "2",
                          "--warmup_chunks", "2", "--hidden_dim", "16", "--vla_update_every", "1",
                          "--replan_steps", "2"])
    return rl.EXPOLearner(FakeAdapter(), 4, args)


def test_discount_partial_terminal_and_hardware_feedback():
    executed = [np.full(14, 0.4), np.full(14, 0.6)]
    transition = rl.make_transition(cache(), executed, cache(), [0, 1], True, 4, 0.9)
    assert transition.reward == pytest.approx(0.9)
    assert transition.bootstrap == 0
    assert transition.steps == 2
    np.testing.assert_allclose(transition.action[:, 0], [0.4, 0.6, 0.6, 0.6])
    live = rl.make_transition(cache(), executed, cache(), [0, 0], False, 2, 0.9)
    assert live.bootstrap == pytest.approx(0.9**2)
    with pytest.raises(ValueError):
        rl.make_transition(cache(), [np.zeros(13)], cache(), [0], True, 2, 0.9)


def test_numpy_wire_matches_client():
    value = {"observation": {"state": np.arange(14, dtype=np.float32),
                             "image": np.zeros((480, 640, 3), dtype=np.uint8)}}
    result = msgpack.unpackb(msgpack.packb(value, default=rl.pack_array), object_hook=rl.unpack_array)
    np.testing.assert_array_equal(result["observation"]["state"], value["observation"]["state"])
    assert result["observation"]["image"].shape == (480, 640, 3)


def test_updates_change_actor_critic_target_and_imitate_only_success(tmp_path):
    agent = learner()
    for success in (True, False):
        t = rl.make_transition(cache(), [np.full(14, 0.5)] * 2, cache(), [0, int(success)], True, 2, 0.99)
        t.success = success
        agent.replay.append(t)
    before_q = copy.deepcopy(agent.critic.state_dict())
    before_edit = copy.deepcopy(agent.edit.state_dict())
    metrics = agent.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(before_q[k], v) for k, v in agent.critic.state_dict().items())
    assert any(not torch.equal(before_edit[k], v) for k, v in agent.edit.state_dict().items())
    assert all(t.success for t in agent.adapter.imitated)
    assert not any(p.grad is not None for p in agent.target.parameters())
    agent.save(tmp_path)
    restored = learner()
    restored.restore(tmp_path)
    assert restored.updates == 1
    assert len(restored.replay) == 2
    for key, value in agent.edit.state_dict().items():
        torch.testing.assert_close(value, restored.edit.state_dict()[key])
    for a, b in zip(agent.adapter.policy.parameters(), restored.adapter.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert all(t.action.device.type == "cpu" for t in restored.replay)
    assert np.isfinite(restored.update()["critic_loss"])


def test_q_ranking_selects_edit_and_grippers_are_bounded():
    agent = learner()
    agent.adapter.horizon = 2

    class PositiveEdit(nn.Module):
        def sample(self, features, base, scale):
            return base + 2, torch.zeros(len(base))

    class SumCritic(nn.Module):
        def forward(self, features, action):
            return action.sum(-1).repeat(2, 1)

    agent.edit = PositiveEdit()
    agent.target = SumCritic()
    selected = agent.choose([cache()])
    assert selected.shape == (1, 2, 14)
    assert selected[0, 0, 0] == 2.5
    assert selected[0, 0, 6] == 1
    assert selected[0, 0, 13] == 1
    assert torch.all(agent.choose([cache()], edits=False) == 0.5)


def test_nonterminal_backup_uses_latest_base_policy():
    agent = learner()
    calls = []
    sample = agent.adapter.sample

    def recording(caches):
        calls.append(len(caches))
        return sample(caches)

    agent.adapter.sample = recording
    for _ in range(2):
        agent.replay.append(rl.make_transition(cache(), [np.full(14, 0.5)] * 2,
                                                cache(), [0, 0], False, 2, 0.99))
    agent.update()
    assert calls == [2] * agent.args.base_candidates


def test_rollout_keeps_partial_success_and_actual_actions(monkeypatch):
    agent = learner()
    replies = iter([
        {"observation": {}},  # reset
        {"done": False, "success": False, "reward": 0},  # before action
        {"action": [0.3] * 14},  # executed action differs from the 0.5 proposal
        {"done": True, "success": False, "reward": 0.7},
    ])
    operations = []

    async def fake_rpc(websocket, operation, **kwargs):
        operations.append(operation)
        return next(replies)

    monkeypatch.setattr(rl, "rpc", fake_rpc)
    result = asyncio.run(rl.collect_episode(None, agent.adapter, agent, agent.args))
    assert result["reward"] == 0.7
    assert result["steps"] == 1
    assert operations == ["reset", "get_info_for_step", "step", "get_info_for_step"]
    transition = agent.replay[0]
    assert transition.bootstrap == 0
    assert transition.reward == 0.7
    assert transition.episode_reward == 0.7
    assert not transition.success
    torch.testing.assert_close(transition.action, torch.full((2, 14), 0.3))


def test_rating_is_requested_after_standby_and_validated():
    import ast

    tree = ast.parse((ROOT / "iloha_rl.py").read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IlohaRLEnv")
    methods = [n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("get_info_for_step", "resolve_terminal_reward")]
    events = []
    scores = iter(["bad", "nan", "-0.1", "1.1", "0.7"])
    lines = iter(["", None])

    async def standby(*args, **kwargs):
        events.append("standby")

    async def wait_line(prompt):
        events.append("prompt")
        return next(scores)

    namespace = {"np": np, "time": SimpleNamespace(perf_counter=lambda: 102),
                 "stop_motors_for_standby": standby}
    cls = ast.ClassDef(name="Env", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "iloha_rl.py", "exec"), namespace)
    env = namespace["Env"]()
    env.args = SimpleNamespace(reward_mode="terminal-score", episode_time_s=30,
                               standby_motor_disable_delay_s=0)
    env.keyboard = SimpleNamespace(poll=lambda: next(lines, None), clear=lambda: None, wait_line=wait_line)
    env.robot, env.verdict, env.episode_reward = None, None, None
    env.motors_in_standby, env.episode_start_t, env.episode_idx = False, 100, 1
    assert env.get_info_for_step() == (True, False, 0, 0)
    asyncio.run(env.resolve_terminal_reward())
    assert events[0] == "standby"
    assert events.count("prompt") == 5
    assert env.get_info_for_step() == (True, False, 0.7, 0)
    asyncio.run(env.resolve_terminal_reward())
    assert events.count("standby") == 1


def test_positive_partial_rating_is_used_for_vla_updates():
    agent = learner()
    for _ in range(2):
        t = rl.make_transition(cache(), [np.full(14, 0.5)] * 2, cache(), [0, 0.7], True, 2, 0.99)
        t.episode_reward = 0.7
        agent.replay.append(t)
    agent.update()
    assert len(agent.adapter.imitated) == 1
    assert agent.adapter.imitated[0].episode_reward == 0.7


def test_invalid_replay_configuration():
    with pytest.raises(SystemExit):
        rl.parse_args(["--policy_path", "unused", "--replay_capacity", "1"])


def test_xvla_raw_camera_route_matches_eval():
    # Import only the actual method to avoid initializing optional hardware SDKs.
    import ast

    tree = ast.parse((ROOT / "iloha_rl.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IlohaRLEnv")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_observation")
    namespace = {"np": np, "CAMERA_NAMES": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
                 "capture_observation": lambda *_: {cam: np.zeros((480, 640, 3), dtype=np.uint8)
                                                     for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "iloha_rl.py", "exec"), namespace)

    async def state():
        return np.arange(14, dtype=np.float32)

    env = SimpleNamespace(robot=None, state_names=[f"joint_{i}" for i in range(14)],
                          read_state=state, args=SimpleNamespace(observation_format="xvla"), prompt="fold towel")
    obs = asyncio.run(namespace["get_observation"](env))
    assert obs["observation.images.cam_high"].shape == (480, 640, 3)
    assert obs["task"] == "fold towel"
    np.testing.assert_array_equal(obs["observation.state"], np.arange(14))
