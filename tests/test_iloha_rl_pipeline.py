"""Hardware-free RL regressions, also runnable without the Learner environment.

Load the actual class definitions via AST to avoid optional GPU/robot imports.
Run with: .venv/bin/python -m unittest discover -s tests -p test_iloha_rl_pipeline.py
"""

import ast
import asyncio
import logging
import time
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from iloha_mapping import JOINT_NAMES, aloha_to_iloha, iloha_to_aloha

ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class IlohaEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=100.0)
        namespace = {
            "np": np, "Optional": Optional, "Iloha": object, "KeyboardInput": object, "LeRobotDataset": object,
            "JOINT_NAMES": JOINT_NAMES, "CAMERA_CONFIGS": {},
            "aloha_to_iloha": aloha_to_iloha, "iloha_to_aloha": iloha_to_aloha,
            "time": SimpleNamespace(perf_counter=lambda: self.clock.now),
        }
        env_class = load_class("iloha_rl.py", "IlohaRLEnv", namespace)
        self.sent = []

        async def send(action, **kwargs):
            self.sent.append(action.copy())
            self.robot.old_action = action.copy()

        self.robot = SimpleNamespace(old_action=np.zeros(14), async_send_action=send)
        self.keys = deque()
        keyboard = SimpleNamespace(poll=lambda: self.keys.popleft() if self.keys else None)
        args = SimpleNamespace(
            task="fold", episode_time_s=30., fps=30, state_source="measured",
            disable_robot_relative_safety=False, relative_warmup_seconds=3.,
            absolute_mode_delta_threshold=.2,
        )
        self.env = env_class(self.robot, keyboard, args)
        self.env.episode_idx = 1  # These tests exercise an active episode, after reset.

    def test_timeout_uses_elapsed_time_instead_of_step_count(self):
        self.env.episode_start_t = 100.
        self.clock.now = 101.
        self.env.steps = 900
        self.assertFalse(self.env.get_info_for_step()[0])
        self.env.steps = 1
        self.clock.now = 130.
        self.assertEqual(self.env.get_info_for_step(), (True, False, 0., 0.))

    def test_timeout_holds_even_when_action_arrives_after_inference(self):
        self.env.episode_start_t = 100.
        self.clock.now = 131.
        result = asyncio.run(self.env.step(iloha_to_aloha(np.ones(14))))
        self.assertEqual(self.sent, [])
        np.testing.assert_allclose(result["executed_action"], iloha_to_aloha(self.robot.old_action))

    def test_success_blocks_next_action(self):
        self.keys.append("1")
        asyncio.run(self.env.step(iloha_to_aloha(np.ones(14))))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.env.get_info_for_step(), (True, True, 1., 0.))

    def test_zero_reports_failure(self):
        self.keys.append("0")
        self.assertEqual(self.env.get_info_for_step(), (True, False, 0., 0.))

    def test_sentinel_hold_still_starts_deadline(self):
        asyncio.run(self.env.step(np.zeros(14)))
        self.assertEqual(self.sent, [])
        self.clock.now += 31.
        self.assertTrue(self.env.get_info_for_step()[0])

    def test_invalid_action_shape_is_rejected_before_motor_io(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.env.step(np.zeros((8, 14))))
        self.assertEqual(self.sent, [])


class PrefixTests(unittest.TestCase):
    def test_prefix_uses_normalized_hardware_feedback_and_matching_delayed_observation(self):
        namespace = {
            "np": np, "deque": deque, "logging": logging, "time": time, "ThreadPoolExecutor": ThreadPoolExecutor,
        }
        sampler_class = load_class(
            "libs/expo-ft/expo_ft/utils/loop_utils.py", "AsyncChunkSampler", namespace,
        )
        calls = []
        actor = SimpleNamespace(
            input_transforms=lambda raw: {"actions": raw["actions"] * 2},
            model_config=SimpleNamespace(action_dim=32),
        )
        agent = SimpleNamespace(actor=actor)

        def pre_cache(obs, prefix_padded):
            calls.append((obs["tick"], prefix_padded))
            return None, agent

        def sample(obs, **kwargs):
            return np.zeros((8, 14)), agent, {"executed_padded": np.full((8, 32), 999.)}

        agent.sample_pre_cache = pre_cache
        agent.sample_actions = sample
        sampler = sampler_class(agent, delay=5, replan_steps=8, executed_prefix=True)
        self.assertIsNone(sampler.executor)
        sampler.observe({"tick": 0})
        sampler.sample(agent, {"tick": 0}, {})
        self.assertIsNone(calls[0][1])
        self.assertEqual(len(sampler.exec_hist), 0)
        for tick in range(8):
            if tick:
                sampler.observe({"tick": tick})
            sampler.record_executed(agent, {"tick": tick}, np.full(14, tick / 10.))
        sampler.observe({"tick": 8})
        sampler.sample(agent, {"tick": 8}, {})
        delayed_tick, prefix = calls[-1]
        self.assertEqual(delayed_tick, 3)
        np.testing.assert_allclose(prefix[:, :14], np.repeat(np.arange(3, 8)[:, None] / 5., 14, axis=1))
        np.testing.assert_array_equal(prefix[:, 14:], 0.)
        sampler.on_episode_end()
        self.assertEqual(len(sampler.exec_hist), 0)
        self.assertEqual(len(sampler.obs_hist), 0)

    def test_terminal_observation_skips_inference_and_dispatch(self):
        tree = ast.parse((ROOT / "libs/expo-ft/train_pi_robo.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For))
        dispatch = next(
            n for n in loop.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "not done"
        )
        # Any accidental work in the terminal branch raises due to missing env/sampler.
        exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "dispatch", "exec"), {"done": True})


class ReplayWindowTests(unittest.TestCase):
    def test_wrapped_windows_do_not_cross_newest_to_oldest(self):
        path = ROOT / "libs/expo-ft/expo_ft/data/replay_buffer.py"
        node = next(
            n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_sample_start_indices"
        )
        namespace = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        starts = namespace["_sample_start_indices"]
        np.testing.assert_array_equal(starts(7, 10, 7, 4), [0, 1, 2])
        np.testing.assert_array_equal(starts(10, 10, 3, 4), [3, 4, 5, 6, 7, 8])
        # Insertion pointer 3 is the oldest frame. Every sampled t..t+4
        # window must advance within the chronological history 3,4,...,2.
        for start in starts(10, 10, 3, 4):
            ages = (((start + np.arange(5)) % 10) - 3) % 10
            np.testing.assert_array_equal(np.diff(ages), 1)


class UpdateBatchTests(unittest.TestCase):
    def test_disabling_prefetch_preserves_used_batches_and_update_ratio(self):
        tree = ast.parse((ROOT / "libs/expo-ft/train_pi_robo.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        function = next(
            n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "run_agent_updates"
        )
        # Supply its enclosing state as globals without importing GPU dependencies.
        function.body[0] = ast.Global(names=function.body[0].names)
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "updates", "exec")

        def run(prefetch):
            used = []
            pool = ThreadPoolExecutor(max_workers=1) if prefetch else None

            class Agent:
                rng = 42
                actor_train_state = SimpleNamespace(step=0)
                critic = SimpleNamespace(step=0)

                def replace(self, **kwargs):
                    return self

                def update(self, agent, batch, utd_ratio, actor_batch):
                    used.append((batch["sequence"], len(batch["actions"]), utd_ratio))
                    return self, {}

            def fetch(rng):
                return {"sequence": rng, "actions": np.zeros((64 * 20, 14))}, None, rng + 1

            namespace = {
                "agent": Agent(), "combine_rng": 0, "pending_batch": None,
                "prefetch_pool": pool, "_fetch": fetch,
                "sink": SimpleNamespace(flush_transitions=lambda: None),
                "FLAGS": SimpleNamespace(tqdm=False, utd_ratio=20),
                "tqdm": SimpleNamespace(tqdm=lambda seq, **kwargs: seq),
                "time": time, "logging": logging, "get_batch_info": lambda batch: {},
                "jax": SimpleNamespace(device_put=lambda value, sharding: value, device_get=lambda value: value),
                "replicated_sharding": None,
                "training_log": SimpleNamespace(record_update_time=lambda *args: None),
            }
            exec(code, namespace)
            try:
                namespace["run_agent_updates"](3, {})
                namespace["run_agent_updates"](2, {})
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            return used

        expected = [(i, 1280, 20) for i in range(5)]
        self.assertEqual(run(False), expected)
        self.assertEqual(run(True), expected)


if __name__ == "__main__":
    unittest.main()
