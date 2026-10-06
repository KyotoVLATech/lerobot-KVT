"""CPU regression tests for streamed UTD scheduling (learner Python environment).

Run from libs/expo-ft:
  JAX_PLATFORMS=cpu .venv/bin/python ../../scripts/iloha_rl_streaming_tests.py
No checkpoint, dataset or robot is opened.
"""

import sys
import unittest
from dataclasses import fields
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from openpi.training import sharding as openpi_sharding

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs/expo-ft"))

from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner
from expo_ft.agents.vla import pi05
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.data.replay_buffer import ALL_CAMERA_KEYS, PiReplayBuffer, _critic_key_to_storage
from expo_ft.utils.augmentation import make_data_augmentation_fn
from openpi.models.model import Observation
from openpi.training.utils import TrainState as PiTrainState


def replay():
    """Small synthetic replay with the production packing/gather implementation."""
    buffer = PiReplayBuffer.__new__(PiReplayBuffer)
    buffer._size = buffer._capacity = 40
    buffer._insert_index = 0
    buffer._replan_steps = 2
    buffer._delay = 1
    buffer._discount = .99
    buffer._valids_keep_terminal_windows = True
    buffer._seed = 42
    buffer._stage = {}
    buffer._gather_camera_keys = ALL_CAMERA_KEYS
    buffer._gather_storage = tuple(map(_critic_key_to_storage, ALL_CAMERA_KEYS))
    rng = np.random.default_rng(42)
    buffer.dataset_dict = {
        "state": rng.normal(size=(40, 4)).astype(np.float32),
        "actions": rng.normal(size=(40, 3, 4)).astype(np.float32),
        "rewards": np.zeros(40, np.float32), "masks": np.ones(40, np.float32),
        "dones": np.zeros(40, bool), "is_success": np.ones(40, bool),
        "episode_step": np.arange(40, dtype=np.int32),
    }
    for key in buffer._gather_storage:
        buffer.dataset_dict[key] = rng.integers(0, 256, (40, 8, 8, 3), dtype=np.uint8)
        buffer.dataset_dict[key + "_mask"] = np.ones(40, bool)
    return buffer


@struct.dataclass
class ToyLearner:
    """Use real orchestration/preparation, replacing only expensive loss functions."""
    rng: object
    value: object
    critic_steps: object
    actor_steps: object
    edit_steps: object
    data_sharding: object = struct.field(pytree_node=False)
    data_augmentation_fn: object = struct.field(pytree_node=False)
    actor: object = struct.field(pytree_node=False)
    delay: int = struct.field(pytree_node=False, default=1)
    action_dim: int = struct.field(pytree_node=False, default=2)
    state_dim: int = struct.field(pytree_node=False, default=4)
    action_horizon: int = struct.field(pytree_node=False, default=3)
    replan_steps: int = struct.field(pytree_node=False, default=2)
    critic_camera_keys: tuple = struct.field(pytree_node=False, default=ALL_CAMERA_KEYS)
    train_base_actor: bool = struct.field(pytree_node=False, default=True)
    actor_success_only: bool = struct.field(pytree_node=False, default=True)
    n_edit_samples: int = struct.field(pytree_node=False, default=8)
    _infer_cache: object = struct.field(pytree_node=False, default=None)
    target_actor_params: object = struct.field(pytree_node=False, default=None)

    _update_streamed = RealTimeEXPOFTLearner._update_streamed
    _prepare_streamed_critic_batch = RealTimeEXPOFTLearner._prepare_streamed_critic_batch.__wrapped__

    def _streamed_actor_step(self, batch):
        batch = batch.copy()
        rng, key = jax.random.split(self.rng)
        batch["image"] = self.data_augmentation_fn(key, batch["image"])
        batch = prepare_critic_batch(batch, 4, 2, 4, 3, 2, ALL_CAMERA_KEYS)
        return self.replace(rng=rng).update_actor(batch)

    def _streamed_base_actor_step(self, batch):
        return self.update_actor(batch)

    def _streamed_edit_step(self, batch):
        agent, info = self.update_edit_actor(batch)
        agent, temperature_info = agent.update_temperature(info["entropy"])
        return agent, {**info, **temperature_info}

    def update_critic(self, batch):
        rng, key = jax.random.split(self.rng)
        value = self.value * .9 + jnp.mean(batch["observations"]) + jax.random.uniform(key)
        return self.replace(rng=rng, value=value, critic_steps=self.critic_steps + 1), {"critic": value}

    _streamed_critic_step = update_critic

    def update_actor(self, batch):
        rng, key = jax.random.split(self.rng)
        value = self.value + jnp.mean(batch["image"]["base_0_rgb"]) + jax.random.uniform(key)
        return self.replace(rng=rng, value=value, actor_steps=self.actor_steps + 1), {"actor": value}

    def update_edit_actor(self, batch):
        rng, key = jax.random.split(self.rng)
        value = self.value + jnp.mean(batch["observations"]) + jax.random.uniform(key)
        return self.replace(rng=rng, value=value, edit_steps=self.edit_steps + 1), {"entropy": value}

    def update_temperature(self, entropy):
        return self, {"temperature": entropy}

    def cache_infer_params(self):
        return self


class StreamingTests(unittest.TestCase):
    def test_prefix_accumulation_matches_full_batch_loss_optimizer_and_ema(self):
        class TinyModel(nnx.Module):
            def __init__(self):
                self.weight = nnx.Param(jnp.ones(4))
                self.bias = nnx.Param(jnp.zeros(4))

        graph, params = nnx.split(TinyModel())
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.01))
        state = PiTrainState(step=jnp.array(0), params=params, model_def=graph, tx=tx,
                             opt_state=tx.init(params), ema_decay=.8, ema_params=jax.tree.map(lambda x: x.copy(), params))
        config = SimpleNamespace(trainable_filter=nnx.Param,
                                 lr_schedule=SimpleNamespace(create=lambda: lambda step: jnp.array(.01)))
        obs = Observation(images={"base_0_rgb": jnp.zeros((16, 8, 8, 3))},
                          image_masks={"base_0_rgb": jnp.ones(16, bool)},
                          state=jax.random.normal(jax.random.PRNGKey(2), (16, 4)))
        actions = jax.random.normal(jax.random.PRNGKey(3), (16, 3, 4))
        # Deliberately unequal supervised-token counts across the four chunks.
        delays = jnp.array([0] * 4 + [1] * 4 + [2] * 4 + [0, 1, 2, 3])

        def preprocess(key, observation, train):
            return observation.replace(state=observation.state + jax.random.uniform(key, observation.state.shape))

        def velocity(model, observation, x_t, time):
            return model.weight.value * observation.state[:, None, :] + model.bias.value + .1 * x_t

        with patch.object(pi05._model, "preprocess_observation", preprocess), patch.object(pi05, "_forward_get_velocity", velocity):
            expected, expected_info = pi05.train_step_p1_prefix(config, jax.random.PRNGKey(4), state, (obs, actions), delays)
            # Match the production wrapper: explicit in_shardings forbids keyword
            # arguments in this JAX version, so the static microbatch is positional.
            compiled = jax.jit(partial(pi05.train_step_p1_prefix, config),
                               in_shardings=(None,) * 4, static_argnames=("microbatch_size",))
            actual, actual_info = compiled(jax.random.PRNGKey(4), state, (obs, actions), delays, 4)
        self.assertEqual(int(actual.step), 1)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
        for key in expected_info:
            np.testing.assert_allclose(actual_info[key], expected_info[key], rtol=2e-6, atol=2e-6)

    def test_real_actor_wrapper_updates_policy_and_target_with_donation(self):
        mesh = openpi_sharding.make_mesh(1)
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))

        class TinyPolicy:
            model_config = SimpleNamespace(action_dim=4)
            replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

            def __init__(self):
                self.mesh = mesh

            def prepare_batch_for_actor(self, batch):
                return batch["state"], batch["full_actions"]

            def train_step_p1_prefix(self, key, state, batch, delays):
                grad = jax.tree.map(lambda x: jnp.ones_like(x) * jnp.mean(batch[1]), state.params)
                return state.apply_gradients(grads=grad), {"loss": jnp.mean(batch[1])}

            def get_params(self, state):
                return state.params

        state = TrainState.create(apply_fn=lambda: None, params={"weight": jnp.ones(4)}, tx=optax.sgd(.1))
        values = {f.name: None for f in fields(RealTimeEXPOFTLearner)}
        values.update(rng=jax.random.PRNGKey(1), actor=TinyPolicy(), actor_train_state=state,
                      target_actor_params={"weight": jnp.zeros(4)}, actor_tau=.25,
                      data_sharding=sharding, data_augmentation_fn=make_data_augmentation_fn(True),
                      action_dim=2, state_dim=4, action_horizon=3, replan_steps=2,
                      critic_camera_keys=ALL_CAMERA_KEYS, p1_use_prefix_conditioning=True, delay=1)
        learner = RealTimeEXPOFTLearner(**values)
        buffer = replay()
        batch = buffer._convert_to_openpi_format(buffer.sample_jax(2, host_batch=True))
        expected = np.ones(4) - .1 * np.mean(batch["actions"][..., :2])
        learner, info = learner._streamed_actor_step(jax.device_put(batch, sharding))
        jax.block_until_ready(learner)
        self.assertEqual(int(learner.actor_train_state.step), 1)
        np.testing.assert_allclose(learner.actor_train_state.params["weight"], expected, rtol=1e-6)
        np.testing.assert_allclose(learner.target_actor_params["weight"], .25 * expected, rtol=1e-6)
        self.assertTrue(np.isfinite(np.asarray(info["loss"])))

    def test_cpu_sampling_matches_device_sampling_and_owns_snapshot(self):
        buffer = replay()
        seed = jax.random.PRNGKey(12)
        buffer.rng = seed
        expected = buffer.sample_jax(320)
        buffer.rng = seed
        actual = buffer.sample_jax(320, host_batch=True)
        for key in expected:
            self.assertIsInstance(actual[key], np.ndarray)
            np.testing.assert_array_equal(actual[key], np.asarray(expected[key]))
        saved = {k: v.copy() for k, v in actual.items()}
        buffer.sample_jax(320, host_batch=True)
        for key in actual:
            np.testing.assert_array_equal(actual[key], saved[key])

    def test_augmentation_matches_full_batch(self):
        images = {k: jax.random.uniform(jax.random.PRNGKey(i), (6, 8, 8, 3), minval=-1)
                  for i, k in enumerate(ALL_CAMERA_KEYS)}
        for full in (False, True):
            augment = make_data_augmentation_fn(full)
            key = jax.random.PRNGKey(41)
            expected = augment(key, images)
            pieces = [augment(key, {k: v[offset:offset + 2] for k, v in images.items()},
                              sample_offset=offset, total_batch_size=6)
                      for offset in range(0, 6, 2)]
            for cam in images:
                np.testing.assert_allclose(jnp.concatenate([p[cam] for p in pieces]), expected[cam], atol=1e-6)

    def test_host_processor_keeps_all_arrays_on_cpu(self):
        for ratio in (0., .5):
            online, offline = replay(), replay()
            processor = BatchProcessor(online, offline, None, 2, 10, ratio, True, False,
                                       replay_prefetch=1, host_batches=True)
            batch, actor_batch, _ = processor.next_batch(jax.random.PRNGKey(0))
            self.assertEqual(batch["actions"].shape[0], 20)
            self.assertEqual(actor_batch["actions"].shape[0], 2)
            for leaf in jax.tree.leaves((batch, actor_batch)):
                self.assertTrue(isinstance(leaf, np.ndarray) or all(d.platform == "cpu" for d in leaf.devices()))

    def test_streamed_update_preserves_utd_actor_rng_and_last_metrics(self):
        mesh = openpi_sharding.make_mesh(1)
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
        for success_only in (False, True):
            learner = ToyLearner(jax.random.PRNGKey(10), jnp.array(0.), jnp.array(0),
                                 jnp.array(0), jnp.array(0), sharding,
                                 make_data_augmentation_fn(True), SimpleNamespace(model_config=SimpleNamespace(action_dim=4)),
                                 actor_success_only=success_only)
            buffer = replay()
            batch = buffer._convert_to_openpi_format(buffer.sample_jax(20, host_batch=True))
            actor_batch = buffer._convert_to_openpi_format(buffer.sample_jax(2, host_batch=True))
            expected, metrics = RealTimeEXPOFTLearner._update_jit.__wrapped__(
                learner, learner, batch, 10, actor_batch, train_actor=True,
            )
            actual, actual_metrics = learner._update_streamed(batch, 10, actor_batch)
            self.assertEqual(int(actual.critic_steps), 10)
            self.assertEqual(int(actual.actor_steps), 1)
            self.assertEqual(int(actual.edit_steps), 1)
            np.testing.assert_array_equal(actual.rng, expected.rng)
            np.testing.assert_allclose(actual.value, expected.value, rtol=1e-6, atol=1e-6)
            for key in metrics:
                np.testing.assert_allclose(actual_metrics[key], metrics[key], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
