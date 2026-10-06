"""Exercise the real learner/client WebSocket protocol with a simulated robot.

Run with the robot environment: .venv/bin/python scripts/iloha_rl_protocol_verify.py
No robot or camera objects are constructed; no serial I/O is performed.
"""

import asyncio
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import websockets.asyncio.client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "libs/expo-ft"))
sys.path.insert(0, str(ROOT / "libs/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src"))

from expo_ft.env.env_client import EnvClient  # noqa: E402

import iloha_rl  # noqa: E402
from iloha_mapping import iloha_to_aloha  # noqa: E402


async def main():
    keys = deque()
    sends = []
    robot = SimpleNamespace(old_action=np.zeros(14, np.float32))

    async def send(action, **kwargs):
        sends.append(action.copy())
        robot.old_action = action.copy()

    async def read():
        return robot.old_action.copy()

    robot.async_send_action = send
    robot.async_read_joint_state = read
    keyboard = SimpleNamespace(poll=lambda: keys.popleft() if keys else None)
    args = SimpleNamespace(
        task=iloha_rl.DEFAULT_PROMPT, fps=30, episode_time_s=30., state_source="measured",
        disable_robot_relative_safety=False, relative_warmup_seconds=3.,
        absolute_mode_delta_threshold=.2, step_timing_threshold_ms=30.,
    )
    env = iloha_rl.IlohaRLEnv(robot, keyboard, args)
    raw = {name: np.full((480, 640, 3), 128, np.uint8) for name in iloha_rl.CAMERA_NAMES}
    client = EnvClient(host="127.0.0.1", port=0)
    client._ensure_server()
    port = client._server.socket.getsockname()[1]
    try:
        with patch.object(iloha_rl, "capture_observation", return_value=raw):
            async with websockets.asyncio.client.connect(
                f"ws://127.0.0.1:{port}", compression=None, max_size=None, ping_interval=None,
            ) as websocket:
                handler = asyncio.create_task(iloha_rl.serve_learner(websocket, env, args))
                try:
                    env_id, prompt = await asyncio.to_thread(client.create_env, {})
                    assert prompt == args.task
                    obs = await asyncio.to_thread(client.get_observation, env_id)
                    assert obs["state"].shape == (14,)
                    assert all(obs[f"{name}_image"].shape == (3, 224, 224) for name in iloha_rl.CAMERA_NAMES)
                    assert await asyncio.to_thread(client.get_info_for_step, env_id) == (False, False, 0., 1.)
                    target = iloha_to_aloha(np.full(14, .01, np.float32))
                    executed, action_type = await asyncio.to_thread(client.step, env_id, target)
                    np.testing.assert_allclose(executed, target, atol=1e-6)
                    assert action_type == "policy" and len(sends) == 1
                    keys.append("1")
                    await asyncio.to_thread(client.get_observation, env_id)
                    assert await asyncio.to_thread(client.get_info_for_step, env_id) == (True, True, 1., 0.)
                    await asyncio.to_thread(client.step, env_id, target + .1)
                    assert len(sends) == 1  # Terminal action is held locally.
                    env._reset_episode_state()
                    keys.append("0")
                    await asyncio.to_thread(client.get_observation, env_id)
                    assert await asyncio.to_thread(client.get_info_for_step, env_id) == (True, False, 0., 0.)
                    print("PASS: real WebSocket serialization, observations, action feedback, 0/1 verdicts, terminal hold", flush=True)
                finally:
                    handler.cancel()
                    await asyncio.gather(handler, return_exceptions=True)
    finally:
        await asyncio.to_thread(client.close)


if __name__ == "__main__":
    asyncio.run(main())
