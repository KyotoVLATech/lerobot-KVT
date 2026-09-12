"""Hardware-free checks for the replay-to-teleoperation handoff."""

import asyncio
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import iloha_catch_game as game


class TrimEpisodeTests(unittest.TestCase):
    def test_trim_preserves_prefix_and_rounds_up_partial_frames(self):
        actions = np.arange(140).reshape(10, 14)
        for seconds, remaining in ((0, 10), (0.2, 8), (0.21, 7), (0.9, 1)):
            np.testing.assert_array_equal(
                game.trim_episode_end(actions, 10, seconds), actions[:remaining],
            )

    def test_invalid_or_empty_replay_is_rejected(self):
        actions = np.zeros((10, 14))
        for seconds in (-1, float("nan"), float("inf"), 0.91, 1, 2):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                game.trim_episode_end(actions, 10, seconds)


class TeleoperationTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_arm_reaches_robot(self):
        angles = np.arange(14, dtype=np.float32) / 10
        expected = angles.copy()
        expected[0] += np.pi / 2
        expected[7] -= np.pi / 2
        for offset in (0, 7):
            expected[offset + 1] -= np.pi / 2
            expected[offset + 2] = -expected[offset + 2] - np.pi / 2
            expected[offset + 3:offset + 6] *= -1
        for color, active, inactive in (("red", slice(0, 7), slice(7, 14)),
                                        ("blue", slice(7, 14), slice(0, 7))):
            robot = SimpleNamespace(old_action=np.zeros(14), async_send_action=AsyncMock())
            node = game.SingleArmTeleoperation(robot, color, np.zeros((1, 14)), 30)
            node.state = "teleoperation"
            node.datagram_received(struct.pack("<B14f", 1, *angles), None)
            robot.async_send_action.side_effect = asyncio.CancelledError
            with self.assertRaises(asyncio.CancelledError):
                await node.control_robot()
            action = robot.async_send_action.call_args.args[0]
            np.testing.assert_allclose(action[active], expected[active])
            np.testing.assert_array_equal(action[inactive], 0)
            self.assertTrue(robot.async_send_action.call_args.kwargs["use_relative"])
            previous = node.latest_action.copy()
            for packet in (b"short", struct.pack("<B14f", 0, *angles),
                           struct.pack("<B14f", 1, *([float("nan")] * 14))):
                node.datagram_received(packet, None)
                np.testing.assert_array_equal(node.latest_action, previous)

    async def test_reset_disconnect_and_reconnect_keep_robot_connected(self):
        robot = SimpleNamespace(old_action=np.zeros(14), async_send_action=AsyncMock())
        node = game.SingleArmTeleoperation(robot, "red", np.zeros((1, 14)), 30)
        node.replay_complete = True
        loop = asyncio.get_running_loop()
        transport = Mock()

        class WebSocket:
            recv = AsyncMock(return_value=json.dumps({"joint_send_port": 5000}))
            send = AsyncMock()

            def __aiter__(self):
                async def messages():
                    yield json.dumps({"command": "teleoperation"})
                    yield json.dumps({"command": "reset_robot"})
                return messages()

        with patch.object(loop, "create_datagram_endpoint", AsyncMock(return_value=(transport, node))), \
             patch.object(game, "reset_robot_to_home", AsyncMock()) as reset:
            for _ in range(2):
                await node.websocket_handler(WebSocket())
                self.assertFalse(node.connected)
                self.assertIsNone(node.latest_action)
                self.assertTrue(all(
                    "control_connection" not in task.get_coro().__qualname__
                    for task in asyncio.all_tasks() if task is not asyncio.current_task()
                ))
            self.assertEqual(reset.await_count, 4)
            self.assertEqual(transport.close.call_count, 2)

    async def test_startup_returns_home_before_quest_server_without_replaying(self):
        events = []
        robot = SimpleNamespace(
            connect=AsyncMock(side_effect=lambda: events.append("connect")),
            disconnect=AsyncMock(side_effect=lambda: events.append("disconnect")),
        )
        args = SimpleNamespace(dataset_path="unused", episode_index=0, base_speed=1.0,
                               max_speedup=1.0, gripper_margin=0.5, speedup_distance=2.0,
                               gripper_threshold=1e-4, skip_end_seconds=0.0, dry_run=False, color="blue", websocket_port=8080)

        async def teleoperate(port):
            events.append("teleoperation")
            raise asyncio.CancelledError

        with patch.object(game, "load_episode", return_value=(np.zeros((1, 14)), 30)), \
             patch.object(game, "Iloha", return_value=robot), \
             patch.object(game, "reset_robot_to_home", AsyncMock(side_effect=lambda *a, **k: events.append("home"))), \
             patch.object(game, "replay_episode", AsyncMock(side_effect=lambda *a, **k: events.append("replay"))), \
             patch.object(game.SingleArmTeleoperation, "run", AsyncMock(side_effect=teleoperate)):
            with self.assertRaises(asyncio.CancelledError):
                await game.main(args)
        self.assertEqual(events, ["connect", "home", "teleoperation", "disconnect"])

    async def test_teleop_triggers_replay_then_home_then_notification(self):
        for command in ("teleop", "teleoperation"):
            robot = SimpleNamespace(old_action=np.zeros(14), async_send_action=AsyncMock())
            node = game.SingleArmTeleoperation(robot, "red", np.zeros((1, 14)), 30, base_speed=2.0)
            packet = struct.pack("<B14f", 1, *([0.1] * 14))
            replay_entered = asyncio.Event()
            finish_replay = asyncio.Event()
            notified = asyncio.Event()
            events = []

            async def replay(*args, **kwargs):
                events.append("replay")
                self.assertEqual(kwargs["base_speed"], 2.0)
                replay_entered.set()
                await finish_replay.wait()
                return 1

            async def home(*args, **kwargs):
                events.append("home")
                if node.state == "replaying":
                    node.datagram_received(packet, None)
                    self.assertIsNone(node.latest_action)

            async def send(message):
                status = json.loads(message)["status"]
                events.append(status)
                if status == "teleoperation_mode":
                    self.assertEqual(node.state, "teleoperation")
                    self.assertIsNone(node.latest_action)
                    notified.set()

            class WebSocket:
                recv = AsyncMock(return_value=json.dumps({"joint_send_port": 5000}))

                def __aiter__(self):
                    async def messages():
                        node.datagram_received(packet, None)
                        self_test.assertIsNone(node.latest_action)
                        self_test.assertNotIn("replay", events)
                        for old_command in ("record", "recording", "reset_robot", "reset_robot"):
                            yield json.dumps({"command": old_command})
                            await asyncio.sleep(0)
                            self_test.assertFalse(replay_entered.is_set())
                            self_test.assertEqual(node.state, "waiting")
                        self_test.assertEqual(events.count("reset_complete"), 2)
                        events.clear()
                        yield json.dumps({"command": command})
                        await replay_entered.wait()
                        yield json.dumps({"command": command})
                        yield json.dumps({"command": "reset_robot"})
                        node.datagram_received(packet, None)
                        self_test.assertIsNone(node.latest_action)
                        self_test.assertFalse(notified.is_set())
                        finish_replay.set()
                        await asyncio.wait_for(notified.wait(), 1)
                        node.datagram_received(packet, None)
                        self_test.assertIsNotNone(node.latest_action)
                    return messages()

            self_test = self
            ws = WebSocket()
            ws.send = send
            with patch.object(asyncio.get_running_loop(), "create_datagram_endpoint",
                              AsyncMock(return_value=(Mock(), node))), \
                 patch.object(game, "replay_episode", AsyncMock(side_effect=replay)) as replay_mock, \
                 patch.object(game, "reset_robot_to_home", AsyncMock(side_effect=home)):
                await asyncio.wait_for(node.websocket_handler(ws), 2)
            replay_mock.assert_awaited_once()
            self.assertLess(events.index("replay"), events.index("home"))
            self.assertLess(events.index("home"), events.index("teleoperation_mode"))
            self.assertEqual(events.count("error"), 2)
            self.assertTrue(node.replay_complete)


if __name__ == "__main__":
    unittest.main()
