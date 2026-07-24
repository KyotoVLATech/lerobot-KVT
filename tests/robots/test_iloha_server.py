import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from iloha_server import RobotCommunicationNode, parse_args
from lerobot.robots.iloha.iloha import StaleMeasuredStateError


def test_parse_args_disables_realsense_auto_exposure():
    args = parse_args(["--disable-realsense-auto-exposure"])

    assert args.disable_realsense_auto_exposure is True


def test_disable_camera_auto_exposure_configures_supported_sensors():
    auto_exposure_option = object()
    supported_sensor = MagicMock()
    supported_sensor.supports.return_value = True
    unsupported_sensor = MagicMock()
    unsupported_sensor.supports.return_value = False
    camera = MagicMock()
    camera.rs_profile.get_device.return_value.query_sensors.return_value = [
        supported_sensor,
        unsupported_sensor,
    ]
    fake_realsense = SimpleNamespace(
        option=SimpleNamespace(enable_auto_exposure=auto_exposure_option),
    )

    with patch.dict(sys.modules, {"pyrealsense2": fake_realsense}):
        configured_sensor_count = RobotCommunicationNode._disable_camera_auto_exposure(camera)

    assert configured_sensor_count == 1
    supported_sensor.set_option.assert_called_once_with(auto_exposure_option, 0.0)
    unsupported_sensor.set_option.assert_not_called()


def test_initialize_cameras_disables_auto_exposure_after_connect():
    camera = MagicMock()
    node = RobotCommunicationNode(disable_realsense_auto_exposure=True)

    with (
        patch("iloha_server.make_cameras_from_configs", return_value={"cam_high": camera}),
        patch("iloha_server.time.sleep"),
        patch.object(node, "_disable_camera_auto_exposure", return_value=1) as disable_auto_exposure,
    ):
        cameras = node._initialize_cameras()

    assert cameras == {"cam_high": camera}
    camera.connect.assert_called_once_with(warmup=True)
    disable_auto_exposure.assert_called_once_with(camera)


def test_record_episode_skips_transient_stale_motor_state():
    node = RobotCommunicationNode()
    node.is_recording = True
    node.recording_ready = True
    call_count = 0

    def record_frame():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise StaleMeasuredStateError("temporary stale state")
        node.is_recording = False
        return True

    with patch.object(node, "_record_frame_sync", side_effect=record_frame):
        asyncio.run(node.record_episode())

    assert call_count == 2
    assert node.recording_error is None


def test_record_episode_marks_persistent_motor_state_failure():
    node = RobotCommunicationNode()
    node.is_recording = True
    node.recording_ready = True
    node.ROBOT_STATE_FAILURE_ABORT_S = 0.0

    with patch.object(
        node,
        "_record_frame_sync",
        side_effect=StaleMeasuredStateError("persistent stale state"),
    ):
        asyncio.run(node.record_episode())

    assert node.is_recording is False
    assert node.recording_ready is False
    assert "回復しないため記録を中止" in node.recording_error


def test_save_episode_discards_buffer_after_recording_failure():
    node = RobotCommunicationNode()
    node.current_dataset = MagicMock()
    node.current_dataset.writer.episode_buffer = {"size": 8}
    node.recording_error = "motor state did not recover"
    websocket = AsyncMock()

    with patch.object(node, "_prepare_next_episode", new=AsyncMock()) as prepare_next:
        asyncio.run(node.save_episode(websocket))

    node.current_dataset.clear_episode_buffer.assert_called_once_with()
    node.current_dataset.save_episode.assert_not_called()
    prepare_next.assert_awaited_once_with()
    response = json.loads(websocket.send.await_args.args[0])
    assert response["status"] == "save_error"
    assert "不完全なエピソードを破棄" in response["message"]
