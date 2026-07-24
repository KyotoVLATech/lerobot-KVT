import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.motors.dynamixel import OperatingMode
from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.robots.iloha.iloha_controller import (
    aloha_controller,
    robstride_port_detection,
)
from lerobot.robots.iloha.iloha_controller.aloha_arm_controller import (
    DYNAMIXEL_CURRENT_MA_PER_UNIT,
    AlohaArm,
    AlohaArmController,
)
from lerobot.robots.iloha.iloha_controller.left_settings import (
    Dynamixel01Constants as LeftDynamixel01Constants,
    Dynamixel02Constants as LeftDynamixel02Constants,
    Dynamixel03Constants as LeftDynamixel03Constants,
    Dynamixel04Constants as LeftDynamixel04Constants,
    Robstride01Constants as LeftRobstride01Constants,
    Robstride02Constants as LeftRobstride02Constants,
    Robstride03Constants as LeftRobstride03Constants,
)
from lerobot.robots.iloha.iloha_controller.right_settings import (
    Dynamixel04Constants as RightDynamixel04Constants,
)
from lerobot.robots.iloha.iloha_controller.robstride.src.robstride import (
    RobStride,
    RobStrideController,
)

LEFT_ROBSTRIDE_CONSTANTS = [
    LeftRobstride01Constants,
    LeftRobstride02Constants,
    LeftRobstride03Constants,
]
LEFT_DYNAMIXEL_CONSTANTS = [
    LeftDynamixel01Constants,
    LeftDynamixel02Constants,
    LeftDynamixel03Constants,
    LeftDynamixel04Constants,
]


class FakeRobstrideController:
    def __init__(self, positions: dict[int, float]) -> None:
        self.positions = positions

    async def get_parameter(self, motor_id, parameter):
        return self.positions[motor_id]


class TransientFailingRobstrideController:
    def __init__(self) -> None:
        self.calls = 0

    async def set_target_position(self, motor_id: int, position: float) -> bytes | None:
        self.calls += 1
        if self.calls == 1:
            return None
        return b"ok"


class FakeDynamixelBus:
    def __init__(self, positions: dict[str, int]) -> None:
        self.positions = positions
        self.motors = dict.fromkeys(positions)
        self.writes = []
        self.sync_read_calls = 0
        self.individual_read_calls = 0

    def sync_read(self, data_name: str, *, normalize: bool, num_retry: int):
        assert data_name == "Present_Position"
        assert normalize is False
        assert num_retry == 0
        self.sync_read_calls += 1
        return self.positions.copy()

    def read(
        self,
        data_name: str,
        motor_name: str,
        *,
        normalize: bool,
        num_retry: int,
    ) -> int:
        assert data_name == "Present_Position"
        assert normalize is False
        assert num_retry == 0
        self.individual_read_calls += 1
        return self.positions[motor_name]

    def sync_write(
        self,
        data_name: str,
        values: int | dict[str, int],
        *,
        normalize: bool,
        num_retry: int,
    ):
        stored_values = values.copy() if isinstance(values, dict) else values
        self.writes.append((data_name, stored_values, normalize, num_retry))

    def write(
        self,
        data_name: str,
        motor: str,
        value: int,
        *,
        normalize: bool,
        num_retry: int,
    ):
        self.writes.append((data_name, motor, value, normalize, num_retry))


class FakeSerialPort:
    def __init__(self) -> None:
        self.reset_input_buffer_calls = 0

    def reset_input_buffer(self) -> None:
        self.reset_input_buffer_calls += 1


class FailingDynamixelBus(FakeDynamixelBus):
    def __init__(self, positions: dict[str, int], failures_before_success: int) -> None:
        super().__init__(positions)
        self.failures_before_success = failures_before_success
        self.serial_port = FakeSerialPort()
        self.port_handler = SimpleNamespace(ser=self.serial_port)

    def sync_read(self, data_name: str, *, normalize: bool, num_retry: int):
        assert data_name == "Present_Position"
        assert normalize is False
        assert num_retry == 0
        self.sync_read_calls += 1
        if self.sync_read_calls <= self.failures_before_success:
            raise ConnectionError("temporary status packet loss")
        return self.positions.copy()


class FakeDynamixelInitializationBus(FakeDynamixelBus):
    def __init__(self, motors, operating_modes) -> None:
        super().__init__(dict.fromkeys(motors, 2048))
        self.motors = motors
        self.operating_modes = operating_modes
        self.model_number_table = {
            motor.model: index
            for index, motor in enumerate(motors.values(), start=1)
        }
        self.is_connected = True
        self.registers = {
            "Present_Position": self.positions.copy(),
            "Goal_Position": self.positions.copy(),
            "Torque_Enable": dict.fromkeys(motors, 0),
            "Operating_Mode": {
                motor_name: mode.value
                for motor_name, mode in operating_modes.items()
            },
        }

    def connect(self, handshake: bool) -> None:
        assert handshake is False

    def set_baudrate(self, baudrate: int) -> None:
        assert baudrate == 57_600

    def ping(self, motor_name: str, *, num_retry: int) -> int:
        assert num_retry == 3
        return self.model_number_table[self.motors[motor_name].model]

    def sync_write(
        self,
        data_name: str,
        values: int | dict[str, int],
        *,
        normalize: bool,
        num_retry: int,
    ) -> None:
        super().sync_write(
            data_name,
            values,
            normalize=normalize,
            num_retry=num_retry,
        )
        if isinstance(values, dict):
            self.registers.setdefault(data_name, {}).update(values)
        else:
            self.registers[data_name] = dict.fromkeys(self.motors, values)

    def read(
        self,
        data_name: str,
        motor_name: str,
        *,
        normalize: bool,
        num_retry: int,
    ) -> int:
        assert normalize is False
        assert num_retry == 0
        return self.registers[data_name][motor_name]

    def write(self, *args, **kwargs) -> None:
        raise AssertionError("Dynamixel initialization must not use individual writes")


class FakeFixedLengthReader:
    def __init__(self, response: bytes) -> None:
        self.response = bytearray(response)
        self.readexactly_sizes: list[int] = []

    def at_eof(self) -> bool:
        return False

    async def read(self, _size: int) -> bytes:
        await asyncio.sleep(1)
        return b""

    async def readexactly(self, size: int) -> bytes:
        self.readexactly_sizes.append(size)
        result = bytes(self.response[:size])
        del self.response[:size]
        return result


class FakeStreamWriter:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def is_closing(self) -> bool:
        return False

    def write(self, frame: bytes) -> None:
        self.frames.append(frame)

    async def drain(self) -> None:
        return None


class FakeDualArmController:
    async def get_pos(self):
        right = AlohaArm(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, -math.pi / 6)
        left = AlohaArm(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -math.pi / 3)
        return right, left


class FakeStagedArmController:
    events: list[tuple[str, str]] = []

    def __init__(self, **params) -> None:
        self.name = params["robstride_port"]

    async def _initialize_controllers(self, *, move_to_initial: bool = True) -> None:
        assert move_to_initial is False
        self.events.append((self.name, "setup"))
        if self.name == "right_robstride":
            raise RuntimeError("right setup failed")

    async def _move_to_initial_position(self) -> None:
        self.events.append((self.name, "move"))

    async def disable(self, *, return_to_initial: bool = True) -> None:
        assert return_to_initial is False
        self.events.append((self.name, "disable"))


@pytest.fixture(autouse=True)
def run_asyncio_to_thread_inline(monkeypatch):
    async def to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)


def make_arm_controller() -> AlohaArmController:
    return AlohaArmController(
        robstride_port="robstride",
        dynamixel_port="dynamixel",
        robstride_constants=LEFT_ROBSTRIDE_CONSTANTS,
        dynamixel_constants=LEFT_DYNAMIXEL_CONSTANTS,
    )


class FakeRobstrideInitializer:
    def __init__(self, positions: dict[int, float]) -> None:
        self.positions = positions
        self.events: list[tuple[str, int, float | None]] = []

    async def ping(self, motor_id: int) -> bool:
        self.events.append(("ping", motor_id, None))
        return True

    async def disable(self, motor_id: int) -> bool:
        self.events.append(("disable", motor_id, None))
        return True

    async def set_mode_pp(self, motor_id: int) -> bool:
        self.events.append(("mode", motor_id, None))
        return True

    async def set_mode_csp(self, motor_id: int) -> bool:
        raise AssertionError(f"CSP must not be used (motor {motor_id})")

    async def apply_pp_limits(
        self,
        motor_id: int,
        *,
        allow_disabled: bool = False,
    ) -> bool:
        assert allow_disabled is True
        self.events.append(("limits", motor_id, None))
        return True

    async def apply_csp_limits(
        self,
        motor_id: int,
        *,
        allow_disabled: bool = False,
    ) -> bool:
        raise AssertionError(
            f"CSP limits must not be used (motor {motor_id}, {allow_disabled=})"
        )

    async def save_parameters(self, motor_id: int) -> bool:
        self.events.append(("save", motor_id, None))
        return True

    async def get_parameter(self, motor_id: int, _parameter) -> float:
        self.events.append(("read", motor_id, None))
        return self.positions[motor_id]

    async def set_target_position(self, motor_id: int, value: float) -> bytes:
        self.events.append(("target", motor_id, value))
        return b"ok"

    async def enable(self, motor_id: int) -> bool:
        self.events.append(("enable", motor_id, None))
        return True


def make_iloha() -> Iloha:
    config = IlohaConfig(
        right_robstride_port="right_robstride",
        left_robstride_port="left_robstride",
        right_dynamixel_port="right_dynamixel",
        left_dynamixel_port="left_dynamixel",
    )
    return Iloha(config)


def test_both_grippers_use_current_position_mode() -> None:
    assert LeftDynamixel04Constants.OPERATING_MODE is OperatingMode.CURRENT_POSITION
    assert RightDynamixel04Constants.OPERATING_MODE is OperatingMode.CURRENT_POSITION


def test_robstride_limits_are_configured_once_for_operation() -> None:
    controller = AlohaArmController(
        robstride_port="robstride",
        dynamixel_port="dynamixel",
        robstride_constants=LEFT_ROBSTRIDE_CONSTANTS,
        dynamixel_constants=LEFT_DYNAMIXEL_CONSTANTS,
        robstride_current_limit={1: 4.0, 2: 16.0, 3: 4.0},
    )

    assert [motor.limits.pp_limit_cur for motor in controller.robstride_motors] == [
        4.0,
        16.0,
        4.0,
    ]


def test_arm_controller_reads_real_positions_from_both_motor_buses() -> None:
    controller = make_arm_controller()
    controller.robstride_controller = FakeRobstrideController({1: 0.1, 2: 0.2, 3: 0.3})
    pulse_positions = {
        "motor4": 2048 + 1024,
        "motor5": 2048 - 1024,
        "motor6": 2048 + 512,
        "motor7": 2048 - 512,
    }
    bus = FakeDynamixelBus(pulse_positions)
    controller.dynamixel_controller = bus

    actual = asyncio.run(controller.get_pos()).get_positions()

    assert actual[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert actual[3:] == pytest.approx([math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4])
    assert bus.sync_read_calls == 1
    assert bus.individual_read_calls == 0


def test_arm_controller_retries_dynamixel_sync_read_after_resetting_input_buffer() -> None:
    controller = make_arm_controller()
    positions = {
        "motor4": 2048,
        "motor5": 2049,
        "motor6": 2050,
        "motor7": 2051,
    }
    bus = FailingDynamixelBus(positions, failures_before_success=1)
    controller.dynamixel_controller = bus

    actual = controller._sync_read_dynamixel_registers("Present_Position")

    assert actual == positions
    assert bus.sync_read_calls == 2
    assert bus.serial_port.reset_input_buffer_calls == 1


def test_arm_controller_reports_dynamixel_port_after_retry_exhaustion() -> None:
    controller = make_arm_controller()
    bus = FailingDynamixelBus(
        {
            "motor4": 2048,
            "motor5": 2049,
            "motor6": 2050,
            "motor7": 2051,
        },
        failures_before_success=2,
    )
    controller.dynamixel_controller = bus

    with pytest.raises(ConnectionError, match=r"port=dynamixel, attempts=2"):
        controller._sync_read_dynamixel_registers("Present_Position")

    assert bus.sync_read_calls == 2
    assert bus.serial_port.reset_input_buffer_calls == 1


def test_robstride_positions_are_wrapped_at_two_pi() -> None:
    controller = make_arm_controller()
    controller.robstride_controller = FakeRobstrideController(
        {
            1: math.tau - 0.05,
            2: -math.tau + 0.04,
            3: 0.0,
        }
    )
    controller.dynamixel_controller = FakeDynamixelBus(
        {
            "motor4": 2048,
            "motor5": 2048,
            "motor6": 2048,
            "motor7": 2048,
        }
    )

    actual = asyncio.run(controller.get_pos()).get_positions()

    assert actual[:3] == pytest.approx([-0.05, 0.04, 0.0])


def test_robstride_binary_response_can_contain_crlf_in_payload() -> None:
    controller = RobStrideController(
        port="robstride",
        motors=[RobStride(id=1, offset=0.0)],
    )
    assert controller.log_latency_stats is False
    response = b"AT" + b"\x00\x00\x00\r\n" + b"\x00" * 8 + b"\r\n"
    assert len(response) == 17
    reader = FakeFixedLengthReader(b"\r\n" + response)
    writer = FakeStreamWriter()
    controller.reader = reader
    controller.writer = writer
    request = b"AT" + b"\x00" * 13 + b"\r\n"

    actual = asyncio.run(controller._send_and_receive(request))

    assert actual == response
    assert reader.readexactly_sizes == [1, 1, 1, 1, 15]
    assert not controller._latencies


def test_robstride_pp_initialization_seeds_current_target_before_enable() -> None:
    controller = make_arm_controller()
    fake = FakeRobstrideInitializer({1: -0.2, 2: 0.1, 3: 0.3})
    controller.robstride_controller = fake

    asyncio.run(controller._setup_robstride_motors())

    event_names = [event[0] for event in fake.events]
    first_enable = event_names.index("enable")
    assert all(name != "enable" for name in event_names[:first_enable])
    assert event_names[:6] == ["ping", "ping", "ping", "disable", "disable", "disable"]
    assert [event for event in fake.events if event[0] == "target"] == [
        ("target", 1, -0.2),
        ("target", 2, 0.1),
        ("target", 3, 0.3),
    ]
    for motor_id in (1, 2, 3):
        target_index = fake.events.index(
            (
                "target",
                motor_id,
                fake.positions[motor_id],
            )
        )
        assert fake.events[target_index + 1] == ("enable", motor_id, None)
    assert event_names.count("enable") == 3


def test_robstride_targets_use_nearest_two_pi_equivalent() -> None:
    controller = make_arm_controller()
    fake = FakeRobstrideInitializer(
        {
            1: math.tau - 0.05,
            2: -math.tau + 0.04,
            3: 0.01,
        }
    )
    controller.robstride_controller = fake
    asyncio.run(controller._setup_robstride_motors())
    fake.events.clear()

    asyncio.run(
        controller._set_robstride_positions(
            {
                1: 0.0,
                2: 0.0,
                3: 0.0,
            }
        )
    )

    assert [event for event in fake.events if event[0] == "target"] == pytest.approx(
        [
            ("target", 1, math.tau),
            ("target", 2, -math.tau),
            ("target", 3, 0.0),
        ]
    )


def test_robstride_target_write_retries_transient_failure() -> None:
    controller = make_arm_controller()
    fake = TransientFailingRobstrideController()
    controller.robstride_controller = fake

    asyncio.run(controller._set_robstride_positions({1: 0.25}))

    assert fake.calls == 2
    assert controller._robstride_target_references[1] == pytest.approx(0.25)


def test_initial_move_detects_motor_moving_away_from_origin() -> None:
    controller = make_arm_controller()

    diverging_ids = controller._find_diverging_robstride_motors(
        np.asarray([0.20, -0.01, 0.01]),
        np.asarray([0.10, -0.07, 0.08]),
    )

    assert diverging_ids == [2, 3]


def test_arm_controller_writes_standard_dynamixel_raw_positions_and_current() -> None:
    controller = make_arm_controller()
    bus = FakeDynamixelBus({})
    controller.dynamixel_controller = bus

    asyncio.run(controller._set_dynamixel_positions_rad({"motor4": math.pi / 2, "motor7": -math.pi / 4}))
    asyncio.run(controller.set_gripper_current(300.0))

    assert bus.writes[0] == (
        "Goal_Position",
        {"motor4": 3072, "motor7": 1536},
        False,
        3,
    )
    assert bus.writes[1] == (
        "Goal_Current",
        "motor7",
        int(300.0 / DYNAMIXEL_CURRENT_MA_PER_UNIT),
        False,
        3,
    )


def test_dynamixel_initialization_uses_status_free_sync_writes() -> None:
    controller = make_arm_controller()
    bus = FakeDynamixelInitializationBus(
        controller.dynamixel_motors,
        controller.dynamixel_operating_modes,
    )
    controller.dynamixel_controller = bus

    asyncio.run(controller._connect_dynamixel())

    assert [write[0] for write in bus.writes] == [
        "Torque_Enable",
        "Operating_Mode",
        "Goal_Position",
        "Goal_Current",
        "Torque_Enable",
    ]
    assert bus.writes[0][1] == 0
    assert bus.writes[-1][1] == 1


def test_dual_arm_setup_failure_prevents_both_initial_moves(monkeypatch) -> None:
    FakeStagedArmController.events = []
    monkeypatch.setattr(
        aloha_controller,
        "AlohaArmController",
        FakeStagedArmController,
    )
    controller = aloha_controller.AlohaController(
        right_robstride_port="right_robstride",
        left_robstride_port="left_robstride",
        right_dynamixel_port="right_dynamixel",
        left_dynamixel_port="left_dynamixel",
        right_robstride_constants=[object()],
        right_dynamixel_constants=[object()],
        left_robstride_constants=[object()],
        left_dynamixel_constants=[object()],
    )

    with pytest.raises(RuntimeError, match="right setup failed"):
        asyncio.run(controller._initialize_controllers())

    assert not [
        event for event in FakeStagedArmController.events if event[1] == "move"
    ]


def test_iloha_measured_state_uses_left_right_order_and_inverse_gripper_mapping() -> None:
    robot = make_iloha()
    robot.aloha = FakeDualArmController()

    measured = asyncio.run(robot.refresh_measured_state())

    np.testing.assert_allclose(
        measured,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.5],
    )
    observation = robot.get_observation()
    np.testing.assert_allclose(
        [observation[f"joint_{index}"] for index in range(14)],
        measured,
    )


def test_robstride_ports_are_detected_by_motor_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        robstride_port_detection,
        "_find_candidate_ports",
        lambda: ["/dev/ttyUSB8", "/dev/ttyUSB2", "/dev/ttyUSB5"],
    )

    async def classify(port: str) -> str | None:
        return {
            "/dev/ttyUSB8": "right",
            "/dev/ttyUSB2": None,
            "/dev/ttyUSB5": "left",
        }[port]

    monkeypatch.setattr(robstride_port_detection, "_classify_port", classify)

    ports = asyncio.run(robstride_port_detection.detect_robstride_ports())

    assert ports.left == "/dev/ttyUSB5"
    assert ports.right == "/dev/ttyUSB8"


def test_robstride_port_classification_probes_expected_ids(monkeypatch) -> None:
    class FakeRobStrideController:
        def __init__(
            self,
            port: str,
            motors: list,
            log_timeout_errors: bool,
        ) -> None:
            self.port = port
            self.probed_ids = []
            assert log_timeout_errors is False

        async def connect(self) -> bool:
            return True

        async def ping(self, motor_id: int) -> bool:
            self.probed_ids.append(motor_id)
            return motor_id in {4, 5, 6}

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr(
        robstride_port_detection,
        "RobStrideController",
        FakeRobStrideController,
    )

    side = asyncio.run(robstride_port_detection._classify_port("/dev/ttyUSB9"))

    assert side == "right"


def test_robstride_auto_detection_rejects_ambiguous_results(monkeypatch) -> None:
    monkeypatch.setattr(
        robstride_port_detection,
        "_find_candidate_ports",
        lambda: ["/dev/ttyUSB2", "/dev/ttyUSB3"],
    )

    async def classify(_port: str) -> str:
        return "left"

    monkeypatch.setattr(robstride_port_detection, "_classify_port", classify)

    with pytest.raises(RuntimeError, match="一意に判定できません"):
        asyncio.run(robstride_port_detection.detect_robstride_ports())
