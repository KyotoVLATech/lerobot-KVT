import asyncio
import math
from dataclasses import dataclass

import pytest

from lerobot.robots.iloha.iloha_controller.aloha_arm_controller import (
    AlohaArmController,
)
from lerobot.robots.iloha.iloha_controller.robstride.src.constants import (
    ParameterIndex,
)
from lerobot.robots.iloha.iloha_controller import robstride_port_detection


@dataclass
class FakeMotor:
    id: int
    offset: float = 0.0


class FakeRobStrideController:
    def __init__(self, positions: dict[int, float]) -> None:
        self.positions = positions
        self.events: list[tuple] = []

    async def ping(self, motor_id: int) -> bool:
        self.events.append(("ping", motor_id))
        return True

    async def disable(self, motor_id: int) -> bool:
        self.events.append(("disable", motor_id))
        return True

    async def set_mode_pp(self, motor_id: int) -> bool:
        self.events.append(("mode", motor_id))
        return True

    async def apply_pp_limits(self, motor_id: int, *, allow_disabled: bool) -> bool:
        self.events.append(("limits", motor_id, allow_disabled))
        return True

    async def save_parameters(self, motor_id: int) -> bool:
        self.events.append(("save", motor_id))
        return True

    async def get_parameter(self, motor_id: int, parameter: ParameterIndex) -> float:
        assert parameter is ParameterIndex.MECH_POS
        self.events.append(("position", motor_id))
        return self.positions[motor_id]

    async def set_target_position(self, motor_id: int, position: float) -> bytes:
        self.events.append(("target", motor_id, position))
        return b"ok"

    async def enable(self, motor_id: int) -> bool:
        self.events.append(("enable", motor_id))
        return True


def make_controller(positions: dict[int, float]) -> tuple[AlohaArmController, FakeRobStrideController]:
    controller = AlohaArmController.__new__(AlohaArmController)
    controller.robstride_motors = [
        FakeMotor(1, 0.1),
        FakeMotor(2, -0.2),
        FakeMotor(3, 0.0),
    ]
    fake = FakeRobStrideController(positions)
    controller.robstride_controller = fake
    controller._robstride_target_references = {}
    return controller, fake


def test_robstride_initialization_seeds_current_target_before_enable() -> None:
    controller, fake = make_controller({1: -0.2, 2: 0.1, 3: 0.3})

    asyncio.run(controller._setup_robstride_motors("PP"))

    event_names = [event[0] for event in fake.events]
    assert event_names[:6] == ["ping", "ping", "ping", "disable", "disable", "disable"]
    target_events = [event for event in fake.events if event[0] == "target"]
    assert [(event[0], event[1]) for event in target_events] == [
        ("target", 1),
        ("target", 2),
        ("target", 3),
    ]
    assert [event[2] for event in target_events] == pytest.approx([-0.3, 0.3, 0.3])
    for motor_id in (1, 2, 3):
        target_index = next(
            index
            for index, event in enumerate(fake.events)
            if event[0] == "target" and event[1] == motor_id
        )
        assert fake.events[target_index + 1] == ("enable", motor_id)


def test_robstride_targets_use_nearest_two_pi_equivalent() -> None:
    controller, fake = make_controller({1: 0.0, 2: 0.0, 3: 0.0})
    controller._robstride_target_references = {
        1: math.tau - 0.05,
        2: -math.tau + 0.04,
        3: 0.01,
    }

    asyncio.run(controller._set_robstride_positions({1: 0.0, 2: 0.0, 3: 0.0}))

    assert [(event[0], event[1]) for event in fake.events] == [
        ("target", 1),
        ("target", 2),
        ("target", 3),
    ]
    assert [event[2] for event in fake.events] == pytest.approx(
        [math.tau, -math.tau, 0.0]
    )


def test_resolve_robstride_ports_replaces_auto_values(monkeypatch) -> None:
    async def fake_detect() -> robstride_port_detection.RobStridePorts:
        return robstride_port_detection.RobStridePorts(
            left="/dev/ttyUSB1",
            right="/dev/ttyUSB3",
        )

    monkeypatch.setattr(robstride_port_detection, "detect_robstride_ports", fake_detect)

    actual = asyncio.run(
        robstride_port_detection.resolve_robstride_ports(left="auto", right="auto")
    )

    assert actual.left == "/dev/ttyUSB1"
    assert actual.right == "/dev/ttyUSB3"
