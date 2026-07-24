import math

import pytest

from scripts.iloha_safe_joint_test import (
    DEFAULT_TEST_JOINTS,
    JOINTS,
    Joint,
    build_result,
    logical_gripper_to_motor,
    motor_gripper_to_logical,
    parse_joint_selection,
)


def test_gripper_logical_motor_conversion_round_trip() -> None:
    logical = 0.35

    motor = logical_gripper_to_motor(logical)

    assert motor == pytest.approx(-logical * math.pi / 3)
    assert motor_gripper_to_logical(motor) == pytest.approx(logical)


def test_joint_selection_preserves_requested_order() -> None:
    selected = parse_joint_selection("R7,L2,L1")

    assert [joint.label for joint in selected] == ["R7", "L2", "L1"]
    assert parse_joint_selection("all") == DEFAULT_TEST_JOINTS
    assert JOINTS[0].label == "L1"


def test_result_detects_measured_direction_mismatch() -> None:
    result = build_result(
        Joint("L1", "left", 0),
        initial=0.0,
        target=0.03,
        moved=-0.02,
        final=0.0,
        commanded_delta=0.03,
    )

    assert result["movement_detected"] is True
    assert result["direction_matches_command"] is False
