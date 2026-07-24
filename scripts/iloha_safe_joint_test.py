#!/usr/bin/env python3
"""iLoHAの原点と各関節の正方向を、安全側の設定で対話確認する。"""

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.robots.iloha.iloha_controller.aloha_arm_controller import (
    DYNAMIXEL_BAUDRATE,
    DYNAMIXEL_CURRENT_MA_PER_UNIT,
    DYNAMIXEL_NUM_RETRY,
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
    Dynamixel01Constants as RightDynamixel01Constants,
    Dynamixel02Constants as RightDynamixel02Constants,
    Dynamixel03Constants as RightDynamixel03Constants,
    Dynamixel04Constants as RightDynamixel04Constants,
    Robstride01Constants as RightRobstride01Constants,
    Robstride02Constants as RightRobstride02Constants,
    Robstride03Constants as RightRobstride03Constants,
)
from lerobot.robots.iloha.iloha_controller.robstride.src.constants import ParameterIndex
from lerobot.robots.iloha.iloha_controller.robstride.src.robstride import (
    RobStrideController,
    RobStrideLimits,
)
from lerobot.robots.iloha.iloha_controller.robstride_port_detection import (
    AUTO_PORT,
    resolve_robstride_ports,
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
RIGHT_ROBSTRIDE_CONSTANTS = [
    RightRobstride01Constants,
    RightRobstride02Constants,
    RightRobstride03Constants,
]
RIGHT_DYNAMIXEL_CONSTANTS = [
    RightDynamixel01Constants,
    RightDynamixel02Constants,
    RightDynamixel03Constants,
    RightDynamixel04Constants,
]
SAFETY_NUM_RETRY = 3


@dataclass(frozen=True)
class Joint:
    label: str
    side: str
    arm_index: int
    description: str = ""

    @property
    def is_robstride(self) -> bool:
        return self.arm_index < 3

    @property
    def is_gripper(self) -> bool:
        return self.arm_index == 6


AXIS_DESCRIPTIONS = (
    "根元ヨー",
    "肩ピッチ",
    "肘ピッチ",
    "前腕ヨー",
    "手首ピッチ",
    "手首ヨー",
    "グリッパー",
)
JOINTS = [
    *(
        Joint(f"L{index + 1}", "left", index, f"左{AXIS_DESCRIPTIONS[index]}")
        for index in range(7)
    ),
    *(
        Joint(f"R{index + 1}", "right", index, f"右{AXIS_DESCRIPTIONS[index]}")
        for index in range(7)
    ),
]
JOINTS_BY_LABEL = {joint.label: joint for joint in JOINTS}
DEFAULT_TEST_JOINTS = [
    *(JOINTS_BY_LABEL[f"L{index}"] for index in range(7, 0, -1)),
    *(JOINTS_BY_LABEL[f"R{index}"] for index in range(7, 0, -1)),
]


class SafeArmConnection:
    def __init__(
        self,
        side: str,
        robstride_port: str,
        dynamixel_port: str,
        robstride_constants: list,
        dynamixel_constants: list,
        args: argparse.Namespace,
    ) -> None:
        self.side = side
        self.args = args
        self.controller = AlohaArmController(
            robstride_port=robstride_port,
            dynamixel_port=dynamixel_port,
            robstride_constants=robstride_constants,
            dynamixel_constants=dynamixel_constants,
            robstride_current_limit=args.robstride_current_a,
        )
        for motor in self.controller.robstride_motors:
            motor.limits = RobStrideLimits(
                pp_vel_max=args.robstride_velocity_rad_s,
                pp_acc_set=args.robstride_acceleration_rad_s2,
                pp_limit_cur=args.robstride_current_a,
            )

    async def connect_safely(self) -> None:
        arm = self.controller
        arm.robstride_controller = RobStrideController(
            port=arm.robstride_port,
            motors=arm.robstride_motors,
            log_latency_stats=False,
        )
        if not await arm.robstride_controller.connect():
            raise RuntimeError(f"{self.side}: RobStride接続に失敗しました")

        arm.dynamixel_controller = DynamixelMotorsBus(
            port=arm.dynamixel_port,
            motors=arm.dynamixel_motors,
        )
        arm.dynamixel_controller.connect(False)
        arm.dynamixel_controller.set_baudrate(DYNAMIXEL_BAUDRATE)

        # 試験開始時は必ず全軸を無効化する。位置指令はまだ送らない。
        for motor in arm.robstride_motors:
            await self._disable_robstride_with_retry(motor.id)
        arm.dynamixel_controller.disable_torque(num_retry=DYNAMIXEL_NUM_RETRY)

        self._validate_dynamixel_configuration()
        arm.dynamixel_controller.sync_write(
            "Profile_Acceleration",
            self.args.dynamixel_profile_acceleration,
            normalize=False,
            num_retry=DYNAMIXEL_NUM_RETRY,
        )
        arm.dynamixel_controller.sync_write(
            "Profile_Velocity",
            self.args.dynamixel_profile_velocity,
            normalize=False,
            num_retry=DYNAMIXEL_NUM_RETRY,
        )
        arm.dynamixel_controller.sync_write(
            "Goal_Current",
            {"motor7": int(self.args.gripper_current_ma / DYNAMIXEL_CURRENT_MA_PER_UNIT)},
            normalize=False,
            num_retry=DYNAMIXEL_NUM_RETRY,
        )

    async def _disable_robstride_with_retry(self, motor_id: int) -> None:
        controller = self.controller.robstride_controller
        assert controller is not None
        for _ in range(1 + SAFETY_NUM_RETRY):
            if await controller.disable(motor_id):
                return
        raise RuntimeError(f"{self.side}: RobStride ID {motor_id}を無効化できません")

    def _validate_dynamixel_configuration(self) -> None:
        arm = self.controller
        assert arm.dynamixel_controller is not None
        bus = arm.dynamixel_controller

        for motor_name, motor in bus.motors.items():
            model_number = bus.ping(
                motor_name,
                num_retry=DYNAMIXEL_NUM_RETRY,
                raise_on_error=True,
            )
            expected_model = bus.model_number_table[motor.model]
            if model_number != expected_model:
                raise RuntimeError(
                    f"{self.side} {motor_name}: model={model_number}, expected={expected_model}"
                )

            mode = bus.read(
                "Operating_Mode",
                motor_name,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            expected_mode = arm.dynamixel_operating_modes[motor_name].value
            if mode != expected_mode:
                raise RuntimeError(
                    f"{self.side} {motor_name}: operating_mode={mode}, expected={expected_mode}"
                )

    async def read_logical_positions(self) -> list[float]:
        positions = (await self.controller.get_pos()).get_positions()
        positions[6] = motor_gripper_to_logical(positions[6])
        return positions

    async def test_joint(self, joint: Joint, logical_step: float) -> dict:
        if joint.is_robstride:
            return await self._test_robstride_joint(joint, logical_step)
        return await self._test_dynamixel_joint(joint, logical_step)

    async def _test_robstride_joint(self, joint: Joint, logical_step: float) -> dict:
        arm = self.controller
        controller = arm.robstride_controller
        assert controller is not None
        motor = arm.robstride_motors[joint.arm_index]

        raw_position = await arm._get_robstride_position(motor)
        if raw_position is None:
            raise RuntimeError(f"{joint.label}: 現在位置を取得できません")
        initial = raw_position - motor.offset
        target = initial + logical_step
        enabled = False

        try:
            if not await self._set_robstride_mode_pp_with_retry(motor.id):
                raise RuntimeError(f"{joint.label}: PPモード設定に失敗しました")

            # トルクON前に目標を現在位置へ合わせ、古い目標への急移動を防ぐ。
            await controller.set_target_position(motor.id, initial)
            await self._apply_robstride_limits_before_enable(motor.id)
            enabled_ok = False
            for _ in range(1 + SAFETY_NUM_RETRY):
                if await controller.enable(motor.id):
                    enabled_ok = True
                    break
            if not enabled_ok:
                raise RuntimeError(f"{joint.label}: モータ有効化に失敗しました")
            enabled = True

            await controller.set_target_position(motor.id, target)
            announce_movement(joint, self.args.hold_s)
            await asyncio.sleep(self.args.hold_s)
            moved_raw = await arm._get_robstride_position(motor)
            if moved_raw is None:
                raise RuntimeError(f"{joint.label}: 移動後位置を取得できません")
            moved = moved_raw - motor.offset
        finally:
            if enabled:
                try:
                    print(f"  {joint.label} ({joint.description})を開始位置へ戻します...")
                    await controller.set_target_position(motor.id, initial)
                    await asyncio.sleep(self.args.return_s)
                finally:
                    await self._disable_robstride_with_retry(motor.id)

        final_raw = await arm._get_robstride_position(motor)
        final = None if final_raw is None else final_raw - motor.offset
        return build_result(joint, initial, target, moved, final, logical_step)

    async def _set_robstride_mode_pp_with_retry(self, motor_id: int) -> bool:
        controller = self.controller.robstride_controller
        assert controller is not None
        for _ in range(1 + SAFETY_NUM_RETRY):
            if await controller.set_mode_pp(motor_id):
                return True
        return False

    async def _apply_robstride_limits_before_enable(self, motor_id: int) -> None:
        controller = self.controller.robstride_controller
        assert controller is not None
        motor = next(
            motor
            for motor in self.controller.robstride_motors
            if motor.id == motor_id
        )
        assert motor.limits is not None

        settings = (
            (ParameterIndex.VEL_MAX, motor.limits.pp_vel_max, "PP velocity", "rad/s"),
            (ParameterIndex.ACC_SET, motor.limits.pp_acc_set, "PP acceleration", "rad/s^2"),
            (ParameterIndex.LIMIT_CUR, motor.limits.pp_limit_cur, "PP current limit", "A"),
        )
        for parameter, value, name, unit in settings:
            assert value is not None
            setting_ok = False
            for _ in range(1 + SAFETY_NUM_RETRY):
                if await controller._set_float_parameter(
                    motor_id,
                    parameter,
                    value,
                    name,
                    unit,
                ):
                    setting_ok = True
                    break
            if not setting_ok:
                raise RuntimeError(
                    f"RobStride ID {motor_id}: {name}の安全設定に失敗しました"
                )

    async def _test_dynamixel_joint(self, joint: Joint, logical_step: float) -> dict:
        arm = self.controller
        bus = arm.dynamixel_controller
        assert bus is not None
        motor_name = f"motor{joint.arm_index + 1}"

        initial_raw = int(
            bus.read(
                "Present_Position",
                motor_name,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
        )
        initial_motor = arm._pulse_to_radian(
            initial_raw - arm.dynamixel_offsets[motor_name]
        )
        initial = (
            motor_gripper_to_logical(initial_motor)
            if joint.is_gripper
            else initial_motor
        )
        motor_step = logical_gripper_to_motor(logical_step) if joint.is_gripper else logical_step
        target_raw = initial_raw + arm._radian_to_pulse(motor_step)
        target = initial + logical_step
        enabled = False

        try:
            # トルクON前に目標を現在位置へ一致させる。
            bus.sync_write(
                "Goal_Position",
                {motor_name: initial_raw},
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            bus.enable_torque(motor_name, num_retry=DYNAMIXEL_NUM_RETRY)
            enabled = True
            bus.sync_write(
                "Goal_Position",
                {motor_name: target_raw},
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            announce_movement(joint, self.args.hold_s)
            await asyncio.sleep(self.args.hold_s)
            moved_raw = int(
                bus.read(
                    "Present_Position",
                    motor_name,
                    normalize=False,
                    num_retry=DYNAMIXEL_NUM_RETRY,
                )
            )
            moved_motor = arm._pulse_to_radian(
                moved_raw - arm.dynamixel_offsets[motor_name]
            )
            moved = (
                motor_gripper_to_logical(moved_motor)
                if joint.is_gripper
                else moved_motor
            )
        finally:
            if enabled:
                try:
                    print(f"  {joint.label} ({joint.description})を開始位置へ戻します...")
                    bus.sync_write(
                        "Goal_Position",
                        {motor_name: initial_raw},
                        normalize=False,
                        num_retry=DYNAMIXEL_NUM_RETRY,
                    )
                    await asyncio.sleep(self.args.return_s)
                finally:
                    bus.disable_torque(motor_name, num_retry=DYNAMIXEL_NUM_RETRY)

        final_raw = int(
            bus.read(
                "Present_Position",
                motor_name,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
        )
        final_motor = arm._pulse_to_radian(final_raw - arm.dynamixel_offsets[motor_name])
        final = motor_gripper_to_logical(final_motor) if joint.is_gripper else final_motor
        return build_result(joint, initial, target, moved, final, logical_step)

    async def close(self) -> None:
        arm = self.controller
        if arm.robstride_controller is not None:
            for motor in arm.robstride_motors:
                try:
                    await self._disable_robstride_with_retry(motor.id)
                except Exception as exc:
                    print(f"WARNING: {self.side} RobStride ID {motor.id}: {exc}")
            await arm.robstride_controller.disconnect()
            arm.robstride_controller = None
        if arm.dynamixel_controller is not None:
            try:
                if arm.dynamixel_controller.is_connected:
                    arm.dynamixel_controller.disable_torque(
                        num_retry=DYNAMIXEL_NUM_RETRY
                    )
            finally:
                if arm.dynamixel_controller.is_connected:
                    arm.dynamixel_controller.disconnect(False)
                arm.dynamixel_controller = None


def logical_gripper_to_motor(value: float) -> float:
    return -value * math.pi / 3


def motor_gripper_to_logical(value: float) -> float:
    return -value * 3 / math.pi


def build_result(
    joint: Joint,
    initial: float,
    target: float,
    moved: float,
    final: float | None,
    commanded_delta: float,
) -> dict:
    measured_delta = moved - initial
    direction_matches_command = measured_delta * commanded_delta > 0
    movement_detected = abs(measured_delta) >= abs(commanded_delta) * 0.2
    return {
        "joint": joint.label,
        "initial": initial,
        "target": target,
        "moved": moved,
        "final": final,
        "commanded_delta": commanded_delta,
        "measured_delta": measured_delta,
        "direction_matches_command": direction_matches_command,
        "movement_detected": movement_detected,
    }


def parse_joint_selection(value: str) -> list[Joint]:
    if value.lower() == "all":
        return DEFAULT_TEST_JOINTS.copy()
    requested = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = set(requested) - JOINTS_BY_LABEL.keys()
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown joints: {sorted(unknown)}; use L1-L7,R1-R7 or all"
        )
    return [JOINTS_BY_LABEL[label] for label in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joints",
        type=parse_joint_selection,
        default=DEFAULT_TEST_JOINTS.copy(),
    )
    parser.add_argument("--step-rad", type=float, default=0.03)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--return-s", type=float, default=1.5)
    parser.add_argument("--robstride-current-a", type=float, default=1.0)
    parser.add_argument("--robstride-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--robstride-acceleration-rad-s2", type=float, default=0.30)
    parser.add_argument("--dynamixel-profile-velocity", type=int, default=10)
    parser.add_argument("--dynamixel-profile-acceleration", type=int, default=5)
    parser.add_argument("--gripper-current-ma", type=float, default=150.0)
    parser.add_argument("--left-robstride-port", default=AUTO_PORT)
    parser.add_argument("--right-robstride-port", default=AUTO_PORT)
    parser.add_argument("--left-dynamixel-port", default="/dev/ttyUSB_LeftDynamixel")
    parser.add_argument("--right-dynamixel-port", default="/dev/ttyUSB_RightDynamixel")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not 0 < args.step_rad <= 0.10:
        parser.error("--step-rad must be in (0, 0.10]")
    if not 0 < args.robstride_current_a <= 2.0:
        parser.error("--robstride-current-a must be in (0, 2.0]")
    if not 0 < args.gripper_current_ma <= 300.0:
        parser.error("--gripper-current-ma must be in (0, 300]")
    return args


def print_positions(left: list[float], right: list[float]) -> None:
    print("\n現在姿勢の実測値（iLoHA論理座標）")
    for joint, value in zip(JOINTS, [*left, *right], strict=True):
        print(f"  {joint.label} ({joint.description}): {format_joint_value(joint, value)}")


def prompt_for_test(joint: Joint, step: float) -> str:
    return input(
        f"\n{joint.label} ({joint.description})を論理正方向へ "
        f"{format_joint_delta(joint, step)}だけ動かします。"
        "周囲と支持を確認し、Enter=実行 / s=skip / q=終了: "
    ).strip().lower()


def format_joint_value(joint: Joint, value: float) -> str:
    if joint.is_gripper:
        motor_rad = logical_gripper_to_motor(value)
        return f"logical={value:+.6f}, motor={motor_rad:+.6f} rad"
    return f"{value:+.6f} rad"


def format_optional_joint_value(joint: Joint, value: float | None) -> str:
    return "unavailable" if value is None else format_joint_value(joint, value)


def format_joint_delta(joint: Joint, value: float) -> str:
    if joint.is_gripper:
        motor_rad = logical_gripper_to_motor(value)
        motor_deg = math.degrees(motor_rad)
        return (
            f"logical={value:+.3f} "
            f"(motor={motor_rad:+.4f} rad / {motor_deg:+.1f} deg)"
        )
    return f"{value:+.3f} rad / {math.degrees(value):+.1f} deg"


def announce_movement(joint: Joint, hold_s: float) -> None:
    print(
        f"  >>> 現在、{joint.label} ({joint.description})が移動位置です。"
        f"{hold_s:.1f}秒間保持します。"
    )


def prompt_physical_direction(joint: Joint) -> bool | None:
    while True:
        answer = input(
            f"  {joint.label} ({joint.description})の物理回転方向は正しいですか？ "
            "y=正しい / n=逆 / u=不明: "
        ).strip().lower()
        if answer in {"y", "n", "u"}:
            return {"y": True, "n": False, "u": None}[answer]
        print("  y、n、uのいずれかを入力してください。")


async def run(args: argparse.Namespace) -> int:
    print(
        """
=== iLoHA safe joint test ===
・非常停止できる状態を維持してください。
・アームを人が支持し、指や物を可動範囲から離してください。
・通常の起動時原点移動は行いません。
・試験する1軸だけを低速・小角度で動かし、元の位置へ戻してトルクOFFにします。
・異常時は非常停止、Ctrl-Cの順で停止してください。
""".strip()
    )
    if input("上記を確認したら YES と入力してください: ").strip() != "YES":
        print("中止しました")
        return 1

    ports = await resolve_robstride_ports(
        left=args.left_robstride_port,
        right=args.right_robstride_port,
    )
    print(f"RobStride ports: left={ports.left}, right={ports.right}")

    arms = {
        "left": SafeArmConnection(
            "left",
            ports.left,
            args.left_dynamixel_port,
            LEFT_ROBSTRIDE_CONSTANTS,
            LEFT_DYNAMIXEL_CONSTANTS,
            args,
        ),
        "right": SafeArmConnection(
            "right",
            ports.right,
            args.right_dynamixel_port,
            RIGHT_ROBSTRIDE_CONSTANTS,
            RIGHT_DYNAMIXEL_CONSTANTS,
            args,
        ),
    }
    results = []

    try:
        # Dynamixel SDKの同期I/Oで、他方のRobStrideタイムアウト判定を
        # ブロックしないよう左右を順番に接続する。
        for arm in arms.values():
            await arm.connect_safely()
        print("\n全軸トルクOFFを確認しました。")
        response = input(
            "ロボットを現在姿勢のまま支持してEnterを押してください。"
            "物理原点へ合わせる必要はありません。"
            "Enter=現在角読取りへ進む / q=動作試験をせず終了: "
        ).strip().lower()
        if response == "q":
            print("qが入力されたため、現在角読取りと関節動作試験は実行していません。")
            return 0

        left, right = await asyncio.gather(
            arms["left"].read_logical_positions(),
            arms["right"].read_logical_positions(),
        )
        print_positions(left, right)
        input(
            "表示値が現在姿勢と大きく矛盾しないことを確認したらEnterを押してください。"
            "次に選択関節の動作確認へ進みます: "
        )

        for joint in args.joints:
            response = prompt_for_test(joint, args.step_rad)
            if response == "q":
                break
            if response == "s":
                results.append({"joint": joint.label, "skipped": True})
                continue

            result = await arms[joint.side].test_joint(joint, args.step_rad)
            print(
                f"  command={format_joint_delta(joint, result['commanded_delta'])}, "
                f"measured={format_joint_delta(joint, result['measured_delta'])}, "
                f"returned={format_optional_joint_value(joint, result['final'])}"
            )
            if not result["movement_detected"]:
                print("  WARNING: 指令量に対して実測移動量が小さすぎます")
            if not result["direction_matches_command"]:
                print("  WARNING: 実測角度の符号が指令と逆です")
            result["physical_direction_ok"] = prompt_physical_direction(joint)
            results.append(result)

    finally:
        print("\n全軸をトルクOFFにして切断します...")
        for side, arm in arms.items():
            try:
                await arm.close()
            except Exception as exc:
                print(f"CRITICAL: {side}の安全切断に失敗しました: {exc}")

    print("\n=== summary ===")
    for result in results:
        if result.get("skipped"):
            print(f"  {result['joint']}: SKIPPED")
            continue
        automatic_ok = result["movement_detected"] and result["direction_matches_command"]
        physical_ok = result["physical_direction_ok"]
        physical_status = (
            "UNKNOWN"
            if physical_ok is None
            else "OK"
            if physical_ok
            else "CHECK"
        )
        print(
            f"  {result['joint']}: sensor={'OK' if automatic_ok else 'CHECK'}, "
            f"physical={physical_status}"
        )

    if args.report is not None:
        args.report.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report: {args.report}")
    return 0


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\n中断しました。非常停止後、全軸トルクOFFを確認してください。")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
