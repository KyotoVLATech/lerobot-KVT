import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus

# RobStride関連のインポート
from .robstride.src.constants import ParameterIndex
from .robstride.src.robstride import (
    RobStride,
    RobStrideController,
    RobStrideLimits,
)

DYNAMIXEL_BAUDRATE = 57_600
DYNAMIXEL_PULSES_PER_REVOLUTION = 4096
DYNAMIXEL_CURRENT_MA_PER_UNIT = 2.69
DYNAMIXEL_MAX_CURRENT_MA = 1193.0
DYNAMIXEL_NUM_RETRY = 3
ROBSTRIDE_NUM_RETRY = 3
INITIAL_MOVE_MONITOR_PERIOD_S = 0.1
INITIAL_MOVE_MIN_DURATION_S = 2.0
INITIAL_MOVE_TIMEOUT_S = 8.0
INITIAL_MOVE_DIVERGENCE_RAD = 0.05
INITIAL_MOVE_POSITION_TOLERANCE_RAD = 0.05
DYNAMIXEL_INITIAL_MOVE_MIN_START_RAD = 0.1
DYNAMIXEL_INITIAL_MOVE_MIN_PROGRESS_RAD = 0.02


@dataclass
class AlohaArm:
    """ALOHAアームの各モーター位置を保持するデータクラス"""

    motor1: float  # RobStride 1番 (rad)
    motor2: float  # RobStride 2番 (rad)
    motor3: float  # RobStride 3番 (rad)
    motor4: float  # Dynamixel 4番 (rad)
    motor5: float  # Dynamixel 5番 (rad)
    motor6: float  # Dynamixel 6番 (rad)
    motor7: float  # Dynamixel 7番 (rad)

    def get_positions(self) -> list[float]:
        """全モーターの位置をリストで取得"""
        return [
            self.motor1,
            self.motor2,
            self.motor3,
            self.motor4,
            self.motor5,
            self.motor6,
            self.motor7,
        ]


class AlohaArmController:
    """
    ALOHA単腕ロボットの制御クラス
    DynamixelとRobStrideモーターを組み合わせて制御
    """

    def __init__(
        self,
        robstride_port: str,
        dynamixel_port: str,
        robstride_constants: list[Any],
        dynamixel_constants: list[Any],
        robstride_current_limit: float | dict[int, float] = 2.0,
        gripper_current_ma: float = 300.0,
    ):
        """
        ALOHAコントローラーを初期化

        Args:
            robstride_port: RobStrideのポート名
            dynamixel_port: Dynamixelのポート名
        """
        self.robstride_port = robstride_port
        self.dynamixel_port = dynamixel_port
        (
            self.robstride01_constants,
            self.robstride02_constants,
            self.robstride03_constants,
        ) = robstride_constants

        (
            self.dynamixel01_constants,
            self.dynamixel02_constants,
            self.dynamixel03_constants,
            self.dynamixel04_constants,
        ) = dynamixel_constants

        assert self.robstride01_constants is not None
        assert self.robstride02_constants is not None
        assert self.robstride03_constants is not None
        assert self.dynamixel01_constants is not None
        assert self.dynamixel02_constants is not None
        assert self.dynamixel03_constants is not None
        assert self.dynamixel04_constants is not None

        # コントローラーのインスタンス
        self.robstride_controller: RobStrideController | None = None
        self.dynamixel_controller: DynamixelMotorsBus | None = None
        self._dynamixel_lock = asyncio.Lock()
        # RobStrideのMECH_POS/LOC_REFは2πを跨いだ値を返せるため、
        # 各軸について直前に指令した回転枝を保持する。
        self._robstride_target_references: dict[int, float] = {}

        # モーター設定
        self.robstride_current_limit = robstride_current_limit
        if not 0.0 <= gripper_current_ma <= DYNAMIXEL_MAX_CURRENT_MA:
            raise ValueError(
                f"gripper_current_ma must be between 0.0 and {DYNAMIXEL_MAX_CURRENT_MA}"
            )
        self.gripper_current_ma = gripper_current_ma
        self._setup_motors()

    def _setup_motors(self) -> None:
        """モーター設定を初期化"""
        def get_limits(m_id):
            limit = self.robstride_current_limit.get(m_id, 2.0) if isinstance(self.robstride_current_limit, dict) else self.robstride_current_limit
            return RobStrideLimits(
                pp_vel_max=np.pi,
                pp_acc_set=np.pi / 2,
                pp_limit_cur=limit,
            )

        self.robstride_motors = [
            RobStride(
                id=self.robstride01_constants.ID,
                offset=self.robstride01_constants.DEFAULT_OFFSET,
                limits=get_limits(self.robstride01_constants.ID),
            ),
            RobStride(
                id=self.robstride02_constants.ID,
                offset=self.robstride02_constants.DEFAULT_OFFSET,
                limits=get_limits(self.robstride02_constants.ID),
            ),
            RobStride(
                id=self.robstride03_constants.ID,
                offset=self.robstride03_constants.DEFAULT_OFFSET,
                limits=get_limits(self.robstride03_constants.ID),
            ),
        ]

        dynamixel_constants = [
            self.dynamixel01_constants,
            self.dynamixel02_constants,
            self.dynamixel03_constants,
            self.dynamixel04_constants,
        ]
        self.dynamixel_motors = {
            f"motor{index}": Motor(
                id=constants.ID,
                model=constants.MODEL,
                norm_mode=MotorNormMode.DEGREES,
            )
            for index, constants in enumerate(dynamixel_constants, start=4)
        }
        self.dynamixel_offsets = {
            f"motor{index}": constants.OFFSET
            for index, constants in enumerate(dynamixel_constants, start=4)
        }
        self.dynamixel_operating_modes = {
            f"motor{index}": constants.OPERATING_MODE
            for index, constants in enumerate(dynamixel_constants, start=4)
        }

    async def _initialize_controllers(
        self,
        *,
        move_to_initial: bool = True,
    ) -> None:
        """全コントローラーを初期化し、モーターを有効化"""
        try:
            print("🔧 ALOHA Controller初期化中...")

            # RobStrideコントローラー初期化
            print("  RobStrideコントローラー接続中...")
            self.robstride_controller = RobStrideController(
                port=self.robstride_port, motors=self.robstride_motors
            )

            # Dynamixelコントローラー初期化
            print("  Dynamixelコントローラー接続中...")
            self.dynamixel_controller = DynamixelMotorsBus(
                port=self.dynamixel_port,
                motors=self.dynamixel_motors,
            )

            await self.robstride_controller.__aenter__()
            await self._connect_dynamixel()

            await self._setup_robstride_motors()

            if move_to_initial:
                await self._move_to_initial_position()

            print("✅ ALOHA Controller初期化完了!")

        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            await self.disable(return_to_initial=False)
            raise

    async def _connect_dynamixel(self) -> None:
        """LeRobot標準MotorsBusでDynamixelを接続・設定する。"""
        assert self.dynamixel_controller is not None

        async with self._dynamixel_lock:
            bus = self.dynamixel_controller
            await asyncio.to_thread(bus.connect, False)
            await asyncio.to_thread(bus.set_baudrate, DYNAMIXEL_BAUDRATE)

            missing_or_wrong = []
            for motor_name, motor in bus.motors.items():
                model_number = await asyncio.to_thread(
                    bus.ping,
                    motor_name,
                    num_retry=DYNAMIXEL_NUM_RETRY,
                )
                expected_model_number = bus.model_number_table[motor.model]
                if model_number != expected_model_number:
                    missing_or_wrong.append(
                        f"{motor_name}(ID={motor.id}, expected={expected_model_number}, found={model_number})"
                    )
            if missing_or_wrong:
                raise RuntimeError(f"Dynamixel接続確認に失敗しました: {missing_or_wrong}")

            # 個別writeは各モーターのStatus Packetを待つため、このUSBバスでは
            # 応答が混ざることがある。設定は応答不要のSync Writeで一括送信する。
            await asyncio.to_thread(
                bus.sync_write,
                "Torque_Enable",
                0,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            await asyncio.to_thread(
                bus.sync_write,
                "Operating_Mode",
                {
                    motor_name: operating_mode.value
                    for motor_name, operating_mode in self.dynamixel_operating_modes.items()
                },
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            actual_operating_modes = await asyncio.to_thread(
                self._read_dynamixel_registers,
                "Operating_Mode",
            )
            wrong_modes = {
                motor_name: (actual_operating_modes[motor_name], expected.value)
                for motor_name, expected in self.dynamixel_operating_modes.items()
                if actual_operating_modes[motor_name] != expected.value
            }
            if wrong_modes:
                raise RuntimeError(
                    f"Dynamixel Operating_Mode設定確認に失敗しました: {wrong_modes}"
                )

            # トルクON時に保存済みの古い目標へ動かないよう、現在位置を目標へ同期する。
            present_positions = await asyncio.to_thread(
                self._read_dynamixel_registers,
                "Present_Position",
            )
            await asyncio.to_thread(
                bus.sync_write,
                "Goal_Position",
                present_positions,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            await asyncio.to_thread(
                bus.sync_write,
                "Goal_Current",
                {
                    "motor7": int(
                        self.gripper_current_ma / DYNAMIXEL_CURRENT_MA_PER_UNIT
                    )
                },
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )

            await asyncio.to_thread(
                bus.sync_write,
                "Torque_Enable",
                1,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )
            torque_enabled = await asyncio.to_thread(
                self._read_dynamixel_registers,
                "Torque_Enable",
            )
            disabled_motors = [
                motor_name
                for motor_name, enabled in torque_enabled.items()
                if enabled != 1
            ]
            if disabled_motors:
                raise RuntimeError(
                    f"DynamixelトルクON確認に失敗しました: {disabled_motors}"
                )

    def _read_dynamixel_registers(self, data_name: str) -> dict[str, int]:
        """受信バッファを掃除しながら、Dynamixelを1台ずつ確実に読む。"""
        assert self.dynamixel_controller is not None
        bus = self.dynamixel_controller
        values: dict[str, int] = {}

        for motor_name in bus.motors:
            last_error: Exception | None = None
            for _ in range(1 + DYNAMIXEL_NUM_RETRY):
                try:
                    if hasattr(bus, "port_handler"):
                        bus.port_handler.clearPort()
                    values[motor_name] = int(
                        bus.read(
                            data_name,
                            motor_name,
                            normalize=False,
                            num_retry=0,
                        )
                    )
                    last_error = None
                    break
                except (ConnectionError, RuntimeError) as error:
                    last_error = error
            if last_error is not None:
                raise ConnectionError(
                    f"Dynamixel {motor_name}の{data_name}読取りに失敗しました"
                ) from last_error

        return values

    async def _setup_robstride_motors(self) -> None:
        """現在位置を目標へ同期してからRobStrideモーターを有効化する。"""
        print("  RobStrideモーターセットアップ中 (PPモード)...")

        assert self.robstride_controller is not None

        # 1. まず全モーターの疎通確認を行う
        failed_motors = []
        statuses = []
        for robstride_motor in self.robstride_motors:
            motor_id = robstride_motor.id
            is_alive = await self.robstride_controller.ping(motor_id)
            if is_alive:
                statuses.append(f"ID{motor_id}:OK")
            else:
                statuses.append(f"ID{motor_id}:FAILED")
                failed_motors.append(motor_id)

        # 接続状況のサマリーを表示
        status_line = " | ".join(statuses)
        print(f"  📊 接続状況: [ {status_line} ]")

        if failed_motors:
            raise RuntimeError(f"接続失敗したRobStrideモーターがあります: {failed_motors}")

        # 古いLOC_REFを保持したままenableしないよう、まず全軸を無効化する。
        for robstride_motor in self.robstride_motors:
            if not await self.robstride_controller.disable(robstride_motor.id):
                raise RuntimeError(
                    f"RobStride Motor{robstride_motor.id} 無効化失敗"
                )

        # モードと制限値はトルクOFFのまま設定する。
        for robstride_motor in self.robstride_motors:
            motor_id = robstride_motor.id
            if not await self.robstride_controller.set_mode_pp(motor_id):
                raise RuntimeError(f"RobStride Motor{motor_id} PPモード設定失敗")
            if not await self.robstride_controller.apply_pp_limits(
                motor_id,
                allow_disabled=True,
            ):
                raise RuntimeError(f"RobStride Motor{motor_id} リミット設定失敗")

        # トルクON前に、各軸の目標値を実測現在位置へ一致させる。
        for motor in self.robstride_motors:
            raw_position = await self._get_robstride_position(motor)
            if raw_position is None:
                raise RuntimeError(
                    f"RobStride Motor{motor.id} 現在位置取得失敗"
                )
            logical_position = raw_position - motor.offset
            result = await self.robstride_controller.set_target_position(
                motor.id,
                logical_position,
            )
            if result is None:
                raise RuntimeError(
                    f"RobStride Motor{motor.id} 現在位置への目標同期失敗"
                )
            self._robstride_target_references[motor.id] = logical_position
            if not await self.robstride_controller.enable(motor.id):
                raise RuntimeError(f"RobStride Motor{motor.id} 有効化失敗")

    async def _move_to_initial_position(self) -> None:
        """PPモードで、現在の回転枝に最も近い原点へ移動する。"""
        print("  初期位置へ移動中 (PP最短経路)...")
        start_positions = np.asarray(
            (await self.get_pos()).get_positions(),
            dtype=np.float64,
        )
        robstride_start = ", ".join(
            f"ID{motor.id}:{position:+.4f}→0"
            for motor, position in zip(
                self.robstride_motors,
                start_positions[:3],
                strict=True,
            )
        )
        print(f"  RobStride初期実測角: {robstride_start}")
        await self.update_pos(AlohaArm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        async with self._dynamixel_lock:
            dynamixel_goals = await asyncio.to_thread(
                self._read_dynamixel_registers,
                "Goal_Position",
            )
        wrong_goals = {
            motor_name: (dynamixel_goals[motor_name], self.dynamixel_offsets[motor_name])
            for motor_name in self.dynamixel_motors
            if dynamixel_goals[motor_name] != self.dynamixel_offsets[motor_name]
        }
        if wrong_goals:
            raise RuntimeError(
                f"Dynamixel原点目標の設定確認に失敗しました: {wrong_goals}"
            )

        initial_robstride = start_positions[:3]
        monitor_steps = math.ceil(
            INITIAL_MOVE_TIMEOUT_S / INITIAL_MOVE_MONITOR_PERIOD_S
        )
        current_robstride = initial_robstride
        for step in range(1, monitor_steps + 1):
            await asyncio.sleep(INITIAL_MOVE_MONITOR_PERIOD_S)
            current_robstride = await self._get_robstride_logical_positions()
            diverging_ids = self._find_diverging_robstride_motors(
                initial_robstride,
                current_robstride,
            )
            if diverging_ids:
                raise RuntimeError(
                    "RobStride原点移動が原点から遠ざかりました。"
                    f" IDs={diverging_ids}, start={initial_robstride.tolist()}, "
                    f"current={current_robstride.tolist()}"
                )
            elapsed_s = step * INITIAL_MOVE_MONITOR_PERIOD_S
            if (
                elapsed_s >= INITIAL_MOVE_MIN_DURATION_S
                and np.all(
                    np.abs(current_robstride)
                    <= INITIAL_MOVE_POSITION_TOLERANCE_RAD
                )
            ):
                await self._verify_dynamixel_initial_move(start_positions[3:])
                return
        pending_ids = [
            motor.id
            for motor, position in zip(
                self.robstride_motors,
                current_robstride,
                strict=True,
            )
            if abs(position) > INITIAL_MOVE_POSITION_TOLERANCE_RAD
        ]
        if pending_ids:
            raise TimeoutError(
                "RobStrideが原点へ到達しませんでした。"
                f" IDs={pending_ids}, position={current_robstride.tolist()}"
            )
        await self._verify_dynamixel_initial_move(start_positions[3:])

    async def _verify_dynamixel_initial_move(
        self,
        initial_positions: np.ndarray,
    ) -> None:
        current_by_name = await self._get_dynamixel_positions_rad()
        current_positions = np.asarray(
            [
                current_by_name[motor_name]
                for motor_name in self.dynamixel_motors
            ],
            dtype=np.float64,
        )
        movement_summary = ", ".join(
            f"{motor_name}:{initial:+.4f}→{current:+.4f}"
            for motor_name, initial, current in zip(
                self.dynamixel_motors,
                initial_positions,
                current_positions,
                strict=True,
            )
        )
        print(f"  Dynamixel原点移動実測: {movement_summary}")

        stalled_motors = [
            motor_name
            for motor_name, initial, current in zip(
                self.dynamixel_motors,
                initial_positions,
                current_positions,
                strict=True,
            )
            if (
                abs(initial) >= DYNAMIXEL_INITIAL_MOVE_MIN_START_RAD
                and abs(current)
                > abs(initial) - DYNAMIXEL_INITIAL_MOVE_MIN_PROGRESS_RAD
            )
        ]
        if stalled_motors:
            raise RuntimeError(
                "Dynamixelが原点方向へ移動していません。"
                f" motors={stalled_motors}"
            )

    async def _get_robstride_logical_positions(self) -> np.ndarray:
        raw_positions = await asyncio.gather(
            *[
                self._get_robstride_position(motor)
                for motor in self.robstride_motors
            ]
        )
        if any(position is None for position in raw_positions):
            raise ConnectionError("RobStride原点移動中の実角度取得に失敗しました")
        return np.asarray(
            [
                self._wrap_angle(float(position) - motor.offset)
                for motor, position in zip(
                    self.robstride_motors,
                    raw_positions,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

    def _find_diverging_robstride_motors(
        self,
        initial_positions: np.ndarray,
        current_positions: np.ndarray,
    ) -> list[int]:
        return [
            motor.id
            for motor, initial, current in zip(
                self.robstride_motors,
                initial_positions,
                current_positions,
                strict=True,
            )
            if abs(current) > abs(initial) + INITIAL_MOVE_DIVERGENCE_RAD
        ]

    async def update_pos(self, target_arm: AlohaArm) -> None:
        """
        アームの位置を一括更新

        Args:
            target_arm: アームの目標位置
        """
        positions = target_arm.get_positions()

        assert self.robstride_controller is not None
        assert self.dynamixel_controller is not None

        # RobStride用の位置辞書を作成
        robstride_positions = {}
        for i, robstride_motor in enumerate(self.robstride_motors):
            robstride_positions[robstride_motor.id] = positions[i]

        # Dynamixel用の位置辞書を作成（radianからpulseへの変換は内部で行われる）
        dynamixel_positions_rad = {
            motor_name: positions[index]
            for index, motor_name in enumerate(self.dynamixel_motors, start=3)
        }

        # RobStrideとDynamixelの位置設定を並列実行
        await asyncio.gather(
            self._set_robstride_positions(robstride_positions),
            # Dynamixelは一括送信
            self._set_dynamixel_positions_rad(dynamixel_positions_rad),
        )

    async def _set_robstride_positions(
        self, logical_positions: dict[int, float]
    ) -> None:
        """各論理角を、現在の回転枝に最も近いLOC_REFへ変換して送信する。"""
        assert self.robstride_controller is not None

        for motor_id, logical_position in logical_positions.items():
            reference = self._robstride_target_references.get(
                motor_id,
                logical_position,
            )
            wire_position = self._nearest_equivalent_angle(
                logical_position,
                reference,
            )
            result = await self.robstride_controller.set_target_position(
                motor_id,
                wire_position,
            )
            if result is None:
                raise ConnectionError(
                    f"RobStride Motor{motor_id} 位置指令送信失敗"
                )
            self._robstride_target_references[motor_id] = wire_position

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """角度を[-π, π]へ正規化する。"""
        return math.remainder(angle, math.tau)

    @staticmethod
    def _nearest_equivalent_angle(angle: float, reference: float) -> float:
        """referenceに最も近い、angleと2π等価な角度を返す。"""
        return angle + math.tau * round((reference - angle) / math.tau)

    async def _set_dynamixel_positions_rad(
        self, positions_rad: dict[str, float]
    ) -> None:
        """Dynamixelの位置をradian値で一括設定"""
        assert self.dynamixel_controller is not None

        positions_pulse = {
            motor_name: self._radian_to_pulse(position_rad) + self.dynamixel_offsets[motor_name]
            for motor_name, position_rad in positions_rad.items()
        }

        async with self._dynamixel_lock:
            await asyncio.to_thread(
                self.dynamixel_controller.sync_write,
                "Goal_Position",
                positions_pulse,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )

    async def get_pos(self) -> AlohaArm:
        """RobStrideとDynamixelから単腕7軸の実測角度を取得する。"""
        assert self.robstride_controller is not None
        assert self.dynamixel_controller is not None

        robstride_reads = [
            self._get_robstride_position(motor)
            for motor in self.robstride_motors
        ]
        robstride_values, dynamixel_values = await asyncio.gather(
            asyncio.gather(*robstride_reads),
            self._get_dynamixel_positions_rad(),
        )

        if any(value is None for value in robstride_values):
            missing_ids = [
                motor.id for motor, value in zip(self.robstride_motors, robstride_values, strict=True)
                if value is None
            ]
            raise ConnectionError(f"RobStride実角度の取得に失敗しました: IDs={missing_ids}")

        positions = [
            self._wrap_angle(float(value) - motor.offset)
            for motor, value in zip(self.robstride_motors, robstride_values, strict=True)
        ]
        positions.extend(dynamixel_values[motor_name] for motor_name in self.dynamixel_motors)
        return AlohaArm(*positions)

    async def _get_robstride_position(self, motor: RobStride) -> float | None:
        assert self.robstride_controller is not None

        for _ in range(1 + ROBSTRIDE_NUM_RETRY):
            value = await self.robstride_controller.get_parameter(
                motor.id,
                ParameterIndex.MECH_POS,
            )
            if value is not None:
                return float(value)
        return None

    async def _get_dynamixel_positions_rad(self) -> dict[str, float]:
        assert self.dynamixel_controller is not None

        async with self._dynamixel_lock:
            raw_positions = await asyncio.to_thread(
                self._read_dynamixel_registers,
                "Present_Position",
            )

        return {
            motor_name: self._pulse_to_radian(raw_position - self.dynamixel_offsets[motor_name])
            for motor_name, raw_position in raw_positions.items()
        }

    @staticmethod
    def _radian_to_pulse(radian: float) -> int:
        return int(radian / (2 * math.pi) * DYNAMIXEL_PULSES_PER_REVOLUTION)

    @staticmethod
    def _pulse_to_radian(pulse: float) -> float:
        return float(pulse) * (2 * math.pi) / DYNAMIXEL_PULSES_PER_REVOLUTION

    async def update_motor_pos(self, motor_num: int, target_pos: float) -> None:
        """
        特定のモーターの位置を更新

        Args:
            motor_num: モーター番号 (1-7)
            target_pos: 目標位置 (radian)
        """
        if not (1 <= motor_num <= 7):
            raise ValueError("motor_num must be between 1 and 7")

        if not (-2 * math.pi <= target_pos <= 2 * math.pi):
            raise ValueError("target_pos must be between -2*PI and 2*PI")

        assert self.robstride_controller is not None
        assert self.dynamixel_controller is not None

        # RobStrideモーター (1-3番)
        if 1 <= motor_num <= 3:
            motor_id = self.robstride_motors[motor_num - 1].id
            await self._set_robstride_positions({motor_id: target_pos})
        # Dynamixelモーター (4-7番)
        elif 4 <= motor_num <= 7:
            motor_name = f"motor{motor_num}"
            await self._set_dynamixel_positions_rad({motor_name: target_pos})

    async def set_gripper_current(self, current_ma: float) -> None:
        """
        グリッパー(Dynamixel 7番)の電流を設定

        Args:
            current_ma: 目標電流 (mA)
        """
        if not (
            0.0
            <= current_ma
            <= DYNAMIXEL_MAX_CURRENT_MA
        ):
            raise ValueError(
                f"current_mA must be between 0.0 and {DYNAMIXEL_MAX_CURRENT_MA} mA"
            )

        assert self.dynamixel_controller is not None

        # mAを内部単位に変換（XM430は約2.69mA/unit）
        current_unit = int(current_ma / DYNAMIXEL_CURRENT_MA_PER_UNIT)

        async with self._dynamixel_lock:
            await asyncio.to_thread(
                self.dynamixel_controller.write,
                "Goal_Current",
                "motor7",
                current_unit,
                normalize=False,
                num_retry=DYNAMIXEL_NUM_RETRY,
            )

    async def disable(self, *, return_to_initial: bool = True) -> None:
        """全モーターを初期位置に戻し、接続を切断"""
        if self.robstride_controller is None and self.dynamixel_controller is None:
            return

        print("🔄 ALOHA Controller終了処理中...")

        try:
            if return_to_initial:
                print("  初期位置に復帰中...")
                initial_arm = AlohaArm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                await self.update_pos(initial_arm)
                await asyncio.sleep(3.0)  # 移動完了まで待機

            # RobStrideモーター無効化（並列実行）
            if (
                self.robstride_controller
                and self.robstride_controller.writer
                and not self.robstride_controller.writer.is_closing()
            ):
                await asyncio.gather(
                    *[
                        self.robstride_controller.disable(motor.id)
                        for motor in self.robstride_motors
                    ]
                )

            # Dynamixelモーター無効化（一括実行）
            if (
                self.dynamixel_controller
                and self.dynamixel_controller.is_connected
            ):
                async with self._dynamixel_lock:
                    await asyncio.to_thread(
                        self.dynamixel_controller.sync_write,
                        "Torque_Enable",
                        0,
                        normalize=False,
                        num_retry=DYNAMIXEL_NUM_RETRY,
                    )

        except Exception as e:
            print(f"⚠️ 終了処理中のエラー: {e}")

        # コントローラー切断（並列実行）
        try:
            tasks = []
            if self.robstride_controller:
                tasks.append(self.robstride_controller.disconnect())
            if (
                self.dynamixel_controller
                and self.dynamixel_controller.is_connected
            ):
                tasks.append(asyncio.to_thread(self.dynamixel_controller.disconnect, False))
            if tasks:
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"⚠️ コントローラー切断エラー: {e}")

        self.robstride_controller = None
        self.dynamixel_controller = None

        print("✅ ALOHA Arm Controller終了完了")

    async def __aenter__(self) -> "AlohaArmController":
        await self._initialize_controllers()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disable()
