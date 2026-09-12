import asyncio
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import numpy as np

# Dynamixel関連のインポート
from .dynamixel.src.constants import Baudrate, ProtocolVersion
from .dynamixel.src.dynamixel import (
    Dynamixel,
    DynamixelController,
)

# RobStride関連のインポート
from .robstride.src.robstride import (
    RobStride,
    RobStrideController,
    RobStrideLimits,
)
from .robstride.src.constants import ParameterIndex


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
        robstride_constants: List[Any],
        dynamixel_constants: List[Any],
        robstride_current_limit: Union[float, dict[int, float]] = 2.0,
        robstride_vel_max: Union[float, dict[int, float]] = np.pi,
        robstride_acc_set: Union[float, dict[int, float]] = np.pi / 2,
    ):
        """
        ALOHAコントローラーを初期化

        Args:
            robstride_port: RobStrideのポート名
            dynamixel_port: Dynamixelのポート名
            robstride_current_limit: PPモードの電流制限 [A]（一律の値、またはID→値）
            robstride_vel_max: PPモードの速度制限 [rad/s]（一律の値、またはID→値）
            robstride_acc_set: PPモードの加速度制限 [rad/s^2]（一律の値、またはID→値）
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
        self.robstride_controller: Optional[RobStrideController] = None
        self.dynamixel_controller: Optional[DynamixelController] = None
        # RobStrideは2πを跨いだ位置を保持できるため、直前の回転枝を記録する。
        self._robstride_target_references: dict[int, float] = {}

        # モーター設定
        self.robstride_current_limit = robstride_current_limit
        self.robstride_vel_max = robstride_vel_max
        self.robstride_acc_set = robstride_acc_set
        self._setup_motors()

    @staticmethod
    def _per_motor(setting: Union[float, dict[int, float]], m_id: int, default: float) -> float:
        """ID→値の辞書、または全モーター共通の値から、そのIDの設定値を取り出す。"""
        if isinstance(setting, dict):
            return setting.get(m_id, default)
        return setting

    def _setup_motors(self) -> None:
        """モーター設定を初期化"""
        def get_limits(m_id):
            limit = self._per_motor(self.robstride_current_limit, m_id, 2.0)
            vel_max = self._per_motor(self.robstride_vel_max, m_id, np.pi)
            acc_set = self._per_motor(self.robstride_acc_set, m_id, np.pi / 2)
            return RobStrideLimits(
                pp_vel_max=vel_max, # PP最大速度 [rad/s]
                pp_acc_set=acc_set,  # PP加速度設定 [rad/s²]
                pp_limit_cur=limit,  # PP電流制限 [A]
                csp_limit_spd=vel_max,  # CSP速度制限 [rad/s]
                csp_limit_cur=limit,  # CSP電流制限 [A]
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

        self.dynamixel_motors = [
            Dynamixel(
                series=self.dynamixel01_constants.SERIES,
                id=self.dynamixel01_constants.ID,
                param=self.dynamixel01_constants.CONTROL_PARAMS,
            ),
            Dynamixel(
                series=self.dynamixel02_constants.SERIES,
                id=self.dynamixel02_constants.ID,
                param=self.dynamixel02_constants.CONTROL_PARAMS,
            ),
            Dynamixel(
                series=self.dynamixel03_constants.SERIES,
                id=self.dynamixel03_constants.ID,
                param=self.dynamixel03_constants.CONTROL_PARAMS,
            ),
            Dynamixel(
                series=self.dynamixel04_constants.SERIES,
                id=self.dynamixel04_constants.ID,
                param=self.dynamixel04_constants.CONTROL_PARAMS,
            ),
        ]

    async def _initialize_controllers(self) -> None:
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
            self.dynamixel_controller = DynamixelController(
                port=self.dynamixel_port,
                motors=self.dynamixel_motors,
                baudrate=Baudrate.BAUD_57600,
                protocol_version=ProtocolVersion.V2_0,
            )

            # 全コントローラーを非同期で並列に開く
            await asyncio.gather(
                self.robstride_controller.__aenter__(),
                self.dynamixel_controller.__aenter__(),
            )

            # RobStrideモーターをPPモードに設定・有効化
            await self._setup_robstride_motors("PP")

            # 初期位置に移動
            await self._move_to_initial_position()

            print("✅ ALOHA Controller初期化完了!")

        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            await self.disable()
            raise

    async def _setup_robstride_motors(self, mode) -> None:
        """RobStrideモーターをモード設定し有効化"""
        print(f"  RobStrideモーターセットアップ中 ({mode}モード)...")

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

        # 保存済みの古いLOC_REFのままenableしないよう、最初に全軸を無効化する。
        for robstride_motor in self.robstride_motors:
            if not await self.robstride_controller.disable(robstride_motor.id):
                raise RuntimeError(
                    f"RobStride Motor{robstride_motor.id} 無効化失敗"
                )

        # トルクOFFのままモードと制限値を設定する。
        for robstride_motor in self.robstride_motors:
            motor_id = robstride_motor.id
            if mode == "CSP":
                if not await self.robstride_controller.set_mode_csp(motor_id):
                    raise RuntimeError(f"RobStride Motor{motor_id} CSPモード設定失敗")
                if not await self.robstride_controller.apply_csp_limits(
                    motor_id, allow_disabled=True
                ):
                    raise RuntimeError(f"RobStride Motor{motor_id} リミット設定失敗")
            elif mode == "PP":
                if not await self.robstride_controller.set_mode_pp(motor_id):
                    raise RuntimeError(f"RobStride Motor{motor_id} PPモード設定失敗")
                if not await self.robstride_controller.apply_pp_limits(
                    motor_id, allow_disabled=True
                ):
                    raise RuntimeError(f"RobStride Motor{motor_id} リミット設定失敗")
            else:
                raise ValueError("mode must be either 'CSP' or 'PP'")

            # 設定したリミットを不揮発メモリに保存（瞬断・再起動時に16Aに戻るのを防ぐ）
            await self.robstride_controller.save_parameters(motor_id)

        # トルクON前に各軸の目標を実測現在位置へ同期する。
        for motor in self.robstride_motors:
            raw_position = await self.robstride_controller.get_parameter(
                motor.id, ParameterIndex.MECH_POS
            )
            if raw_position is None:
                raise RuntimeError(f"RobStride Motor{motor.id} 現在位置取得失敗")
            logical_position = float(raw_position) - motor.offset
            result = await self.robstride_controller.set_target_position(
                motor.id, logical_position
            )
            if result is None:
                raise RuntimeError(
                    f"RobStride Motor{motor.id} 現在位置への目標同期失敗"
                )
            self._robstride_target_references[motor.id] = logical_position
            if not await self.robstride_controller.enable(motor.id):
                raise RuntimeError(f"RobStride Motor{motor.id} 有効化失敗")

    async def _move_to_initial_position(self) -> None:
        """全モーターを初期位置(0.0 rad)に移動"""
        print("  初期位置へ移動中...")
        initial_arm = AlohaArm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        await self.update_pos(initial_arm)
        await asyncio.sleep(2.0)  # 移動完了まで待機

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
        dynamixel_positions_rad = {}
        for i, dynamixel_motor in enumerate(self.dynamixel_motors):
            motor_id = dynamixel_motor.id
            position_rad = positions[i + 3]
            dynamixel_positions_rad[motor_id] = position_rad

        # RobStrideとDynamixelの位置設定を並列実行
        await asyncio.gather(
            self._set_robstride_positions(robstride_positions),
            # Dynamixelは一括送信
            self._set_dynamixel_positions_rad(dynamixel_positions_rad),
        )

    async def _set_robstride_positions(
        self, logical_positions: dict[int, float]
    ) -> None:
        """現在の回転枝に最も近い2π等価角をRobStrideへ送る。"""
        assert self.robstride_controller is not None

        for motor_id, logical_position in logical_positions.items():
            reference = self._robstride_target_references.get(
                motor_id, logical_position
            )
            wire_position = self._nearest_equivalent_angle(
                logical_position, reference
            )
            result = await self.robstride_controller.set_target_position(
                motor_id, wire_position
            )
            if result is None:
                raise ConnectionError(
                    f"RobStride Motor{motor_id} 位置指令送信失敗"
                )
            self._robstride_target_references[motor_id] = wire_position

    @staticmethod
    def _nearest_equivalent_angle(angle: float, reference: float) -> float:
        """referenceに最も近い、angleと2π等価な角度を返す。"""
        return angle + math.tau * round((reference - angle) / math.tau)

    async def _set_dynamixel_positions_rad(
        self, positions_rad: dict[int, float]
    ) -> None:
        """Dynamixelの位置をradian値で一括設定"""
        assert self.dynamixel_controller is not None

        # radian値をpulse値に変換
        positions_pulse = {}
        for motor_id, position_rad in positions_rad.items():
            if motor_id not in self.dynamixel_controller.motors:
                continue
            motor = self.dynamixel_controller.motors[motor_id]
            pulse = self.dynamixel_controller.radian_to_pulse(
                position_rad, motor.dynamixel_params.param.PULSE_PER_REVOLUTION
            )
            positions_pulse[motor_id] = pulse

        # 一括送信
        await self.dynamixel_controller.set_goal_positions_async(positions_pulse)

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
            motor_id = self.dynamixel_motors[motor_num - 4].id
            # radian値をpulse値に変換して設定
            await self._set_dynamixel_positions_rad({motor_id: target_pos})

    async def set_gripper_current(self, current_mA: float) -> None:
        """
        グリッパー(Dynamixel 7番)の電流を設定

        Args:
            current_mA: 目標電流 (mA)
        """
        if not (
            0.0
            <= current_mA
            <= self.dynamixel_motors[3].dynamixel_params.param.MAX_CURRENT
        ):
            raise ValueError(
                f"current_mA must be between 0.0 and {self.dynamixel_motors[3].dynamixel_params.param.MAX_CURRENT} mA"
            )

        assert self.dynamixel_controller is not None

        motor_id = self.dynamixel_motors[3].id  # 7番モーター

        # mAを内部単位に変換（XM430は約2.69mA/unit）
        current_unit = int(current_mA / 2.69)

        await self.dynamixel_controller.set_goal_currents_async(
            {motor_id: current_unit}
        )

    async def disable(self) -> None:
        """全モーターを初期位置に戻し、接続を切断"""
        print("🔄 ALOHA Controller終了処理中...")

        try:
            # 初期位置に戻す
            print("  初期位置に復帰中...")
            initial_arm = AlohaArm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            await self.update_pos(initial_arm)
            await asyncio.sleep(3.0)  # 移動完了まで待機

            # RobStrideモーター無効化（並列実行）
            if self.robstride_controller:
                await asyncio.gather(
                    *[
                        self.robstride_controller.disable(motor_id)
                        for motor_id in range(1, 4)
                    ]
                )

            # Dynamixelモーター無効化（一括実行）
            if self.dynamixel_controller:
                torque_off = {motor_id: False for motor_id in range(4, 8)}
                await self.dynamixel_controller.set_torque_enable_async(torque_off)

        except Exception as e:
            print(f"⚠️ 終了処理中のエラー: {e}")

        # コントローラー切断（並列実行）
        try:
            tasks = []
            if self.robstride_controller:
                tasks.append(self.robstride_controller.__aexit__(None, None, None))
            if self.dynamixel_controller:
                tasks.append(self.dynamixel_controller.__aexit__(None, None, None))
            if tasks:
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"⚠️ コントローラー切断エラー: {e}")

        print("✅ ALOHA Arm Controller終了完了")

    async def __aenter__(self) -> "AlohaArmController":
        await self._initialize_controllers()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disable()
