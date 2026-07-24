import asyncio
import contextlib
import logging
import threading
import time
from typing import Any

import numpy as np

from lerobot.robots.iloha.config_iloha import IlohaConfig

from .iloha_controller.aloha_controller import AlohaArm, AlohaController
from .iloha_controller.robstride_port_detection import resolve_robstride_ports

JOINT_NAMES = [f"joint_{i}" for i in range(14)]
STATE_POLL_FREQUENCY_HZ = 10.0
STATE_POLL_WARNING_INTERVAL_S = 10.0

logger = logging.getLogger(__name__)


class MeasuredStateError(RuntimeError):
    """Base error for an unavailable measured motor state."""


class MeasuredStateUnavailableError(MeasuredStateError):
    """Raised before the first measured motor state is available."""


class StaleMeasuredStateError(MeasuredStateError):
    """Raised when the cached measured motor state is too old."""


class Iloha:
    config_class = IlohaConfig
    name = "iloha"

    def __init__(
        self,
        config: IlohaConfig,
        debug: bool = False,
        cameras: dict | None = None,
    ):
        # super().__init__(config)
        self.config = config
        self.debug = debug
        self.aloha = None
        self.cameras = cameras if cameras is not None else {}
        # old_action を ndarray で管理（14要素：L側7要素 + R側7要素）
        self.old_action = np.zeros(14, dtype=np.float32)
        # 指数平滑化フィルタの設定
        self.filter_alpha = 0.5
        self.filtered_joint_angles = None
        self._measured_state: np.ndarray | None = (
            np.zeros(14, dtype=np.float32) if self.debug else None
        )
        self._measured_state_timestamp: float | None = time.monotonic() if self.debug else None
        self._measured_state_lock = threading.Lock()
        self._state_polling_task: asyncio.Task | None = None
        self._last_state_poll_warning_at = float("-inf")
        self._consecutive_state_poll_failures = 0

    @property
    def observation_features(self) -> dict:
        """
        観測データの構造を定義
        - 画像: 各カメラごとにLeRobotのデータセット形式で定義
        - 関節角度: 状態ベクトルとして定義
        """
        features = {}
        # カメラ画像の特徴 - データセット形式では (height, width, channels)
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (480, 640, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (480, 640, 1)
        # 関節角度の特徴 - 状態ベクトルとして定義
        for joint_name in JOINT_NAMES:
            features[joint_name] = float
        return features

    @property
    def action_features(self) -> dict:
        """
        アクションデータの構造を定義
        - 14個の関節角度目標値
        """
        features = {}
        for joint_name in JOINT_NAMES:
            features[joint_name] = float
        return features

    def get_observation(self) -> dict[str, Any]:
        """
        現在の観測データを取得
        - カメラ画像: RGBと有効な深度をH, W, C形式で取得
        - 関節角度: "state"という単一のキーで14要素のベクトルとして返す
        """
        obs = {}
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                obs[name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                obs[f"{name}_depth"] = camera.read_latest_depth()
        measured_state = self.get_measured_state()
        for i, joint_name in enumerate(JOINT_NAMES):
            obs[joint_name] = measured_state[i]
        return obs

    async def connect(self) -> None:
        if not self.debug:
            robstride_ports = await resolve_robstride_ports(
                left=self.config.left_robstride_port,
                right=self.config.right_robstride_port,
            )
            print(
                "RobStrideポート: "
                f"left={robstride_ports.left}, right={robstride_ports.right}"
            )
            self.aloha = AlohaController(
                robstride_ports.right,
                robstride_ports.left,
                self.config.right_dynamixel_port,
                self.config.left_dynamixel_port,
                robstride_current_limit=self.config.current_limit_robstride,
                right_gripper_current_ma=self.config.current_limit_gripper_R * 1000,
                left_gripper_current_ma=self.config.current_limit_gripper_L * 1000,
            )
            try:
                # AlohaControllerを非同期で初期化
                await self.aloha.__aenter__()
                measured_state = await self.refresh_measured_state()
                self.old_action = measured_state.copy()
                self.filtered_joint_angles = measured_state.copy()
                self._state_polling_task = asyncio.create_task(self._state_polling_loop())
            except Exception:
                await self.aloha.disable(return_to_initial=False)
                self.aloha = None
                raise

    async def refresh_measured_state(self) -> np.ndarray:
        """全モータから実角度を取得し、iLoHA論理関節座標でキャッシュする。"""
        if self.debug:
            measured_state = self.old_action.copy()
        else:
            if self.aloha is None:
                raise RuntimeError("Iloha is not connected")
            right_arm, left_arm = await self.aloha.get_pos()
            measured_state = np.asarray(
                [*left_arm.get_positions(), *right_arm.get_positions()],
                dtype=np.float32,
            )
            # send_actionで論理グリッパー値をモータ角へ変換しているため、その逆変換を行う。
            measured_state[6] = -measured_state[6] * 3 / np.pi
            measured_state[13] = -measured_state[13] * 3 / np.pi

        with self._measured_state_lock:
            self._measured_state = measured_state.copy()
            self._measured_state_timestamp = time.monotonic()
        return measured_state

    def get_measured_state(self, max_age_s: float | None = None) -> np.ndarray:
        """最新の実測関節角度を取得し、必要なら鮮度も検証する。"""
        with self._measured_state_lock:
            measured_state = None if self._measured_state is None else self._measured_state.copy()
            timestamp = self._measured_state_timestamp

        if measured_state is None or timestamp is None:
            raise MeasuredStateUnavailableError("Measured motor state is not available yet")
        age_s = time.monotonic() - timestamp
        if max_age_s is not None and age_s > max_age_s:
            raise StaleMeasuredStateError(f"Measured motor state is stale: age={age_s * 1000:.1f} ms")
        return measured_state

    async def _state_polling_loop(self) -> None:
        period_s = 1.0 / STATE_POLL_FREQUENCY_HZ
        while True:
            started_at = time.monotonic()
            try:
                await self.refresh_measured_state()
                if self._consecutive_state_poll_failures:
                    logger.info(
                        "Motor state polling recovered after %d consecutive failure(s).",
                        self._consecutive_state_poll_failures,
                    )
                    self._consecutive_state_poll_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_state_poll_failures += 1
                now = time.monotonic()
                if now - self._last_state_poll_warning_at >= STATE_POLL_WARNING_INTERVAL_S:
                    logger.warning(
                        "Motor state polling failed (%d consecutive); keeping the last measured state: %s",
                        self._consecutive_state_poll_failures,
                        exc,
                    )
                    self._last_state_poll_warning_at = now

            sleep_s = period_s - (time.monotonic() - started_at)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_L = AlohaArm(
            motor1=action.get(JOINT_NAMES[0], self.old_action_L.motor1),
            motor2=action.get(JOINT_NAMES[1], self.old_action_L.motor2),
            motor3=action.get(JOINT_NAMES[2], self.old_action_L.motor3),
            motor4=action.get(JOINT_NAMES[3], self.old_action_L.motor4),
            motor5=action.get(JOINT_NAMES[4], self.old_action_L.motor5),
            motor6=action.get(JOINT_NAMES[5], self.old_action_L.motor6),
            motor7=-action.get(JOINT_NAMES[6], self.old_action_L.motor7)*np.pi/3,
        )
        action_R = AlohaArm(
            motor1=action.get(JOINT_NAMES[7], self.old_action_R.motor1),
            motor2=action.get(JOINT_NAMES[8], self.old_action_R.motor2),
            motor3=action.get(JOINT_NAMES[9], self.old_action_R.motor3),
            motor4=action.get(JOINT_NAMES[10], self.old_action_R.motor4),
            motor5=action.get(JOINT_NAMES[11], self.old_action_R.motor5),
            motor6=action.get(JOINT_NAMES[12], self.old_action_R.motor6),
            motor7=-action.get(JOINT_NAMES[13], self.old_action_R.motor7)*np.pi/3,
        )
        asyncio.run(self.aloha.update_pos(action_R, action_L)) # ループが既にある場合はバグるかも
        return action

    async def async_send_action(self, action: np.ndarray, use_relative=False, use_filter=True, use_unwrap=True) -> dict[str, float]:
        # 1. unwrap 処理（ndarray で実行）
        if use_unwrap:
            unwrapped_action = self._unwrap_angle_target(action, self.old_action)
        else:
            unwrapped_action = action.copy()
        
        # 2. use_relative=True の場合にのみ、変化量を制限する（ndarray で実行）
        if use_relative:
            limited_action = self._limit_relative_target(unwrapped_action, self.old_action)
        else:
            limited_action = unwrapped_action
        
        # 3. use_filter=True の場合、指数平滑化フィルタを適用（ndarray で実行）
        if use_filter:
            if self.filtered_joint_angles is None:
                # 初回はそのまま使用
                self.filtered_joint_angles = limited_action.copy()
            else:
                # フィルタリング
                self.filtered_joint_angles = (
                    self.filter_alpha * limited_action +
                    (1 - self.filter_alpha) * self.filtered_joint_angles
                )
            final_action = self.filtered_joint_angles
        else:
            final_action = limited_action
            self.filtered_joint_angles = final_action.copy()
        
        # 4. ndarray を AlohaArm に変換（update_pos 用）
        final_action_L = AlohaArm(
            motor1=float(final_action[0]),
            motor2=float(final_action[1]),
            motor3=float(final_action[2]),
            motor4=float(final_action[3]),
            motor5=float(final_action[4]),
            motor6=float(final_action[5]),
            motor7=-float(final_action[6])*np.pi/3,
        )
        final_action_R = AlohaArm(
            motor1=float(final_action[7]),
            motor2=float(final_action[8]),
            motor3=float(final_action[9]),
            motor4=float(final_action[10]),
            motor5=float(final_action[11]),
            motor6=float(final_action[12]),
            motor7=-float(final_action[13])*np.pi/3,
        )
        # 5. 最終的なアクションをロボットに送信し、状態を更新する
        if not self.debug:
            await self.aloha.update_pos(final_action_R, final_action_L)
        else:
            # print(f"L: {final_action_L.motor4*180/np.pi:.3f} R: {final_action_R.motor4*180/np.pi:.3f}")
            pass
        self.old_action = final_action.copy()
        return action

    def _unwrap_angle_target(self, current: np.ndarray, old: np.ndarray) -> np.ndarray:
        """
        新しい目標角度(current)を、古い角度(old)に最も近い連続的な値に変換（アンラップ）します。
        ndarray版：14要素の配列を処理
        """
        unwrapped = current.copy()
        # グリッパー以外の関節（インデックス0-5と7-12）に対してアンラップを適用
        # グリッパー（インデックス6と13）は除外
        for i in range(14):
            if i == 6 or i == 13:  # グリッパーはスキップ
                continue
            delta = current[i] - old[i]
            if delta > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif delta < -np.pi:
                unwrapped[i] += 2 * np.pi
        return unwrapped

    def _limit_relative_target(self, current: np.ndarray, old: np.ndarray) -> np.ndarray:
        """
        連続的な角度(current)と古い角度(old)の差分を計算し、最大変化量で制限します。
        この関数は、入力角度が既にアンラップされていることを前提とします。
        ndarray版：14要素の配列を処理
        """
        limited = current.copy()
        # 最大変化量のリスト（L側7要素 + R側7要素）
        max_deltas = [
            self.config.max_relative_target_1,  # L motor1
            self.config.max_relative_target_2,  # L motor2
            self.config.max_relative_target_3,  # L motor3
            self.config.max_relative_target_4,  # L motor4
            self.config.max_relative_target_5,  # L motor5
            self.config.max_relative_target_6,  # L motor6
            0,  # L gripper（制限なし）
            self.config.max_relative_target_1,  # R motor1
            self.config.max_relative_target_2,  # R motor2
            self.config.max_relative_target_3,  # R motor3
            self.config.max_relative_target_4,  # R motor4
            self.config.max_relative_target_5,  # R motor5
            self.config.max_relative_target_6,  # R motor6
            0,  # R gripper（制限なし）
        ]
        
        for i in range(14):
            if i == 6 or i == 13:  # グリッパーはスキップ
                continue
            delta = current[i] - old[i]
            if abs(delta) > max_deltas[i]:
                delta = max_deltas[i] * np.sign(delta)
            limited[i] = old[i] + delta
        
        return limited


    async def disconnect(self):
        if self._state_polling_task is not None:
            self._state_polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._state_polling_task
            self._state_polling_task = None
        if self.aloha is not None:
            await self.aloha.disable()
            self.aloha = None
