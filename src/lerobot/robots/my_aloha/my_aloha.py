# from ..robot import Robot
from lerobot.robots.my_aloha.config_my_aloha import MyAlohaConfig
# from .aloha_abc import AlohaArm, AlohaController
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from kvt_aloha_python_controller.kvt_aloha.aloha_controller import AlohaController, AlohaArm

class MyAloha():
    config_class = MyAlohaConfig
    name = "my_aloha"

    def __init__(
        self,
        config: MyAlohaConfig,
    ):
        # super().__init__(config)
        self.config = config
        self.aloha = None
        # old_action を ndarray で管理（14要素：L側7要素 + R側7要素）
        self.old_action = np.zeros(14, dtype=np.float32)
        # 指数平滑化フィルタの設定
        self.filter_alpha = 0.5
        self.filtered_joint_angles = None

    async def connect(self) -> None:
        self.aloha = AlohaController(
            self.config.right_robstride_port,
            self.config.left_robstride_port,
            self.config.right_dynamixel_port,
            self.config.left_dynamixel_port,
        )
        # AlohaControllerを非同期で初期化
        await self.aloha.__aenter__()
        # グリッパー電流を設定
        await self.aloha.set_gripper_current("left", self.config.current_limit_gripper_L*1000)
        await self.aloha.set_gripper_current("right", self.config.current_limit_gripper_R*1000)

    async def send_action(self, action: np.ndarray, use_relative=False, use_filter=True, use_unwrap=True) -> dict[str, float]:
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
        
        print(f"Final Action R_3: {final_action_R.motor4*180/np.pi:.3f}")
        
        # 5. 最終的なアクションをロボットに送信し、状態を更新する
        await self.aloha.update_pos(final_action_R, final_action_L)
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
        if self.aloha is not None:
            await self.aloha.disable()
            self.aloha = None
