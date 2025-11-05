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
        self.old_action_L = AlohaArm(0, 0, 0, 0, 0, 0, 0)
        self.old_action_R = AlohaArm(0, 0, 0, 0, 0, 0, 0)

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
        await self.aloha.set_gripper_current("left", self.config.current_limit_gripper_R*1000)
        await self.aloha.set_gripper_current("right", self.config.current_limit_gripper_L*1000)

    async def send_action(self, action: dict[str, float], use_relative=True) -> dict[str, float]:
        action_L = AlohaArm(
            motor1=action.get("joint_L_0", self.old_action_L.motor1),
            motor2=action.get("joint_L_1", self.old_action_L.motor2),
            motor3=action.get("joint_L_2", self.old_action_L.motor3),
            motor4=action.get("joint_L_3", self.old_action_L.motor4),
            motor5=action.get("joint_L_4", self.old_action_L.motor5),
            motor6=action.get("joint_L_5", self.old_action_L.motor6),
            motor7=-action.get("gripper_L", self.old_action_L.motor7)*np.pi/3,
        )
        action_R = AlohaArm(
            motor1=action.get("joint_R_0", self.old_action_R.motor1),
            motor2=action.get("joint_R_1", self.old_action_R.motor2),
            motor3=action.get("joint_R_2", self.old_action_R.motor3),
            motor4=action.get("joint_R_3", self.old_action_R.motor4),
            motor5=action.get("joint_R_4", self.old_action_R.motor5),
            motor6=action.get("joint_R_5", self.old_action_R.motor6),
            motor7=-action.get("gripper_R", self.old_action_R.motor7)*np.pi/3,
        )
        unwrapped_L = self._unwrap_angle_target(action_L, self.old_action_L)
        unwrapped_R = self._unwrap_angle_target(action_R, self.old_action_R)
        # 最終的にモーターへ送るアクションを格納する変数
        final_action_L = unwrapped_L
        final_action_R = unwrapped_R
        # 2. use_relative=True の場合にのみ、変化量を制限する
        if use_relative:
            final_action_L = self._limit_relative_target(unwrapped_L, self.old_action_L)
            final_action_R = self._limit_relative_target(unwrapped_R, self.old_action_R)
        # 3. 最終的なアクションをロボットに送信し、状態を更新する
        await self.aloha.update_pos(final_action_R, final_action_L)
        self.old_action_L = final_action_L
        self.old_action_R = final_action_R
        return action

    def _unwrap_angle_target(self, current: AlohaArm, old: AlohaArm) -> AlohaArm:
        """
        新しい目標角度(current)を、古い角度(old)に最も近い連続的な値に変換（アンラップ）します。
        """
        def _unwrap(new, old):
            # 差分がπを超えている場合、2πを足し引きして最短経路の角度表現にする
            delta = new - old
            if delta > np.pi:
                new -= 2 * np.pi
            elif delta < -np.pi:
                new += 2 * np.pi
            return new
        return AlohaArm(
            motor1=_unwrap(current.motor1, old.motor1),
            motor2=_unwrap(current.motor2, old.motor2),
            motor3=_unwrap(current.motor3, old.motor3),
            motor4=_unwrap(current.motor4, old.motor4),
            motor5=_unwrap(current.motor5, old.motor5),
            motor6=_unwrap(current.motor6, old.motor6),
            motor7=current.motor7,
        )

    def _limit_relative_target(self, current: AlohaArm, old: AlohaArm) -> AlohaArm:
        """
        連続的な角度(current)と古い角度(old)の差分を計算し、最大変化量で制限します。
        この関数は、入力角度が既にアンラップされていることを前提とします。
        """
        def _limit(new, old, delta_max):
            delta = new - old
            if abs(delta) > delta_max:
                delta = delta_max * np.sign(delta)
            return old + delta
        return AlohaArm(
            motor1=_limit(current.motor1, old.motor1, self.config.max_relative_target_1),
            motor2=_limit(current.motor2, old.motor2, self.config.max_relative_target_2),
            motor3=_limit(current.motor3, old.motor3, self.config.max_relative_target_3),
            motor4=_limit(current.motor4, old.motor4, self.config.max_relative_target_4),
            motor5=_limit(current.motor5, old.motor5, self.config.max_relative_target_5),
            motor6=_limit(current.motor6, old.motor6, self.config.max_relative_target_6),
            motor7=current.motor7,
        )


    async def disconnect(self):
        if self.aloha is not None:
            await self.aloha.disable()
            self.aloha = None
