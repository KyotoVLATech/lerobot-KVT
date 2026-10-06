#!/usr/bin/env python3
"""Run the normal whole-robot initialization/homing, without Policy or Learner.

This MOVES all joints. Requires an operator ready to cut power.
On an error, cut power: motor stop cannot be guaranteed after communication loss.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iloha_eval import reset_robot_to_home
from lerobot.robots.iloha import Iloha, IlohaConfig


async def main():
    robot = Iloha(IlohaConfig(
        left_robstride_port="auto",
        right_robstride_port="auto",
        left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
        right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
        current_limit_robstride={1: 4.0, 2: 16.0, 3: 4.0, 4: 4.0, 5: 16.0, 6: 4.0},
        current_limit_gripper_R=0.3,
        current_limit_gripper_L=0.3,
    ), debug=False)
    print("全腕の通常初期化・原点復帰を実行します。右ベース180度／左ベース270度。", flush=True)
    completed = False
    try:
        await robot.connect()  # normal initialization itself includes homing
        await reset_robot_to_home(robot)
        measured = await robot.async_read_joint_state()
        print(f"原点復帰後の論理角度 [rad]: {measured.tolist()}", flush=True)
        print("初期位置復帰の指令・実測取得が完了しました。実際の姿勢を確認してください。", flush=True)
        completed = True
    finally:
        if completed:
            await robot.disconnect()
            print("終了処理完了。", flush=True)
        elif robot.aloha is not None:
            # Do NOT invoke disconnect/disable's automatic homing on failure.
            # Stop commands only; never issue another position target here.
            print("異常停止。位置指令は追加しません。非常停止／電源OFFを準備してください。", flush=True)
            for arm in (robot.aloha.right_arm_controller, robot.aloha.left_arm_controller):
                if arm is None:
                    continue
                if arm.robstride_controller is not None:
                    controller = arm.robstride_controller
                    for motor_id in controller.motors:
                        try:
                            if not await controller.disable(motor_id):
                                print(f"ID {motor_id}停止確認失敗。電源を切ってください。", flush=True)
                        except Exception as exc:
                            print(f"ID {motor_id}停止失敗: {exc}", flush=True)
                    await controller.disconnect()
                if arm.dynamixel_controller is not None:
                    controller = arm.dynamixel_controller
                    try:
                        if not await controller.set_torque_enable_async({mid: False for mid in controller.motors}):
                            print("Dynamixel停止確認失敗。電源を切ってください。", flush=True)
                    finally:
                        await controller.disconnect_async()


if __name__ == "__main__":
    asyncio.run(main())
