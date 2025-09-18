# from ..robot import Robot
from lerobot.robots.my_aloha.config_my_aloha import MyAlohaConfig
# from .aloha_abc import AlohaArm, AlohaController
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from kvt_aloha_python_controller.kvt_aloha import AlohaController, AlohaArm

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

    def connect(self) -> None:
        self.aloha = AlohaController(
            self.config.right_robstride_port,
            self.config.left_robstride_port,
            self.config.right_dynamixel_port,
            self.config.left_dynamixel_port,
        )
        self.aloha.set_gripper_current("left", self.config.current_limit_gripper_R*1000)
        self.aloha.set_gripper_current("right", self.config.current_limit_gripper_L*1000)

    def send_action(self, action: dict[str, float], use_relative=True) -> dict[str, float]:
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
        self.aloha.update_pos(final_action_R, final_action_L)
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


    def disconnect(self):
        if self.aloha is not None:
            self.aloha.disable()
            self.aloha = None

if __name__ == "__main__":
    import math
    import time
    
    def test_my_aloha():
        """
        MyAloha双腕ロボットの基本制御テスト
        サンプルコード aloha_dual_arm_demo.py を参考にした実装
        """
        print("🤖 MyAloha双腕制御テスト開始")
        print("=" * 50)
        
        # --- 設定項目 ---
        # 各ポート名をご自身の環境に合わせて変更してください
        config = MyAlohaConfig(
            right_robstride_port="COM8",
            right_dynamixel_port="COM6",
            left_robstride_port="COM7",
            left_dynamixel_port="COM5",
            max_relative_target=0.02
        )
        # -----------------
        
        # MyAlohaインスタンスを作成
        robot = MyAloha(config)
        
        try:
            # ロボットに接続
            print("\n🔗 ロボットに接続中...")
            robot.connect()
            print("✅ 接続完了")
            
            print("\n🎯 双腕テストシーケンス開始")
            
            # --- テスト1: 初期位置確認 ---
            print("\n📍 テスト1: 初期位置 (全ジョイント 0.0 rad)")
            initial_action = {
                "joint_L_0": 0.0, "joint_L_1": 0.0, "joint_L_2": 0.0,
                "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0,
                "gripper_L": 0.0,
                "joint_R_0": 0.0, "joint_R_1": 0.0, "joint_R_2": 0.0,
                "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0,
                "gripper_R": 0.0
            }
            robot.send_action(initial_action)
            time.sleep(3)
            
            # グリッパーの電流を設定（AlohaControllerに直接アクセス）
            if robot.aloha and hasattr(robot.aloha, 'set_gripper_current'):
                robot.aloha.set_gripper_current("right", 500.0)  # 500mAに設定
                robot.aloha.set_gripper_current("left", 500.0)   # 500mAに設定
                time.sleep(2)
            
            # --- テスト2: 基本動作 (サンプルと同じ角度) ---
            print("\n📍 テスト2: 基本動作")
            test_action = {
                "joint_L_0": -math.pi / 6, "joint_L_1": -math.pi / 6, "joint_L_2": -math.pi / 6,
                "joint_L_3": -math.pi / 6, "joint_L_4": -math.pi / 6, "joint_L_5": -math.pi / 6,
                "gripper_L": 0.0,
                "joint_R_0": -math.pi / 6, "joint_R_1": -math.pi / 6, "joint_R_2": -math.pi / 6,
                "joint_R_3": -math.pi / 6, "joint_R_4": -math.pi / 6, "joint_R_5": -math.pi / 6,
                "gripper_R": 0.0
            }
            robot.send_action(test_action)
            time.sleep(4)
            
            # --- テスト3: 個別ジョイント制御 (サンプルと同様に7から1まで順に0に戻す) ---
            print("\n📍 テスト3: 個別ジョイント制御")
            
            # 右アーム: joint_R_5 から joint_R_0 まで順に 0.0 radに移動
            joint_names_right = [
                ("joint_R_5", "右アーム ジョイント6"),
                ("joint_R_4", "右アーム ジョイント5"), 
                ("joint_R_3", "右アーム ジョイント4"),
                ("joint_R_2", "右アーム ジョイント3"),
                ("joint_R_1", "右アーム ジョイント2"),
                ("joint_R_0", "右アーム ジョイント1")
            ]
            
            current_action = test_action.copy()
            
            for joint_key, joint_desc in joint_names_right:
                print(f"  → {joint_desc}を 0.0 radに移動")
                current_action[joint_key] = 0.0
                robot.send_action(current_action)
                time.sleep(1)
            
            # 左アーム: joint_L_5 から joint_L_0 まで順に 0.0 radに移動
            joint_names_left = [
                ("joint_L_5", "左アーム ジョイント6"),
                ("joint_L_4", "左アーム ジョイント5"),
                ("joint_L_3", "左アーム ジョイント4"),
                ("joint_L_2", "左アーム ジョイント3"),
                ("joint_L_1", "左アーム ジョイント2"),
                ("joint_L_0", "左アーム ジョイント1")
            ]
            
            for joint_key, joint_desc in joint_names_left:
                print(f"  → {joint_desc}を 0.0 radに移動")
                current_action[joint_key] = 0.0
                robot.send_action(current_action)
                time.sleep(1)
            
            # --- テスト4: グリッパー制御 ---
            print("\n📍 テスト4: グリッパー制御")
            gripper_test_action = current_action.copy()
            
            print("  → グリッパーを開く")
            gripper_test_action["gripper_L"] = 0.5
            gripper_test_action["gripper_R"] = 0.5
            robot.send_action(gripper_test_action)
            time.sleep(2)
            
            print("  → グリッパーを閉じる")
            gripper_test_action["gripper_L"] = -0.5
            gripper_test_action["gripper_R"] = -0.5
            robot.send_action(gripper_test_action)
            time.sleep(2)
            
            # 最終的にグリッパーを中立位置に戻す
            print("  → グリッパーを中立位置に戻す")
            gripper_test_action["gripper_L"] = 0.0
            gripper_test_action["gripper_R"] = 0.0
            robot.send_action(gripper_test_action)
            time.sleep(1)
            
            print("\n🎉 MyAloha双腕テストが正常に完了しました！")
            print("構成: 右アーム(1-3番RobStride, 4-7番Dynamixel)")
            print("      左アーム(1-3番RobStride, 4-7番Dynamixel)")
            print(f"最大相対目標値: {config.max_relative_target} rad")
            
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # MyAlohaクラスの disconnect メソッドを使用
            print("\n🔌 ロボットとの接続を切断中...")
            robot.disconnect()
            print("✅ 切断完了")
    
    # テスト実行
    test_my_aloha()
