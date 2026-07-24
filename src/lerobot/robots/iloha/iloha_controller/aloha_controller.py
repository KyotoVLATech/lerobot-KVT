import asyncio
from typing import Any

from .aloha_arm_controller import AlohaArm, AlohaArmController


class AlohaController:
    """
    ALOHA双腕ロボットの制御クラス
    左右のAlohaArmControllerを統合して制御
    """

    def __init__(
        self,
        right_robstride_port: str,
        left_robstride_port: str,
        right_dynamixel_port: str,
        left_dynamixel_port: str,
        robstride_current_limit: float | dict[int, float] = 2.0,
        right_gripper_current_ma: float = 300.0,
        left_gripper_current_ma: float = 300.0,
        right_robstride_constants: list[Any] | None = None,
        right_dynamixel_constants: list[Any] | None = None,
        left_robstride_constants: list[Any] | None = None,
        left_dynamixel_constants: list[Any] | None = None,
    ):
        """
        ALOHAコントローラーを初期化

        Args:
            right_robstride_port: 右アームRobStrideのポート名
            left_robstride_port: 左アームRobStrideのポート名
            right_dynamixel_port: 右アームDynamixelのポート名
            left_dynamixel_port: 左アームDynamixelのポート名
            right_robstride_constants: 右アーム RobStride設定リスト
            right_dynamixel_constants: 右アーム Dynamixel設定リスト
            left_robstride_constants: 左アーム RobStride設定リスト
            left_dynamixel_constants: 左アーム Dynamixel設定リスト
        """
        # デフォルト設定の読み込み
        if right_robstride_constants is None or right_dynamixel_constants is None:
            from .right_settings import (
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            )

            right_robstride_constants = [
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            ]
            right_dynamixel_constants = [
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
            ]

        if left_robstride_constants is None or left_dynamixel_constants is None:
            from .left_settings import (
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            )

            left_robstride_constants = [
                Robstride01Constants,
                Robstride02Constants,
                Robstride03Constants,
            ]
            left_dynamixel_constants = [
                Dynamixel01Constants,
                Dynamixel02Constants,
                Dynamixel03Constants,
                Dynamixel04Constants,
            ]

        # アームコントローラーのインスタンス
        self.right_arm_controller: AlohaArmController | None = None
        self.left_arm_controller: AlohaArmController | None = None

        # 初期化パラメータを保存
        self.right_params = {
            "robstride_port": right_robstride_port,
            "dynamixel_port": right_dynamixel_port,
            "robstride_constants": right_robstride_constants,
            "dynamixel_constants": right_dynamixel_constants,
            "robstride_current_limit": robstride_current_limit,
            "gripper_current_ma": right_gripper_current_ma,
        }
        self.left_params = {
            "robstride_port": left_robstride_port,
            "dynamixel_port": left_dynamixel_port,
            "robstride_constants": left_robstride_constants,
            "dynamixel_constants": left_dynamixel_constants,
            "robstride_current_limit": robstride_current_limit,
            "gripper_current_ma": left_gripper_current_ma,
        }

    async def _initialize_controllers(self) -> None:
        """両アームコントローラーを初期化（並列実行）"""
        try:
            print("🤖 ALOHA双腕コントローラー初期化中...")

            # 右アームと左アームのコントローラーを作成
            self.right_arm_controller = AlohaArmController(**self.right_params)
            self.left_arm_controller = AlohaArmController(**self.left_params)

            # 片腕の設定失敗中にもう片腕だけが動き始めないよう、
            # 接続・設定と原点移動を明確に分ける。
            print("  両アーム接続・設定中（この段階では移動しません）...")
            await self.right_arm_controller._initialize_controllers(
                move_to_initial=False
            )
            await self.left_arm_controller._initialize_controllers(
                move_to_initial=False
            )

            print("  両アーム設定完了。原点移動を開始します...")
            await self.right_arm_controller._move_to_initial_position()
            await self.left_arm_controller._move_to_initial_position()

            print("✅ ALOHA双腕コントローラー初期化完了!")

        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            await self.disable(return_to_initial=False)
            raise

    async def update_pos(self, right_arm: AlohaArm, left_arm: AlohaArm) -> None:
        """
        両アームの位置を一括更新（並列実行）

        Args:
            right_arm: 右アームの目標位置
            left_arm: 左アームの目標位置
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        # 両アームを並列更新
        await asyncio.gather(
            self.right_arm_controller.update_pos(right_arm),
            self.left_arm_controller.update_pos(left_arm),
        )

    async def get_pos(self) -> tuple[AlohaArm, AlohaArm]:
        """左右両腕の実測関節角度を並列取得する。"""
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        right_arm, left_arm = await asyncio.gather(
            self.right_arm_controller.get_pos(),
            self.left_arm_controller.get_pos(),
        )
        return right_arm, left_arm

    async def update_motor_pos(
        self, arm: str, motor_num: int, target_pos: float
    ) -> None:
        """
        特定のアームの特定のモーターの位置を更新

        Args:
            arm: アーム指定 ("right" または "left")
            motor_num: モーター番号 (1-7)
            target_pos: 目標位置 (radian)
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        if arm == "right":
            await self.right_arm_controller.update_motor_pos(motor_num, target_pos)
        elif arm == "left":
            await self.left_arm_controller.update_motor_pos(motor_num, target_pos)
        else:
            raise ValueError("armは'right'または'left'を指定してください")

    async def set_gripper_current(self, arm: str, current_ma: float) -> None:
        """
        指定したアームのグリッパーの電流を設定

        Args:
            arm: アーム指定 ("right" または "left")
            current_ma: 目標電流 (mA)
        """
        if self.right_arm_controller is None or self.left_arm_controller is None:
            raise RuntimeError("コントローラーが初期化されていません")

        if arm == "right":
            await self.right_arm_controller.set_gripper_current(current_ma)
        elif arm == "left":
            await self.left_arm_controller.set_gripper_current(current_ma)
        else:
            raise ValueError("armは'right'または'left'を指定してください")

    async def disable(self, *, return_to_initial: bool = True) -> None:
        """全モーターを初期位置に戻し、接続を切断（並列実行）"""
        print("🔄 ALOHA双腕コントローラー終了処理中...")

        # 右アームと左アームの終了処理を並列実行
        tasks = []

        if self.right_arm_controller:

            async def disable_right() -> None:
                try:
                    print("  右アーム終了処理中...")
                    await self.right_arm_controller.disable(
                        return_to_initial=return_to_initial
                    )
                except Exception as e:
                    print(f"⚠️ 右アーム終了処理エラー: {e}")
                finally:
                    self.right_arm_controller = None

            tasks.append(disable_right())

        if self.left_arm_controller:

            async def disable_left() -> None:
                try:
                    print("  左アーム終了処理中...")
                    await self.left_arm_controller.disable(
                        return_to_initial=return_to_initial
                    )
                except Exception as e:
                    print(f"⚠️ 左アーム終了処理エラー: {e}")
                finally:
                    self.left_arm_controller = None

            tasks.append(disable_left())

        if tasks:
            await asyncio.gather(*tasks)

        print("✅ ALOHA双腕コントローラー終了完了")

    async def __aenter__(self) -> "AlohaController":
        await self._initialize_controllers()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disable()
