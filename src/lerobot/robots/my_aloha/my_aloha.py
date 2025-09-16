import logging
import time
from functools import cached_property
from typing import Any
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_STATE
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.motors.robstride import RobStride
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_my_aloha import MyAlohaConfig

logger = logging.getLogger(__name__)


class MyAloha(Robot):
    config_class = MyAlohaConfig
    name = "my_aloha"

    def __init__(
        self,
        config: MyAlohaConfig,
    ):
        super().__init__(config)
        self.config = config
        self.dbus_r = DynamixelMotorsBus(
            port=self.config.u2d2_port1,
            motors={
                "waist_R": Motor(1, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_angle_R": Motor(7, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_rotate_R": Motor(8, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "gripper_R": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_100),
            },
        )
        self.dbus_l = DynamixelMotorsBus(
            port=self.config.u2d2_port2,
            motors={
                "waist_L": Motor(1, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_angle_L": Motor(7, "xm540-w270", MotorNormMode.RANGE_M100_100),
                "wrist_rotate_L": Motor(8, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "gripper_L": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_100),
            },
        )
        self.robstride = {
            "shoulder_R": RobStride(port=self.config.can_port1, motor_id=1),
            "elbow_R": RobStride(port=self.config.can_port1, motor_id=5),
            "forearm_roll_R": RobStride(port=self.config.can_port1, motor_id=6),
            "shoulder_L": RobStride(port=self.config.can_port2, motor_id=2),
            "elbow_L": RobStride(port=self.config.can_port2, motor_id=5),
            "forearm_roll_L": RobStride(port=self.config.can_port2, motor_id=6),
        }
        # self.cameras = make_cameras_from_configs(config.cameras)
        self.is_connected = False

    def connect(self) -> None:
        # Dynamixelバスを接続
        self.dbus_r.connect()
        self.dbus_l.connect()
        
        # RobStrideモーターを接続・有効化
        for motor_name, motor in self.robstride.items():
            try:
                if motor.connect():
                    motor.enable()
                    logger.info(f"RobStrideモーター {motor_name} 接続・有効化完了")
                else:
                    logger.error(f"RobStrideモーター {motor_name} の接続に失敗")
            except Exception as e:
                logger.error(f"RobStrideモーター {motor_name} の接続エラー: {e}")
        
        self.configure()
        self.is_connected = True

    def configure(self) -> None:
        with self.dbus_r.torque_disabled(), self.dbus_l.torque_disabled():
            # Dynamixel右腕
            self.dbus_r.configure_motors()
            self.dbus_r.write("Velocity_Limit", 131)
            for motor in self.dbus_r.motors:
                if motor != "gripper_R":
                    self.dbus_r.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
            self.dbus_r.write("Operating_Mode", "gripper_R", OperatingMode.CURRENT_POSITION.value)
            
            # Dynamixel左腕
            self.dbus_l.configure_motors()
            self.dbus_l.write("Velocity_Limit", 131)
            for motor in self.dbus_l.motors:
                if motor != "gripper_L":
                    self.dbus_l.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
            self.dbus_l.write("Operating_Mode", "gripper_L", OperatingMode.CURRENT_POSITION.value)
        
        # RobStrideモーターをPP（Position Profile）モードに設定
        for motor_name, motor in self.robstride.items():
            try:
                motor.set_mode_pp()
                motor.set_pp_velocity(2.0)  # 2.0 rad/s の速度制限
                motor.set_pp_acceleration(5.0)  # 5.0 rad/s^2 の加速度
                logger.info(f"RobStrideモーター {motor_name} をPPモードに設定")
            except Exception as e:
                logger.warning(f"RobStrideモーター {motor_name} の設定エラー: {e}")

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # .posサフィックスを削除してアクションを正規化
        normalized_action = {}
        for key, value in action.items():
            motor_name = key.replace(".pos", "") if key.endswith(".pos") else key
            normalized_action[motor_name] = value
        
        # DynamixelとRobStrideモーターに分離
        dynamixel_actions_r = {}
        dynamixel_actions_l = {}
        robstride_actions = {}
        
        for motor_name, target_pos in normalized_action.items():
            if motor_name in self.dbus_r.motors:
                dynamixel_actions_r[motor_name] = target_pos
            elif motor_name in self.dbus_l.motors:
                dynamixel_actions_l[motor_name] = target_pos
            elif motor_name in self.robstride:
                robstride_actions[motor_name] = target_pos
        
        sent_actions = {}
        
        # Dynamixel右腕の制御
        if dynamixel_actions_r:
            if self.config.max_relative_target is not None:
                present_pos_r = self.dbus_r.sync_read("Present_Position")
                goal_present_pos_r = {key: (target_pos, present_pos_r[key]) 
                                    for key, target_pos in dynamixel_actions_r.items()}
                safe_goal_pos_r = ensure_safe_goal_position(goal_present_pos_r, self.config.max_relative_target)
            else:
                safe_goal_pos_r = dynamixel_actions_r
            
            self.dbus_r.sync_write("Goal_Position", safe_goal_pos_r)
            sent_actions.update(safe_goal_pos_r)
        
        # Dynamixel左腕の制御
        if dynamixel_actions_l:
            if self.config.max_relative_target is not None:
                present_pos_l = self.dbus_l.sync_read("Present_Position")
                goal_present_pos_l = {key: (target_pos, present_pos_l[key]) 
                                    for key, target_pos in dynamixel_actions_l.items()}
                safe_goal_pos_l = ensure_safe_goal_position(goal_present_pos_l, self.config.max_relative_target)
            else:
                safe_goal_pos_l = dynamixel_actions_l
            
            self.dbus_l.sync_write("Goal_Position", safe_goal_pos_l)
            sent_actions.update(safe_goal_pos_l)
        
        # RobStrideモーターの制御
        for motor_name, target_pos in robstride_actions.items():
            try:
                motor = self.robstride[motor_name]
                motor.set_target_position(target_pos)
                sent_actions[motor_name] = target_pos
            except Exception as e:
                logger.warning(f"RobStrideモーター {motor_name} の制御に失敗: {e}")
        
        return sent_actions

    def disconnect(self):
        """全てのモーターとの接続を安全に切断します"""
        if not self.is_connected:
            return
        
        try:
            # RobStrideモーターを停止・切断
            for motor_name, motor in self.robstride.items():
                try:
                    # 目標位置を現在位置に設定して停止
                    motor.set_target_position(0.0)
                    motor.disable()
                    motor.disconnect()
                    logger.info(f"RobStrideモーター {motor_name} を切断しました")
                except Exception as e:
                    logger.warning(f"RobStrideモーター {motor_name} の切断に失敗: {e}")
            
            # Dynamixelバスを切断
            try:
                self.dbus_r.disconnect()
                logger.info("Dynamixel右腕バスを切断しました")
            except Exception as e:
                logger.warning(f"Dynamixel右腕バスの切断に失敗: {e}")
            
            try:
                self.dbus_l.disconnect()
                logger.info("Dynamixel左腕バスを切断しました")
            except Exception as e:
                logger.warning(f"Dynamixel左腕バスの切断に失敗: {e}")
        
        finally:
            self.is_connected = False
            logger.info("MyAlohaロボットとの接続を切断しました")
