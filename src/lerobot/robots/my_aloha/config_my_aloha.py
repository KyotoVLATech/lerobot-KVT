from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("my_aloha")
@dataclass
class MyAlohaConfig(RobotConfig):
    right_robstride_port: str
    left_robstride_port: str
    right_dynamixel_port: str
    left_dynamixel_port: str
    max_relative_target: float = 0.1 # radians
    current_limit_gripper_R: float = 0.3 # Amperes
    current_limit_gripper_L: float = 0.3 # Amperes