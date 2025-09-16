from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("my_aloha")
@dataclass
class MyAlohaConfig(RobotConfig):
    # 後で適当に変更する
    port: str  # Port to connect to the arm
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] = 5.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)