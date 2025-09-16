from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("my_aloha")
@dataclass
class MyAlohaConfig(RobotConfig):
    u2d2_port1: str
    u2d2_port2: str
    can_port1: str
    can_port2: str
    max_relative_target: float | dict[str, float] = 5.0
    # cameras: dict[str, CameraConfig] = field(default_factory=dict)