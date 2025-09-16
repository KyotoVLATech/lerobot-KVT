import abc
from dataclasses import dataclass

@dataclass
class AlohaArm:
    joint0: float
    joint1: float
    joint2: float
    joint3: float
    joint4: float
    joint5: float
    gripper: float

class AlohaController(abc.ABC):
    @abc.abstractmethod
    def __init__(self, right_robstride_port: str, left_robstride_port: str, right_dynamixel_port: str, left_dynamixel_port: str):
        pass

    @abc.abstractmethod
    def update_pos(self, right_arm, left_arm):
        pass

    @abc.abstractmethod
    def update_motor_pos(self, arm: str, motor_num: int, target_pos: float):
        pass

    @abc.abstractmethod
    def disable(self):
        pass