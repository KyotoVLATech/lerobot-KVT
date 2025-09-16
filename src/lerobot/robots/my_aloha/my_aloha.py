from ..robot import Robot
from .config_my_aloha import MyAlohaConfig
from .aloha_abc import AlohaArm, AlohaController
# from kvt_aloha import AlohaController AlohaArm

class MyAloha(Robot):
    config_class = MyAlohaConfig
    name = "my_aloha"

    def __init__(
        self,
        config: MyAlohaConfig,
    ):
        super().__init__(config)
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

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        action_L = AlohaArm(
            joint0=action.get("joint_L_0", self.old_action_L.joint0),
            joint1=action.get("joint_L_1", self.old_action_L.joint1),
            joint2=action.get("joint_L_2", self.old_action_L.joint2),
            joint3=action.get("joint_L_3", self.old_action_L.joint3),
            joint4=action.get("joint_L_4", self.old_action_L.joint4),
            joint5=action.get("joint_L_5", self.old_action_L.joint5),
            gripper=action.get("gripper_L", self.old_action_L.gripper),
        )
        action_R = AlohaArm(
            joint0=action.get("joint_R_0", self.old_action_R.joint0),
            joint1=action.get("joint_R_1", self.old_action_R.joint1),
            joint2=action.get("joint_R_2", self.old_action_R.joint2),
            joint3=action.get("joint_R_3", self.old_action_R.joint3),
            joint4=action.get("joint_R_4", self.old_action_R.joint4),
            joint5=action.get("joint_R_5", self.old_action_R.joint5),
            gripper=action.get("gripper_R", self.old_action_R.gripper),
        )
        action_L = self.apply_max_relative_target(action_L, self.old_action_L)
        action_R = self.apply_max_relative_target(action_R, self.old_action_R)
        self.aloha.update_pos(action_R, action_L)
        self.old_action_L = action_L
        self.old_action_R = action_R
        return action

    def apply_max_relative_target(self, current: AlohaArm, old: AlohaArm) -> AlohaArm:
        def limit_change(new, old):
            delta = new - old
            if abs(delta) > self.config.max_relative_target:
                delta = self.config.max_relative_target * (1 if delta > 0 else -1)
            return old + delta
        return AlohaArm(
            joint0=limit_change(current.joint0, old.joint0),
            joint1=limit_change(current.joint1, old.joint1),
            joint2=limit_change(current.joint2, old.joint2),
            joint3=limit_change(current.joint3, old.joint3),
            joint4=limit_change(current.joint4, old.joint4),
            joint5=limit_change(current.joint5, old.joint5),
            gripper=current.gripper,
        )

    def disconnect(self):
        if self.aloha is not None:
            self.aloha.disable()
            self.aloha = None