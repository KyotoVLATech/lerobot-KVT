"""iLoHa towel folding: bimanual ALOHA-style robot, 14D absolute joint actions at 30 Hz.

Demos come from scripts/iloha/export_lerobot_to_hdf5.py. The online rollout client is not
client/run_client.py but iloha_rl.py at the lerobot-KVT root (it runs in the robot's own
lerobot environment), so `env` is not set here.
"""

import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "iloha"
    config.env_name = "iloha_towel"
    config.language_instruction = "Grab the edge of the towel and fold it twice."

    # traj.hdf5 action stream: action/joint_position already includes both grippers
    # (joint_6, joint_13), so there is no separate gripper stream.
    config.action_space = "joint_position"
    config.gripper_action_space = None

    config.control_hz = 30

    # Action-shaped zero used for env/replay-buffer initialization.
    config.example_action = np.zeros((1, 14))

    # The xyz/gripper-only edit mask assumes DROID's cartesian layout; edit every joint.
    config.edit_action_xyzg = False

    # Bimanual: the online critic sees both wrists (default is base + left wrist only).
    config.critic_camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    return config
