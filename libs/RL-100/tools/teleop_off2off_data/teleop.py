import os
import sys
import time
import math
import numpy as np
# from pynput import keyboard
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R

from xarm_wrapper import XArmWrapper
from franka_wrapper import FrankaWrapper
from robotiq_wrapper import RobotiqWrapper
from realsense import RealSense


AVP_IP = "192.168.31.207"

def SE3_inv(SE3):
    SE3_inv = np.eye(4, dtype=np.float32)
    SE3_inv[:3, :3] = SE3[:3, :3].T
    SE3_inv[:3, 3] = -SE3[:3, :3].T @ SE3[:3, 3]
    return SE3_inv


class AVPTeleop():
    def __init__(self):
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
    
    def set_init_pose(self, left_6d_pose=None, right_6d_pose=None):

        if left_6d_pose is not None:  # XArm
            self.robot_init_left = np.eye(4)
            self.robot_init_left[:3, 3] = left_6d_pose[:3]
            self.robot_init_left[:3, :3] = R.from_euler('xyz', left_6d_pose[3:6], degrees=False).as_matrix()
            self.hand_init_left = self.get_wrist_data('left')
        if right_6d_pose is not None:  # Franka
            self.robot_init_right = np.eye(4)
            self.robot_init_right[:3, 3] = right_6d_pose[:3]
            self.robot_init_right[:3, :3] = R.from_rotvec(right_6d_pose[3:6], degrees=False).as_matrix()
            self.hand_init_right = self.get_wrist_data('right')
        
    def print_frame(self, SE3):
        print("Translation:", SE3[:3, 3])
        print("Rotation (Euler xyz):", R.from_matrix(SE3[:3, :3]).as_euler('xyz', degrees=True))
        print("*" * 20)
        
    def get_avp_data(self, enable_left=True, enable_right=True):
        avp_dict = {}
        if enable_left:
            assert self.robot_init_left is not None
            avp_dict['left_arm'] = self.get_arm_pose('left')
            avp_dict['left_gripper'] = 1 if self.get_pinch_data('left') < 0.03 else 0
        if enable_right:
            assert self.robot_init_right is not None
            avp_dict['right_arm'] = self.get_arm_pose('right')
            avp_dict['right_gripper'] = 1 if self.get_pinch_data('right') < 0.03 else 0
        return avp_dict
    
    def is_terminate(self):
        left_pinch = self.get_ring_pinch_data('left') < 0.03
        right_pinch = self.get_ring_pinch_data('right') < 0.03
        return left_pinch or right_pinch
    
    def get_wrist_data(self, side):
        r = self.vps.latest
        wrist = r['left_wrist'] if side == 'left' else r['right_wrist']
        return wrist[0]
    
    def get_pinch_data(self, side):
        r = self.vps.latest
        pinch = r['left_pinch_distance'] if side == 'left' else r['right_pinch_distance']
        return pinch
    
    def get_ring_pinch_data(self, side):
        r = self.vps.latest
        fingers = r['left_fingers'] if side == 'left' else r['right_fingers']
        pinch = np.linalg.norm(fingers[4][:3, 3] - fingers[24][:3, 3])
        return pinch
        
    def get_arm_pose(self, side):
        robot_init = self.robot_init_left if side == 'left' else self.robot_init_right
        hand_init = self.hand_init_left if side == 'left' else self.hand_init_right

        X_VR2Robot = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ]) if side == 'left' else np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ])
        
        hand_pose = self.get_wrist_data(side)
        hand_transform_VR = np.eye(4, dtype=np.float32)
        hand_transform_VR[:3, :3] = hand_pose[:3, :3] @ hand_init[:3, :3].T
        hand_transform_VR[:3, 3] = hand_pose[:3, 3] - hand_init[:3, 3]
        hand_transform_robot = X_VR2Robot @ hand_transform_VR @ SE3_inv(X_VR2Robot)
        robot_pose = np.eye(4, dtype=np.float32)
        robot_pose[:3, :3] = hand_transform_robot[:3, :3] @ robot_init[:3, :3]
        robot_pose[:3, 3] = robot_init[:3, 3] + hand_transform_robot[:3, 3]
        
        # self.print_frame(robot_pose)
        # time.sleep(0.1)  # 确保打印输出的可读性
        return robot_pose


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    print("********** Camera is initialized **********")

    xarm = XArmWrapper(joints_init=[3.3, -14.1, -99.8, 1.3, 113.5, 4.3])
    xarm_gripper = RobotiqWrapper(robot='xarm')
    franka = FrankaWrapper(joints_init=(0.0845, -0.5603, -0.1064, -2.0000, -0.0475, 1.4359, 0.0))
    franka_gripper = RobotiqWrapper(robot='franka')
    print("********** Robots are initialized **********")
    print("********** AVP is initialized **********")
    
    demo_idx = 0
    demo_dir = f'/home/jkzhao/xarm_franka_teleop/data/demo_{demo_idx:03d}.npy'
    while os.path.exists(demo_dir):
        demo_idx += 1
        demo_dir = f'/home/jkzhao/xarm_franka_teleop/data/demo_{demo_idx:03d}.npy'

    # time.sleep(5)
    xarm_init_pose = xarm.get_position()
    franka_init_pose = franka.get_tcp_pose()
    avp_teleop = AVPTeleop()
    
    avp_teleop.set_init_pose(left_6d_pose=xarm_init_pose, right_6d_pose=franka_init_pose)
    demo_data = []
    
    dt = 1 / 20
    frame_idx = 0
    # while True:
    #     avp_dict = avp_teleop.get_avp_data(enable_left=False)
    while True:
        iter_start_time = time.time()

        xarm_q = xarm.get_joint()
        xarm_pose = xarm.get_position()
        xarm_gripper_state = xarm_gripper.get_state()
        franka_q = franka.get_joint()
        franka_pose = franka.get_tcp_pose()
        franka_gripper_state = franka_gripper.get_state()
        q_pos = np.concatenate([xarm_q, [xarm_gripper_state], franka_q, [franka_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state], franka_pose, [franka_gripper_state]])
        frame = camera.get_frame()

        avp_dict = avp_teleop.get_avp_data()

        # XArm
        xarm_target = np.zeros(6, dtype=np.float32)
        xarm_target[:3] = avp_dict['left_arm'][:3, 3] * 1000
        xarm_target[3:] = R.from_matrix(avp_dict['left_arm'][:3, :3]).as_euler('xyz', degrees=True)
        xarm.set_servo_cartesian(xarm_target)
        if avp_dict['left_gripper']:
            xarm_gripper.close()
        else:
            xarm_gripper.open()
        
        # Franka
        franka_target = np.zeros(6, dtype=np.float32)
        franka_target[:3] = avp_dict['right_arm'][:3, 3]
        franka_target[3:] = R.from_matrix(avp_dict['right_arm'][:3, :3]).as_rotvec()
        franka.franka.servoL(franka_target, dt)
        
        franka.franka.schedule_waypoint(franka_target, time.time() + dt)
        if avp_dict['right_gripper']:
            franka_gripper.close()
        else:
            franka_gripper.open()
        
        xarm_target[:3] /= 1000
        xarm_target[3:] = xarm_target[3:] / 180 * np.pi
        action = np.concatenate([xarm_target, [avp_dict['left_gripper']], franka_target, [avp_dict['right_gripper']]])
        demo_data.append({
            'demo_frame_idx': frame_idx,
            'depth': frame['depth'],
            'depth_scale': frame['depth_scale'],
            'qpos': q_pos,
            'eepose': ee_pose,
            'action': action
        })

        if avp_teleop.is_terminate():
            np.save(demo_dir, demo_data, allow_pickle=True)
            print("#" * 64)
            print(f"Success! Demo saved to {demo_dir}.")
            print(f"Demo length: {len(demo_data)}")
            print("#" * 64)
            break
        else:
            frame_idx += 1

        while time.time() - iter_start_time < dt:
            time.sleep(dt / 20)
        print(f'Iter: {frame_idx}, Frequency: {1 / (time.time() - iter_start_time):.3f} Hz')

    xarm_gripper.open()
    franka_gripper.open()
