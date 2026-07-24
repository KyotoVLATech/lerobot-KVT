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

# def SE3_inv(SE3):
#     SE3_inv = np.eye(4, dtype=np.float32)
#     SE3_inv[:3, :3] = SE3[:3, :3].T
#     SE3_inv[:3, 3] = -SE3[:3, :3].T @ SE3[:3, 3]
#     return SE3_inv

def SE3_inv(SE3):
    """
    计算SE3变换矩阵的逆，使用数值稳定的方法
    """
    R = SE3[:3, :3]
    t = SE3[:3, 3]
    
    # 检查旋转矩阵的条件数，如果接近奇异则使用更稳定的方法
    cond = np.linalg.cond(R)
    if cond > 1e12:  # 条件数过大，使用SVD分解
        U, s, Vt = np.linalg.svd(R)
        # 过滤掉过小的奇异值
        s_inv = np.where(s > 1e-12, 1.0/s, 0.0)
        R_inv = Vt.T @ np.diag(s_inv) @ U.T
    else:
        # 直接求逆，更高效
        R_inv = np.linalg.inv(R)
    
    # 确保旋转矩阵是正交的（修正数值误差）
    U, _, Vt = np.linalg.svd(R_inv)
    R_inv = U @ Vt
    
    # 计算平移部分
    t_inv = -R_inv @ t
    
    # 返回逆矩阵
    SE3_inv = np.eye(4, dtype=np.float32)
    SE3_inv[:3, :3] = R_inv
    SE3_inv[:3, 3] = t_inv
    return SE3_inv

class AVPTeleop():
    def __init__(self):
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
    
    def set_init_pose(self, left_6d_pose=None, right_6d_pose=None):
        if left_6d_pose is not None:  # Franka (左手控制)
            self.robot_init_left = np.eye(4)
            self.robot_init_left[:3, 3] = left_6d_pose[:3]
            self.robot_init_left[:3, :3] = R.from_rotvec(left_6d_pose[3:6], degrees=False).as_matrix()
            self.hand_init_left = self.get_wrist_data('left')
        if right_6d_pose is not None:  # XArm (右手控制)
            self.robot_init_right = np.eye(4)
            self.robot_init_right[:3, 3] = right_6d_pose[:3]
            self.robot_init_right[:3, :3] = R.from_euler('xyz', right_6d_pose[3:6], degrees=False).as_matrix()
            self.hand_init_right = self.get_wrist_data('right')
        print('self.hand_init_left', self.hand_init_left)
        print('self.hand_init_right', self.hand_init_right)
        
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

        # 保持原始的坐标变换矩阵，只交换控制映射
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
        
        # 数值稳定性检查
        if np.any(np.isnan(hand_pose)) or np.any(np.isinf(hand_pose)):
            print(f"[WARNING] {side} hand_pose contains NaN or Inf values")
            # 返回初始位姿作为安全值
            return robot_init
        
        # 检查hand_pose的旋转矩阵是否接近正交
        R_hand = hand_pose[:3, :3]
        R_hand_orthogonal_error = np.linalg.norm(R_hand @ R_hand.T - np.eye(3))
        if R_hand_orthogonal_error > 1e-6:
            print(f"[WARNING] {side} hand_pose rotation matrix not orthogonal, error: {R_hand_orthogonal_error}")
            # 强制正交化
            U, _, Vt = np.linalg.svd(R_hand)
            hand_pose[:3, :3] = U @ Vt
        
        # 计算相对变换
        hand_transform_VR = np.eye(4, dtype=np.float32)
        hand_transform_VR[:3, :3] = hand_pose[:3, :3] @ hand_init[:3, :3].T
        hand_transform_VR[:3, 3] = hand_pose[:3, 3] - hand_init[:3, 3]
        
        # 检查相对变换的合理性
        pos_delta_norm = np.linalg.norm(hand_transform_VR[:3, 3])
        if pos_delta_norm > 1.0:  # 如果位置变化超过1米，可能是异常值
            print(f"[WARNING] {side} position delta too large: {pos_delta_norm:.3f}m")
            # 限制位置变化
            if pos_delta_norm > 0.5:
                hand_transform_VR[:3, 3] = hand_transform_VR[:3, 3] * 0.5 / pos_delta_norm
        
        # 计算机器人坐标系下的变换
        hand_transform_robot = X_VR2Robot @ hand_transform_VR @ SE3_inv(X_VR2Robot)
        
        # 检查变换矩阵的合理性
        if np.any(np.isnan(hand_transform_robot)) or np.any(np.isinf(hand_transform_robot)):
            print(f"[WARNING] {side} hand_transform_robot contains NaN or Inf values")
            return robot_init
        
        # 计算最终机器人位姿
        robot_pose = np.eye(4, dtype=np.float32)
        robot_pose[:3, :3] = hand_transform_robot[:3, :3] @ robot_init[:3, :3]
        robot_pose[:3, 3] = robot_init[:3, 3] + hand_transform_robot[:3, 3]
        
        # 最终检查
        if np.any(np.isnan(robot_pose)) or np.any(np.isinf(robot_pose)):
            print(f"[WARNING] {side} robot_pose contains NaN or Inf values")
            return robot_init
        
        # 对于左手（Franka），添加额外的调试信息
        if side == 'left':
            pos_norm = np.linalg.norm(robot_pose[:3, 3])
            if pos_norm < 1e-3:  # 如果位置接近零
                print(f"[DEBUG] {side} robot_pose position very small: {robot_pose[:3, 3]}")
                print(f"[DEBUG] hand_transform_VR[:3, 3]: {hand_transform_VR[:3, 3]}")
                print(f"[DEBUG] hand_transform_robot[:3, 3]: {hand_transform_robot[:3, 3]}")
        
        return robot_pose
    
    def close(self):
        """关闭AVP连接"""
        try:
            if hasattr(self, 'vps') and self.vps is not None:
                self.vps.close()
        except Exception as e:
            print(f"[WARNING] Failed to close AVP connection: {e}")


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    print("********** Camera is initialized **********")

    xarm = XArmWrapper(joints_init=[3.3, -14.1, -99.8, 1.3, 113.5, 4.3])
    
    # 初始化夹爪（允许失败）
    try:
        xarm_gripper = RobotiqWrapper(robot='xarm')
        print("[INFO] XArm gripper initialized successfully")
    except Exception as e:
        print(f"[WARNING] Failed to initialize XArm gripper: {e}")
        xarm_gripper = None
    
    try:
        franka = FrankaWrapper(joints_init=(0.0845, -0.5603, -0.1064, -2.0000, -0.0475, 1.4359, 0.0))
        print("[INFO] Franka robot initialized successfully")
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize Franka robot: {e}")
        raise
    
    try:
        franka_gripper = RobotiqWrapper(robot='franka')
        print("[INFO] Franka gripper initialized successfully")
    except Exception as e:
        print(f"[WARNING] Failed to initialize Franka gripper: {e}")
        franka_gripper = None
    
    print("********** Robots are initialized **********")
    print("********** AVP is initialized **********")
    print("********** 左手控制Franka，右手控制XArm **********")
    
    demo_idx = 0
    demo_dir = f'/home/jkzhao/xarm_franka_teleop/data/demo_{demo_idx:03d}.npy'
    while os.path.exists(demo_dir):
        demo_idx += 1
        demo_dir = f'/home/jkzhao/xarm_franka_teleop/data/demo_{demo_idx:03d}.npy'

    # time.sleep(5)
    franka_init_pose = franka.get_tcp_pose()  # 左手控制Franka
    xarm_init_pose = xarm.get_position()      # 右手控制XArm
    avp_teleop = AVPTeleop()
    
    avp_teleop.set_init_pose(left_6d_pose=franka_init_pose, right_6d_pose=xarm_init_pose)
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
        # 获取Franka状态，如果死亡则退出程序
        if not franka.is_alive():
            print("[CRITICAL] Franka process is not alive!")
            print("[SAFETY] Terminating program for safety")
            # 安全清理
            if xarm_gripper is not None:
                try:
                    xarm_gripper.open()
                except:
                    pass
            if franka_gripper is not None:
                try:
                    franka_gripper.open()
                except:
                    pass
            exit(1)
        
        try:
            franka_q = franka.get_joint()
            franka_pose = franka.get_tcp_pose()
            franka_gripper_state = franka_gripper.get_state()
        except Exception as e:
            print(f"[CRITICAL] Failed to get Franka state: {e}")
            print("[SAFETY] Terminating program for safety")
            # 安全清理
            if xarm_gripper is not None:
                try:
                    xarm_gripper.open()
                except:
                    pass
            if franka_gripper is not None:
                try:
                    franka_gripper.open()
                except:
                    pass
            exit(1)
            
        q_pos = np.concatenate([xarm_q, [xarm_gripper_state], franka_q, [franka_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state], franka_pose, [franka_gripper_state]])
        frame = camera.get_frame()
        avp_dict = avp_teleop.get_avp_data()
        # 检查初始化是否成功
        if not franka.is_alive():
            print("[CRITICAL] Franka process is not alive during initialization!")
            print("[SAFETY] Terminating program for safety")
            # 安全清理
            if xarm_gripper is not None:
                try:
                    xarm_gripper.open()
                except:
                    pass
            if franka_gripper is not None:
                try:
                    franka_gripper.open()
                except:
                    pass
            exit(1)
        
        # 检查AVP数据是否正常，如果不正常则重启
        restart_start_time = time.time()
        max_restart_time = 10.0  # 最大重启时间10秒
        
        while np.linalg.norm(avp_dict['left_arm'][:3, 3]) < 0.1:
            current_time = time.time()
            elapsed_time = current_time - restart_start_time
            
            if elapsed_time > max_restart_time:
                print(f"[CRITICAL] AVP data abnormal for {elapsed_time:.1f}s, exceeding {max_restart_time}s limit")
                print('avp_dict left_arm:', avp_dict['left_arm'][:3, 3])
                print('avp_dict right_arm:', avp_dict['right_arm'][:3, 3])
                print("[SAFETY] Terminating program for safety")
                # 安全清理
                if xarm_gripper is not None:
                    try:
                        xarm_gripper.open()
                    except:
                        pass
                if franka_gripper is not None:
                    try:
                        franka_gripper.open()
                    except:
                        pass
                exit(1)
            
            print(f"[WARNING] AVP data abnormal (elapsed: {elapsed_time:.1f}s), restarting AVP and Franka...")
            print('avp_dict left_arm:', avp_dict['left_arm'][:3, 3])
            print('avp_dict right_arm:', avp_dict['right_arm'][:3, 3])
            
            # 重启Franka
            try:
                print("[RESTART] Restarting Franka...")
                franka.restart()
                time.sleep(2.0)  # 等待Franka启动
                
                # 重新获取Franka初始位姿
                franka_init_pose = franka.get_tcp_pose()
                print("[RESTART] Franka restarted successfully")
                
            except Exception as e:
                print(f"[ERROR] Failed to restart Franka: {e}")
                continue
            
            # 重启AVP
            try:
                print("[RESTART] Restarting AVP...")
                avp_teleop.close()  # 关闭旧的AVP连接
                time.sleep(1.0)
                
                # 重新创建AVP连接
                avp_teleop.vps = VisionProStreamer(ip=AVP_IP, record=True)
                time.sleep(2.0)  # 等待AVP连接建立
                
                # 重新设置初始位姿
                avp_teleop.set_init_pose(left_6d_pose=franka_init_pose, right_6d_pose=xarm_init_pose)
                print("[RESTART] AVP restarted successfully")
                
            except Exception as e:
                print(f"[ERROR] Failed to restart AVP: {e}")
                continue
            
            # 重新获取AVP数据
            try:
                avp_dict = avp_teleop.get_avp_data()
                print(f"[RESTART] New AVP data - left_arm norm: {np.linalg.norm(avp_dict['left_arm'][:3, 3]):.6f}")
            except Exception as e:
                print(f"[ERROR] Failed to get new AVP data: {e}")
                continue
            
            # 短暂等待后再次检查
            time.sleep(0.5)
        
        # 如果成功恢复，打印成功信息
        if time.time() - restart_start_time > 0.1:  # 如果进行了重启
            print(f"[SUCCESS] AVP and Franka restarted successfully in {time.time() - restart_start_time:.1f}s")
    
           
        # XArm (右手控制)
        xarm_target = np.zeros(6, dtype=np.float32)
        xarm_target[:3] = avp_dict['right_arm'][:3, 3] * 1000
        xarm_target[3:] = R.from_matrix(avp_dict['right_arm'][:3, :3]).as_euler('xyz', degrees=True)
        xarm.set_servo_cartesian(xarm_target)
        if xarm_gripper is not None:
            if avp_dict['right_gripper']:
                xarm_gripper.close()
            else:
                xarm_gripper.open()
        # Franka (左手控制)
        # 再次检查Franka进程是否存活（双重检查）
        if not franka.is_alive():
            print("[CRITICAL] Franka process died during execution!")
            print("[SAFETY] Terminating program for safety")
            # 安全清理
            if xarm_gripper is not None:
                try:
                    xarm_gripper.open()
                except:
                    pass
            if franka_gripper is not None:
                try:
                    franka_gripper.open()
                except:
                    pass
            exit(1)
        
        try:
            franka_target = np.zeros(6, dtype=np.float32)
            franka_target[:3] = avp_dict['left_arm'][:3, 3]
            franka_target[3:] = R.from_matrix(avp_dict['left_arm'][:3, :3]).as_rotvec()
            print('\n franka_pose', franka_pose,'\n franka_target', franka_target, '\n avp_dict', avp_dict['left_arm'][:3, 3])
            print('franka_init_pose', franka_init_pose)
            
            # 发送控制命令
            franka.franka.servoL(franka_target, dt)
            franka.franka.schedule_waypoint(franka_target, time.time() + dt)
            
            # 控制夹爪
            if franka_gripper is not None:
                if avp_dict['left_gripper']:
                    franka_gripper.close()
                else:
                    franka_gripper.open()
                
        except Exception as e:
            print(f"[CRITICAL] Franka control error: {e}")
            print("[SAFETY] Terminating program for safety")
            # 安全清理
            if xarm_gripper is not None:
                try:
                    xarm_gripper.open()
                except:
                    pass
            if franka_gripper is not None:
                try:
                    franka_gripper.open()
                except:
                    pass
            exit(1)
        
        
        
        # 准备action数据 (保持原有格式: [xarm_action, xarm_gripper, franka_action, franka_gripper])
        xarm_target[:3] /= 1000
        xarm_target[3:] = xarm_target[3:] / 180 * np.pi
        action = np.concatenate([xarm_target, [avp_dict['right_gripper']], franka_target, [avp_dict['left_gripper']]])
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

    # 安全清理
    print("[SAFETY] Opening all grippers for safety")
    if xarm_gripper is not None:
        try:
            xarm_gripper.open()
        except Exception as e:
            print(f"[WARNING] Failed to open XArm gripper: {e}")
    
    if franka_gripper is not None:
        try:
            franka_gripper.open()
        except Exception as e:
            print(f"[WARNING] Failed to open Franka gripper: {e}")
    
    # 关闭AVP连接
    try:
        avp_teleop.close()
    except Exception as e:
        print(f"[WARNING] Failed to close AVP connection: {e}")
    
    print("[SAFETY] All grippers opened, program terminated safely") 