#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import websockets
import socket
import json
import struct
import threading
from typing import Optional
from collections import deque
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# 日本語フォントを追加
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Meiryo'
from lerobot.robots.my_aloha import MyAloha, MyAlohaConfig

class RobotCommunicationNode:
    def __init__(self, plot_data: bool = False):
        self.websocket_port = 8080
        self.unity_joint_port: Optional[int] = None
        self.is_connected = False
        self.is_receiving_joints = False
        self.joint_thread: Optional[threading.Thread] = None
        self.plot_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        self.robot: Optional[MyAloha] = None
        self.robot_connected = False
        
        # グラフ用データ保存
        self.max_data_points = 200  # 保存する最大データ数
        self.left_arm_data = {
            'time': deque(maxlen=self.max_data_points),
            'joint_L_0': deque(maxlen=self.max_data_points),
            'joint_L_1': deque(maxlen=self.max_data_points),
            'joint_L_2': deque(maxlen=self.max_data_points),
            'joint_L_3': deque(maxlen=self.max_data_points),
            'joint_L_4': deque(maxlen=self.max_data_points),
            'joint_L_5': deque(maxlen=self.max_data_points),
            'gripper_L': deque(maxlen=self.max_data_points),
        }
        self.right_arm_data = {
            'time': deque(maxlen=self.max_data_points),
            'joint_R_0': deque(maxlen=self.max_data_points),
            'joint_R_1': deque(maxlen=self.max_data_points),
            'joint_R_2': deque(maxlen=self.max_data_points),
            'joint_R_3': deque(maxlen=self.max_data_points),
            'joint_R_4': deque(maxlen=self.max_data_points),
            'joint_R_5': deque(maxlen=self.max_data_points),
            'gripper_R': deque(maxlen=self.max_data_points),
        }
        
        # グラフ表示設定
        self.show_plot = plot_data
        self.plot_initialized = False

    def initialize_robot(self):
        """MyAlohaロボットの初期化"""
        try:
            config = MyAlohaConfig(
                u2d2_port1="/dev/ttyUSB0",
                u2d2_port2="/dev/ttyUSB1",
                can_port1="/dev/ttyACM0",
                can_port2="/dev/ttyACM1",
                max_relative_target=5.0,
                # cameras={}
            )
            self.robot = MyAloha(config)
            self.robot.connect()
            self.robot_connected = True
            if self.robot_connected:
                home_action = {
                    "joint_L_0": 0.0, "joint_L_1": 0.0, "joint_L_2": 0.0,
                    "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                    "joint_R_0": 0.0, "joint_R_1": 0.0, "joint_R_2": 0.0,
                    "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                }
                self.robot.send_action(home_action)
            print("MyAlohaロボット初期化・接続完了")
        except Exception as e:
            print(f"ロボット初期化エラー: {e}")
            self.robot_connected = False

    async def websocket_handler(self, websocket):
        """WebSocket接続ハンドラー"""
        print(f"WebSocket接続: {websocket.remote_address}")
        # 新しい接続でロボットが切断されている場合は再接続を試みる
        if not self.robot_connected:
            print("ロボット再接続を試行中...")
            self.initialize_robot()
        try:
            print("Unity側からのメッセージを待機中...")
            message = await websocket.recv()
            data = json.loads(message)
            self.unity_joint_port = data.get('joint_send_port')
            self.is_connected = True
            self.start_udp_communication()
            await asyncio.sleep(0.5)
            # 接続確認レスポンスを送信
            response = {"status": "connected", "message": "接続情報受信完了"}
            await websocket.send(json.dumps(response))
            await self.handle_websocket_messages(websocket)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続がクライアント側から閉じられました")
        except Exception as e:
            print(f"WebSocketエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("WebSocket接続終了")
            self.cleanup_connection()

    async def handle_websocket_messages(self, websocket):
        """WebSocketメッセージの継続受信処理"""
        try:
            async for message in websocket:
                print(f"WebSocketメッセージ受信: {message}")
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    if command == 'reset_robot':
                        print("ロボットリセット要求を受信しました")
                        await self.handle_reset_request(websocket)
                    elif command == 'save_data':
                        print("データ保存要求を受信しました")
                    else:
                        print(f"不明なコマンド: {command}")
                except json.JSONDecodeError:
                    print(f"JSON解析エラー: {message}")
                except Exception as e:
                    print(f"メッセージ処理エラー: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続が閉じられました（メッセージ受信中）")
        except Exception as e:
            print(f"WebSocketメッセージ受信エラー: {e}")

    async def handle_reset_request(self, websocket):
        """リセット要求処理"""
        try:
            print("ロボットリセット処理開始...")
            if self.robot_connected:
                home_action = {
                    "joint_L_0": 0.0, "joint_L_1": 0.0, "joint_L_2": 0.0,
                    "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                    "joint_R_0": 0.0, "joint_R_1": 0.0, "joint_R_2": 0.0,
                    "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                }
                self.robot.send_action(home_action)
            await asyncio.sleep(2.0)
            print("ロボットリセット処理完了")
            response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"リセット処理エラー: {e}")
            error_response = {"status": "reset_error", "message": f"リセット処理でエラーが発生しました: {e}"}
            await websocket.send(json.dumps(error_response))

    def start_udp_communication(self):
        """UDP通信スレッドを開始"""
        if self.unity_joint_port:
            self.stop_threads = False
            self.is_receiving_joints = True
            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
            self.joint_thread.daemon = True
            self.joint_thread.start()
            print("UDP通信スレッドを開始しました")
            
            # グラフ表示スレッドを開始
            if self.show_plot:
                self.plot_thread = threading.Thread(target=self.plot_manager_thread)
                self.plot_thread.daemon = True
                self.plot_thread.start()
                print("グラフ表示スレッドを開始しました")

    def joint_receiver_thread(self):
        """関節角度受信スレッド（MyAlohaロボット制御統合版）"""
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(1.0)
        try:
            while self.is_receiving_joints and not self.stop_threads:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) >= 57:  # 1バイト（モード） + 56バイト（14個のfloat32）
                        mode = data[0]
                        joint_angles = []
                        for i in range(14):
                            offset = 1 + i * 4
                            angle = struct.unpack('<f', data[offset:offset+4])[0]
                            joint_angles.append(angle)
                        # データをqueueに保存（グラフ用）
                        current_time = time.time()
                        joint_angles[0] -= np.pi / 2
                        joint_angles[7] += np.pi / 2
                        joint_angles[1] -= np.pi / 2
                        joint_angles[8] -= np.pi / 2
                        joint_angles[2] = -joint_angles[2] - np.pi / 2
                        joint_angles[9] = -joint_angles[9] - np.pi / 2
                        joint_angles[3] = -joint_angles[3]
                        joint_angles[10] = -joint_angles[10]
                        joint_angles[5] = -joint_angles[5]
                        joint_angles[12] = -joint_angles[12]
                        self.save_joint_data(joint_angles, current_time)
                        
                        # ロボット制御
                        if mode == 1 and self.robot_connected:
                            action = {
                                # 左腕
                                "joint_L_0": joint_angles[0],
                                "joint_L_1": joint_angles[1], 
                                "joint_L_2": joint_angles[2],
                                "joint_L_3": joint_angles[3],
                                "joint_L_4": joint_angles[4],
                                "joint_L_5": joint_angles[5],
                                "gripper_L": joint_angles[6],
                                # 右腕
                                "joint_R_0": joint_angles[7],
                                "joint_R_1": joint_angles[8],
                                "joint_R_2": joint_angles[9], 
                                "joint_R_3": joint_angles[10],
                                "joint_R_4": joint_angles[11],
                                "joint_R_5": joint_angles[12],
                                "gripper_R": joint_angles[13],
                            }
                            self.robot.send_action(action)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_receiving_joints:
                        print(f"関節角度受信エラー: {e}")
                    break
        finally:
            sock.close()
            print("関節角度受信終了")

    def save_joint_data(self, joint_angles, timestamp):
        """関節角度データをqueueに保存"""
        joint_angles = np.array(joint_angles)
        joint_angles *= 180 / np.pi
        # 左腕データ（0-6）
        self.left_arm_data['time'].append(timestamp)
        self.left_arm_data['joint_L_0'].append(joint_angles[0])
        self.left_arm_data['joint_L_1'].append(joint_angles[1])
        self.left_arm_data['joint_L_2'].append(joint_angles[2])
        self.left_arm_data['joint_L_3'].append(joint_angles[3])
        self.left_arm_data['joint_L_4'].append(joint_angles[4])
        self.left_arm_data['joint_L_5'].append(joint_angles[5])
        self.left_arm_data['gripper_L'].append(joint_angles[6])
        # 右腕データ（7-13）
        self.right_arm_data['time'].append(timestamp)
        self.right_arm_data['joint_R_0'].append(joint_angles[7])
        self.right_arm_data['joint_R_1'].append(joint_angles[8])
        self.right_arm_data['joint_R_2'].append(joint_angles[9])
        self.right_arm_data['joint_R_3'].append(joint_angles[10])
        self.right_arm_data['joint_R_4'].append(joint_angles[11])
        self.right_arm_data['joint_R_5'].append(joint_angles[12])
        self.right_arm_data['gripper_R'].append(joint_angles[13])
    
    def plot_manager_thread(self):
        """グラフ表示を管理するスレッド"""
        try:
            # matplotlibの初期設定
            plt.ion()  # インタラクティブモードをオン
            
            # 図とサブプロットを作成
            self.fig, (self.ax_left, self.ax_right) = plt.subplots(2, 1, figsize=(12, 10))
            self.fig.suptitle('ロボット関節角度リアルタイムプロット', fontsize=16)
            
            # 左腕グラフの設定
            self.ax_left.set_title('左腕関節角度')
            self.ax_left.set_ylabel('角度 (度)')
            self.ax_left.set_ylim(-180, 180)
            self.ax_left.grid(True, alpha=0.3)
            
            # 右腕グラフの設定
            self.ax_right.set_title('右腕関節角度')
            self.ax_right.set_xlabel('時刻 (秒)')
            self.ax_right.set_ylabel('角度 (度)')
            self.ax_right.set_ylim(-180, 180)
            self.ax_right.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # リアルタイム更新ループ
            while not self.stop_threads:
                try:
                    self.update_plot()
                    plt.pause(0.1)  # 0.1秒間隔で更新
                except Exception as e:
                    if not self.stop_threads:
                        print(f"グラフ更新エラー: {e}")
                    break
                    
        except Exception as e:
            print(f"グラフスレッドエラー: {e}")
        finally:
            try:
                plt.close('all')
            except:
                pass
            print("グラフスレッド終了")
    
    def update_plot(self):
        """グラフを更新"""
        if len(self.left_arm_data['time']) < 2:
            return
            
        try:
            # データを配列に変換（長さを統一）
            left_times = list(self.left_arm_data['time'])
            right_times = list(self.right_arm_data['time'])
            
            # データ長の確認と調整
            left_data_length = len(left_times)
            right_data_length = len(right_times)
            
            if left_data_length == 0 or right_data_length == 0:
                return
            
            # 開始時刻を基準とした相対時間に変換
            start_time = left_times[0] if left_times else right_times[0]
            left_relative_times = [t - start_time for t in left_times]
            right_relative_times = [t - start_time for t in right_times]
            
            # 左腕グラフをクリア
            self.ax_left.clear()
            self.ax_left.set_title('Left Arm Joint Angles')
            self.ax_left.set_ylabel('Angle (degrees)')
            self.ax_left.set_ylim(-180, 180)
            self.ax_left.grid(True, alpha=0.3)
            
            # 左腕データをプロット（データ長を時間データに合わせる）
            joint_names = ['joint_L_0', 'joint_L_1', 'joint_L_2', 'joint_L_3', 'joint_L_4', 'joint_L_5', 'gripper_L']
            joint_keys = ['joint_L_0', 'joint_L_1', 'joint_L_2', 'joint_L_3', 'joint_L_4', 'joint_L_5', 'gripper_L']

            for i, (name, key) in enumerate(zip(joint_names, joint_keys)):
                joint_data = list(self.left_arm_data[key])
                # データ長を時間データに合わせる
                min_length = min(len(left_relative_times), len(joint_data))
                if min_length > 0:
                    time_slice = left_relative_times[:min_length]
                    data_slice = joint_data[:min_length]
                    self.ax_left.plot(time_slice, data_slice, label=name, linewidth=1.5)
            
            self.ax_left.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 右腕グラフをクリア
            self.ax_right.clear()
            self.ax_right.set_title('Right Arm Joint Angles')
            self.ax_right.set_xlabel('Time (seconds)')
            self.ax_right.set_ylabel('Angle (degrees)')
            self.ax_right.set_ylim(-180, 180)
            self.ax_right.grid(True, alpha=0.3)

            joint_names = ['joint_R_0', 'joint_R_1', 'joint_R_2', 'joint_R_3', 'joint_R_4', 'joint_R_5', 'gripper_R']
            joint_keys = ['joint_R_0', 'joint_R_1', 'joint_R_2', 'joint_R_3', 'joint_R_4', 'joint_R_5', 'gripper_R']
            
            # 右腕データをプロット（データ長を時間データに合わせる）
            for i, (name, key) in enumerate(zip(joint_names, joint_keys)):
                joint_data = list(self.right_arm_data[key])
                # データ長を時間データに合わせる
                min_length = min(len(right_relative_times), len(joint_data))
                if min_length > 0:
                    time_slice = right_relative_times[:min_length]
                    data_slice = joint_data[:min_length]
                    self.ax_right.plot(time_slice, data_slice, label=name, linewidth=1.5)
            
            self.ax_right.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"グラフ更新中のエラー: {e}")

    def cleanup_connection(self):
        """WebSocket接続のクリーンアップ（ロボット接続は保持）"""
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        if self.joint_thread and self.joint_thread.is_alive():
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        if self.plot_thread and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=2)
        self.plot_thread = None
        print("WebSocket接続クリーンアップ完了")

    def cleanup(self):
        """完全なリソースクリーンアップ"""
        self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                self.robot.disconnect()
                print("ロボット接続切断")
                self.robot_connected = False
            except Exception as e:
                print(f"ロボット切断エラー: {e}")
        print("完全クリーンアップ完了")

    async def start_server(self):
        """WebSocketサーバーを開始"""
        print(f"WebSocketサーバーを開始: ポート{self.websocket_port}")
        self.initialize_robot()
        try:
            async with websockets.serve(self.websocket_handler, "0.0.0.0", self.websocket_port):
                print("サーバー起動完了。Unityからの接続を待機中...")
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    node = RobotCommunicationNode()
    print("Unity-MyAloha通信サーバーを起動します...")
    asyncio.run(node.start_server())
