#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import websockets
import socket
import json
import struct
import threading
import numpy as np
from typing import Optional
from lerobot.robots.my_aloha import MyAloha, MyAlohaConfig

class RobotCommunicationNode:
    def __init__(self):
        self.websocket_port = 8080
        self.unity_joint_port: Optional[int] = None
        self.is_connected = False
        self.is_receiving_joints = False
        self.joint_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        self.robot: Optional[MyAloha] = None
        self.robot_connected = False

    def initialize_robot(self):
        """MyAlohaロボットの初期化"""
        try:
            config = MyAlohaConfig(
                right_robstride_port="COM3",
                left_robstride_port="COM4",
                right_dynamixel_port="COM5",
                left_dynamixel_port="COM6",
                max_relative_target=0.1,
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
                        joint_angles[0] -= np.pi / 2
                        joint_angles[1] -= np.pi / 2
                        joint_angles[2] = -joint_angles[2] - np.pi / 2
                        joint_angles[3] = -joint_angles[3]
                        joint_angles[5] = -joint_angles[5]
                        joint_angles[7] += np.pi / 2
                        joint_angles[8] -= np.pi / 2
                        joint_angles[9] = -joint_angles[9] - np.pi / 2
                        joint_angles[10] = -joint_angles[10]
                        joint_angles[12] = -joint_angles[12]
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

    def cleanup_connection(self):
        """WebSocket接続のクリーンアップ（ロボット接続は保持）"""
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        if self.joint_thread and self.joint_thread.is_alive():
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
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

# uv run src\lerobot\my_aloha_server.py
