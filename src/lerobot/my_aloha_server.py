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
import time
from concurrent.futures import ThreadPoolExecutor
import queue
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
        # 非同期ロボット制御用
        self.robot_executor: Optional[ThreadPoolExecutor] = None
        self.robot_command_queue: queue.Queue = queue.Queue(maxsize=2)  # 最大2つのコマンドをキューイング
        self.robot_control_thread: Optional[threading.Thread] = None
        # スレッド同期と排他制御用（デバッグ版）
        self.stop_event = threading.Event()  # スレッド停止用Event
        self.robot_lock = threading.Lock()  # ロボット制御排他用Lock
        self.reset_in_progress = threading.Event()  # リセット処理中フラグ

    def initialize_robot(self):
        """MyAlohaロボットの初期化"""
        try:
            config = MyAlohaConfig(
                right_dynamixel_port="COM6",
                right_robstride_port="COM9",
                left_dynamixel_port="COM5",
                left_robstride_port="COM7",
                max_relative_target=0.05,
                current_limit_gripper_R=0.3,
                current_limit_gripper_L=0.5,
            )
            self.robot = MyAloha(config)
            self.robot.connect()
            self.robot_connected = True
            time.sleep(2.0)
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
        """リセット要求処理 - 改善されたスレッド同期と排他制御（デバッグ版）"""
        try:
            print("ロボットリセット処理開始...")
            if not self.robot_connected:
                response = {"status": "reset_error", "message": "ロボットが接続されていません"}
                await websocket.send(json.dumps(response))
                return
            # リセット処理中フラグを設定
            self.reset_in_progress.set()
            with self.robot_lock:  # ロボット制御を排他制御
                try:
                    # 1. 停止シグナルを送信
                    print("制御スレッド停止シグナル送信中...")
                    self.stop_event.set()
                    self.stop_threads = True
                    # 2. 制御スレッドの確実な停止を待機
                    if self.robot_control_thread and self.robot_control_thread.is_alive():
                        print("ロボット制御スレッドの停止を待機中...")
                        self.robot_control_thread.join(timeout=2.0)
                        # スレッドがまだ生きている場合は警告
                        if self.robot_control_thread.is_alive():
                            print("警告: ロボット制御スレッドが停止できませんでした")
                    # 3. キューを安全にクリア
                    print("コマンドキューをクリア中...")
                    cleared_count = 0
                    while not self.robot_command_queue.empty():
                        try:
                            self.robot_command_queue.get_nowait()
                            cleared_count += 1
                        except queue.Empty:
                            break
                    print(f"キューから{cleared_count}個のコマンドをクリアしました")
                    # 4. ロボットを排他制御下でホームポジションにリセット
                    print("ホームポジションに移動中...")
                    home_action = {
                        "joint_L_0": self.robot.old_action_L.motor1, "joint_L_1": self.robot.old_action_L.motor2, "joint_L_2": self.robot.old_action_L.motor3,
                        "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                        "joint_R_0": self.robot.old_action_R.motor1, "joint_R_1": self.robot.old_action_R.motor2, "joint_R_2": self.robot.old_action_R.motor3,
                        "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                    }
                    self.robot.send_action(home_action, use_relative=False)
                    await asyncio.sleep(1.0)
                    home_action = {
                        "joint_L_0": 0.0, "joint_L_1": 0.0, "joint_L_2": 0.0,
                        "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                        "joint_R_0": 0.0, "joint_R_1": 0.0, "joint_R_2": 0.0,
                        "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                    }
                    self.robot.send_action(home_action, use_relative=False)
                    await asyncio.sleep(1.0)
                    print("ホームポジション移動完了")
                finally:
                    self.stop_event.clear()
                    self.stop_threads = False
                    if self.is_receiving_joints:
                        print("UDP受信スレッドとロボット制御スレッドを再開中...")
                        if not self.joint_thread or not self.joint_thread.is_alive():
                            print("UDP受信スレッドを再起動中...")
                            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
                            self.joint_thread.daemon = True
                            self.joint_thread.start()
                        self.start_robot_control_thread()
                    self.reset_in_progress.clear()
            print("ロボットリセット処理完了")
            response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.reset_in_progress.clear()
            print(f"リセット処理エラー: {e}")
            import traceback
            traceback.print_exc()
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
            self.start_robot_control_thread()
            print("UDP通信スレッドと非同期ロボット制御スレッドを開始しました")

    def start_robot_control_thread(self):
        """非同期ロボット制御スレッドを開始"""
        if self.robot_connected:
            if self.robot_control_thread and self.robot_control_thread.is_alive():
                print("ロボット制御スレッドは既に動作中です")
                return
            self.robot_control_thread = threading.Thread(target=self.robot_control_worker)
            self.robot_control_thread.daemon = True
            self.robot_control_thread.start()
            print("新しいロボット制御スレッドを開始しました")

    def robot_control_worker(self):
        """ロボット制御ワーカースレッド - 改善されたスレッド同期と排他制御（デバッグ版）"""
        print("ロボット制御ワーカー開始")
        try:
            while self.robot_connected and not self.stop_threads and not self.stop_event.is_set():
                if self.reset_in_progress.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    action = self.robot_command_queue.get(timeout=0.1)
                    if action is not None:
                        with self.robot_lock:
                            if not self.reset_in_progress.is_set() and self.robot_connected:
                                self.robot.send_action(action)
                            else:
                                print("リセット処理中のため、ロボットコマンドをスキップします")
                        self.robot_command_queue.task_done()
                except queue.Empty:
                    # キューが空の場合は継続
                    continue
                except Exception as e:
                    print(f"ロボット制御エラー: {e}")
                    try:
                        self.robot_command_queue.task_done()
                    except ValueError:
                        pass  # task_doneが既に呼ばれている場合
                    continue
        finally:
            print("ロボット制御ワーカー終了")

    def send_robot_command_async(self, action):
        """非同期ロボットコマンド送信"""
        try:
            if self.robot_command_queue.full():
                try:
                    self.robot_command_queue.get_nowait()
                except queue.Empty:
                    pass
            self.robot_command_queue.put_nowait(action)
        except queue.Full:
            print("警告: ロボットコマンドキューが満杯です")

    def joint_receiver_thread(self):
        """関節角度受信スレッド（MyAlohaロボット制御統合版）"""
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(1.0)
        try:
            while self.is_receiving_joints and not self.stop_threads:
                try:
                    data, addr = sock.recvfrom(256)
                    if len(data) >= 57:  # 1バイト（モード） + 56バイト（14個のfloat32）
                        mode = data[0]
                        joint_angles = [
                            struct.unpack('<f', data[1 + i * 4:5 + i * 4])[0] 
                            for i in range(14)
                        ]
                        # 角度変換
                        joint_angles[0] += np.pi / 2
                        joint_angles[1] -= np.pi / 2
                        joint_angles[2] = -joint_angles[2] - np.pi / 2
                        joint_angles[3] = -joint_angles[3]
                        joint_angles[4] = -joint_angles[4]
                        joint_angles[5] = -joint_angles[5]
                        joint_angles[7] -= np.pi / 2
                        joint_angles[8] -= np.pi / 2
                        joint_angles[9] = -joint_angles[9] - np.pi / 2
                        joint_angles[10] = -joint_angles[10]
                        joint_angles[11] = -joint_angles[11]
                        joint_angles[12] = -joint_angles[12]
                        print(f"受信: {joint_angles[7]}")
                        if mode == 1 and self.robot_connected:
                            action = {
                                # 左腕
                                "joint_L_0": joint_angles[0], # ok
                                "joint_L_1": joint_angles[1], # ok
                                "joint_L_2": joint_angles[2], # ok
                                "joint_L_3": joint_angles[3], # ok
                                "joint_L_4": joint_angles[4], # ok
                                "joint_L_5": joint_angles[5], # ok
                                "gripper_L": joint_angles[6], # ok
                                # 右腕
                                "joint_R_0": joint_angles[7], # ok
                                "joint_R_1": joint_angles[8], # ok
                                "joint_R_2": joint_angles[9], # ok
                                "joint_R_3": joint_angles[10], # ok
                                "joint_R_4": joint_angles[11], # ok
                                "joint_R_5": joint_angles[12], # ok
                                "gripper_R": joint_angles[13], # ok
                            }
                            self.send_robot_command_async(action)
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
        print("WebSocket接続クリーンアップ開始...")
        
        # 接続状態フラグを更新
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        
        # UDP受信スレッドの終了
        if self.joint_thread and self.joint_thread.is_alive():
            print("UDP受信スレッドを終了中...")
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        
        # ロボット制御スレッドの終了
        if self.robot_control_thread and self.robot_control_thread.is_alive():
            print("ロボット制御スレッドを終了中...")
            self.robot_control_thread.join(timeout=2)
        self.robot_control_thread = None
        
        # キューをクリア
        print("コマンドキューをクリア中...")
        while not self.robot_command_queue.empty():
            try:
                self.robot_command_queue.get_nowait()
            except queue.Empty:
                break
        
        # WebSocket切断時にロボットを初期位置に戻す
        if self.robot_connected and self.robot:
            try:
                print("ロボットを初期位置に戻しています...")
                home_action = {
                    "joint_L_0": self.robot.old_action_L.motor1, "joint_L_1": self.robot.old_action_L.motor2, "joint_L_2": self.robot.old_action_L.motor3,
                    "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                    "joint_R_0": self.robot.old_action_R.motor1, "joint_R_1": self.robot.old_action_R.motor2, "joint_R_2": self.robot.old_action_R.motor3,
                    "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                }
                self.robot.send_action(home_action, use_relative=False)
                time.sleep(1.0)
                home_action = {
                    "joint_L_0": 0.0, "joint_L_1": 0.0, "joint_L_2": 0.0,
                    "joint_L_3": 0.0, "joint_L_4": 0.0, "joint_L_5": 0.0, "gripper_L": 0.0,
                    "joint_R_0": 0.0, "joint_R_1": 0.0, "joint_R_2": 0.0,
                    "joint_R_3": 0.0, "joint_R_4": 0.0, "joint_R_5": 0.0, "gripper_R": 0.0,
                }
                self.robot.send_action(home_action, use_relative=False)
                time.sleep(1.0)
                print("ロボット初期位置復帰完了")
            except Exception as e:
                print(f"初期位置復帰エラー: {e}")
        
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
