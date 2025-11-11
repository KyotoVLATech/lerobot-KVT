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
        self.robot_control_task: Optional[asyncio.Task] = None
        # 非同期同期プリミティブ
        self.stop_event = asyncio.Event()  # タスク停止用Event
        self.robot_lock = asyncio.Lock()  # ロボット制御排他用Lock
        self.reset_in_progress = asyncio.Event()  # リセット処理中フラグ
        # UDP受信スレッドとの通信用（スレッドセーフ）
        self.latest_action = None
        self.action_lock = threading.Lock()  # UDP受信スレッド用
        self.control_frequency = 60 # Hz

    async def initialize_robot(self):
        try:
            config = MyAlohaConfig(
                right_dynamixel_port="/dev/ttyUSB0",
                right_robstride_port="/dev/ttyUSB2",
                left_robstride_port="/dev/ttyUSB3",
                left_dynamixel_port="/dev/ttyUSB1",
                max_relative_target_1=0.03, # yaw
                max_relative_target_2=0.01, # pitch
                max_relative_target_3=0.01, # pitch
                max_relative_target_4=0.03, # yaw
                max_relative_target_5=0.01, # pitch
                max_relative_target_6=0.03, # yaw
                current_limit_gripper_R=0.5,
                current_limit_gripper_L=0.5,
            )
            self.robot = MyAloha(config)
            await self.robot.connect()
            self.robot_connected = True
            await asyncio.sleep(2.0)
            print("MyAlohaロボット初期化・接続完了")
        except Exception as e:
            print(f"ロボット初期化エラー: {e}")
            self.robot_connected = False

    async def websocket_handler(self, websocket):
        print(f"WebSocket接続: {websocket.remote_address}")
        # 新しい接続でロボットが切断されている場合は再接続を試みる
        if not self.robot_connected:
            print("ロボット再接続を試行中...")
            await self.initialize_robot()
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
            await self.cleanup_connection()

    async def handle_websocket_messages(self, websocket):
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
                    elif command == 'discard_data':
                        print("データ破棄要求を受信しました")
                    elif command == 'recording':
                        print("recordingを受信しました")
                    elif command == 'teleoperation':
                        print("teleoperationを受信しました")
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
        try:
            print("ロボットリセット処理開始...")
            if not self.robot_connected:
                response = {"status": "reset_error", "message": "ロボットが接続されていません"}
                await websocket.send(json.dumps(response))
                return
            # リセット処理中フラグを設定
            self.reset_in_progress.set()
            async with self.robot_lock:  # ロボット制御を排他制御
                try:
                    print("制御タスク停止シグナル送信中...")
                    self.stop_event.set()
                    self.stop_threads = True
                    if self.robot_control_task and not self.robot_control_task.done():
                        print("ロボット制御タスクの停止を待機中...")
                        try:
                            await asyncio.wait_for(self.robot_control_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            print("警告: ロボット制御タスクが停止できませんでした")
                            self.robot_control_task.cancel()
                    home_action = self.robot.old_action.copy()
                    home_action[4:7] = 0.0  # L_3, L_4, L_5
                    home_action[11:14] = 0.0  # R_3, R_4, R_5
                    await self.robot.send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(1.0)
                    home_action = np.zeros_like(home_action)
                    await self.robot.send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(1.0)
                    print("ホームポジション移動完了")
                finally:
                    with self.action_lock:
                        self.latest_action = None
                    self.stop_event.clear()
                    self.stop_threads = False
                    if self.is_receiving_joints:
                        print("UDP受信スレッドとロボット制御タスクを再開中...")
                        if not self.joint_thread or not self.joint_thread.is_alive():
                            print("UDP受信スレッドを再起動中...")
                            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
                            self.joint_thread.daemon = True
                            self.joint_thread.start()
                        await self.start_robot_control_task()
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
        if self.unity_joint_port:
            self.stop_threads = False
            self.is_receiving_joints = True
            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
            self.joint_thread.daemon = True
            self.joint_thread.start()
            # 非同期タスクを開始するためのヘルパー
            asyncio.create_task(self.start_robot_control_task())
            print("UDP通信スレッドと非同期ロボット制御タスクを開始しました")

    async def start_robot_control_task(self):
        if self.robot_connected:
            if self.robot_control_task and not self.robot_control_task.done():
                print("ロボット制御タスクは既に動作中です")
                return
            self.robot_control_task = asyncio.create_task(self.robot_control_worker())
            print("新しいロボット制御タスクを開始しました")

    async def robot_control_worker(self):
        print("ロボット制御ワーカー開始")
        try:
            first_action_time = None
            current_latest_action = None
            while self.robot_connected and not self.stop_threads and not self.stop_event.is_set():
                start_time = time.perf_counter()
                # Unity側からデータが届くまで待機
                with self.action_lock:
                    if self.latest_action is not None:
                        current_latest_action = self.latest_action.copy()
                
                # データがまだ届いていない場合は待機してcontinue
                if current_latest_action is None:
                    await asyncio.sleep(0.01)
                    continue
                
                if self.reset_in_progress.is_set():
                    await asyncio.sleep(0.1)
                    continue
                
                if first_action_time is None:
                    first_action_time = time.time()
                    print("初回アクション受信を記録しました。3秒後にuse_relativeがFalseになります。")
                # 初回アクション受信から3秒経過したかチェック
                elapsed_since_first_action = time.time() - first_action_time
                use_relative = elapsed_since_first_action < 3.0
                async with self.robot_lock:
                    if not self.reset_in_progress.is_set() and self.robot_connected:
                        await self.robot.send_action(current_latest_action, use_relative=use_relative, use_filter=not use_relative)
                elapsed_time = time.perf_counter() - start_time
                sleep_duration = 1.0 / self.control_frequency - elapsed_time
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            print("ロボット制御ワーカーがキャンセルされました")
        finally:
            print("ロボット制御ワーカー終了")

    def joint_receiver_thread(self):
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(0.1)
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
                        # print(f"受信: {joint_angles[7]}")
                        if mode == 1 and self.robot_connected:
                            with self.action_lock:
                                self.latest_action = np.array(joint_angles, dtype=np.float32)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_receiving_joints:
                        print(f"関節角度受信エラー: {e}")
                    break
        finally:
            sock.close()
            print("関節角度受信終了")

    async def cleanup_connection(self):
        print("WebSocket接続クリーンアップ開始...")
        # 接続状態フラグを更新
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        self.stop_event.set()
        
        # ロボット制御タスクの終了
        if self.robot_control_task and not self.robot_control_task.done():
            print("ロボット制御タスクを終了中...")
            self.robot_control_task.cancel()
            try:
                await asyncio.wait_for(self.robot_control_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self.robot_control_task = None
        
        # UDP受信スレッドの終了
        if self.joint_thread and self.joint_thread.is_alive():
            print("UDP受信スレッドを終了中...")
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        
        with self.action_lock:
            self.latest_action = None
        
        # WebSocket切断時にロボットを初期位置に戻す
        if self.robot_connected and self.robot:
            try:
                print("ロボットを初期位置に戻しています...")
                home_action = self.robot.old_action.copy()
                home_action[4:7] = 0.0  # L_3, L_4, L_5
                home_action[11:14] = 0.0  # R_3, R_4, R_5
                await self.robot.send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(1.0)
                home_action = np.zeros_like(home_action)
                await self.robot.send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(1.0)
                print("ロボット初期位置復帰完了")
            except Exception as e:
                print(f"初期位置復帰エラー: {e}")
        print("WebSocket接続クリーンアップ完了")

    async def cleanup(self):
        await self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                await self.robot.disconnect()
                print("ロボット接続切断")
                self.robot_connected = False
            except Exception as e:
                print(f"ロボット切断エラー: {e}")
        print("完全クリーンアップ完了")

    async def start_server(self):
        print(f"WebSocketサーバーを開始: ポート{self.websocket_port}")
        await self.initialize_robot()
        try:
            async with websockets.serve(self.websocket_handler, "0.0.0.0", self.websocket_port):
                print("サーバー起動完了。Unityからの接続を待機中...")
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            await self.cleanup()

if __name__ == "__main__":
    node = RobotCommunicationNode()
    print("Unity-MyAloha通信サーバーを起動します...")
    asyncio.run(node.start_server())

# uv run src/lerobot/my_aloha_server.py
