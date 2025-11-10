#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Communicator用のダミー通信ノード
WebSocket接続、UDP関節角度受信をシミュレート
Aloha Transfer環境でActionを実行
"""

import asyncio
import websockets
import socket
import json
import struct
import time
import threading
from typing import Optional
import numpy as np
from lerobot.envs.factory import make_env_config, make_env

class DummyCommunicationNode:
    def __init__(self):
        # 通信設定
        self.websocket_port = 8080
        self.unity_joint_port: Optional[int] = None
        # 通信状態
        self.is_connected = False
        self.is_receiving_joints = False
        # スレッド制御
        self.joint_thread: Optional[threading.Thread] = None
        self.websocket_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        # データ受信周期
        self.receive_interval = 0.01
        # 指数平滑フィルタの設定
        self.filter_alpha = 0.5  # 平滑化係数（0.0〜1.0、1.0に近いほど新しい値の影響が大きい）
        self.filtered_joint_angles = None  # フィルタ済み関節角度（初回受信時に初期化）
        self.delta_angle = None
        
        # Aloha Transfer環境の作成
        print("Aloha Transfer環境を初期化中...")
        self.env_cfg = make_env_config("aloha", task="AlohaTransferCube-v0")
        env_dict = make_env(self.env_cfg, n_envs=1)
        self.vec_env = env_dict["aloha"][0]
        print("環境の初期化完了")
        
        # 環境をリセット
        self.reset_environment()

    def reset_environment(self):
        """環境をリセット"""
        print("環境をリセット中...")
        observation, info = self.vec_env.reset()
        self.filtered_joint_angles = None
        self.delta_angle = None
        print("環境のリセット完了")
        return observation, info

    async def websocket_handler(self, websocket):
        """WebSocket接続ハンドラー"""
        print(f"WebSocket接続: {websocket.remote_address}")
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
            self.cleanup()

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
            # 環境をリセット
            self.reset_environment()
            await asyncio.sleep(2.0)  # 2秒間のリセット処理をシミュレート
            print("ロボットリセット処理が完了しました")
            response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"リセット処理エラー: {e}")
            error_response = {"status": "reset_error", "message": f"リセット処理でエラーが発生しました: {e}"}
            try:
                await websocket.send(json.dumps(error_response))
            except:
                print("エラー通知の送信に失敗しました")

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
        """関節角度受信スレッド"""
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(1.0)
        sock.setblocking(False)
        start_time = time.time()
        try:
            while self.is_receiving_joints and not self.stop_threads:
                latest_data_packet = None
                addr = None
                try:
                    # バッファが空になるまでループで読み出す
                    while True:
                        latest_data_packet, addr = sock.recvfrom(1024)
                except (BlockingIOError, socket.error):
                    # バッファが空で、もう読み出すデータがない
                    pass
                
                if latest_data_packet and len(latest_data_packet) >= 57:
                    mode = latest_data_packet[0]
                    joint_angles = []
                    for i in range(14):
                        offset = 1 + i * 4
                        angle = struct.unpack('<f', latest_data_packet[offset:offset+4])[0]
                        joint_angles.append(angle)
                    
                    # 指数平滑フィルタの適用
                    if self.filtered_joint_angles is None:
                        # 初回受信時は受信値をそのまま使用
                        self.filtered_joint_angles = np.array(joint_angles)
                    else:
                        self.delta_angle = -self.filtered_joint_angles
                        # 指数平滑フィルタを適用
                        self.filtered_joint_angles = (
                            self.filter_alpha * np.array(joint_angles) +
                            (1.0 - self.filter_alpha) * self.filtered_joint_angles
                        )
                        self.delta_angle += self.filtered_joint_angles
                else:
                    # パケットをロスしている際にデータを補完する処理を行う
                    if self.delta_angle is not None:
                        self.filtered_joint_angles += self.delta_angle
                
                # 環境にActionを反映
                if self.filtered_joint_angles is not None:
                    try:
                        # Actionを環境に適用
                        action = self.filtered_joint_angles.reshape(1, -1)  # (1, 14)の形状に変換
                        observation, reward, terminated, truncated, info = self.vec_env.step(action)
                        
                        # エピソード終了時の処理
                        if terminated[0] or truncated[0]:
                            print(f"エピソード終了 - reward: {reward[0]:.3f}, terminated: {terminated[0]}, truncated: {truncated[0]}")
                            # 自動的にリセット（AutoresetModeがSAME_STEPなので自動リセットされる）
                    except Exception as e:
                        print(f"環境ステップエラー: {e}")
                        import traceback
                        traceback.print_exc()
                
                time.sleep(max(0.0, self.receive_interval - (time.time() - start_time)))
                start_time = time.time()
        except Exception as e:
            print(f"関節角度受信スレッドエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sock.close()
            print("関節角度受信終了")

    def cleanup(self):
        """リソースクリーンアップ"""
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        
        if self.joint_thread and self.joint_thread.is_alive():
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        
        # 環境のクリーンアップ
        if hasattr(self, 'vec_env') and self.vec_env is not None:
            try:
                self.vec_env.close()
                print("環境をクローズしました")
            except Exception as e:
                print(f"環境クローズエラー: {e}")
        
        print("クリーンアップ完了")

    async def start_server(self):
        """WebSocketサーバーを開始"""
        print(f"WebSocketサーバーを開始: ポート{self.websocket_port}")
        try:
            async with websockets.serve(
                self.websocket_handler,
                "0.0.0.0",
                self.websocket_port
            ):
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            self.cleanup()

    def start_server_thread(self):
        """WebSocketサーバーを別スレッドで開始"""
        def run_server():
            asyncio.run(self.start_server())
        self.websocket_thread = threading.Thread(target=run_server)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()
        print("WebSocketサーバースレッドを開始しました")

    def run(self):
        """メインの実行関数"""
        print("=" * 60)
        print("UDP Test Server with Aloha Transfer Environment")
        print("=" * 60)
        # WebSocketサーバーを別スレッドで開始
        self.start_server_thread()
        # メインスレッドで待機
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        finally:
            self.cleanup()

if __name__ == "__main__":
    node = DummyCommunicationNode()
    node.run()
