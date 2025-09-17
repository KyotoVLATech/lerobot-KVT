#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Communicator用のダミー通信ノード
WebSocket接続、UDP関節角度受信をシミュレート
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
        self.stop_threads = False
        self.old = 0.0

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
        def limit_change(new, old):
            max_relative_target = 0.03
            delta = new - old
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            if abs(delta) > max_relative_target:
                delta = max_relative_target * np.sign(delta)
            new = old + delta
            return (new + np.pi) % (2 * np.pi) - np.pi
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ソケット再利用を許可
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(1.0)  # 1秒タイムアウト
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
                        joint_angles[7] -= np.pi / 2
                        joint_angles[7] = limit_change(joint_angles[7], self.old)
                        self.old = joint_angles[7]
                        print(f"R joint 1: {joint_angles[7] * 180.0 / np.pi}")

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_receiving_joints:
                        print(f"関節角度受信エラー: {e}")
                    break
        except Exception as e:
            print(f"関節角度受信スレッドエラー: {e}")
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
                await asyncio.Future()  # 永続的に実行
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    node = DummyCommunicationNode()
    asyncio.run(node.start_server())