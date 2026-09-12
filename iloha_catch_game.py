#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""記録済みエピソードを再生し、初期位置から片腕の遠隔操作へ移行する。"""
import argparse
import asyncio
import json
import struct
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import websockets

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.robots.iloha.iloha_controller import left_settings, right_settings
from iloha_mapping import JOINT_NAMES, aloha_to_iloha


def build_replay_intervals(
    actions: np.ndarray,
    fps: float,
    base_speed: float = 1.0,
    max_speedup: float = 1.0,
    gripper_margin: float = 0.5,
    speedup_distance: float = 2.0,
    gripper_threshold: float = 1e-4,
) -> np.ndarray:
    """隣接フレーム間の待機時間を計算する。

    左右どちらかのグリッパー指令のフレーム間変化が閾値を超えた区間を
    動作区間とする。距離・余白は倍率適用前の記録時間（秒）で指定する。
    余白の外側ではsmoothstepで追加倍率を1からmax_speedupまで上げる。
    動作区間がない場合は全区間で最大倍率を使う。
    """
    for name, value in (("fps", fps), ("base_speed", base_speed),
                        ("speedup_distance", speedup_distance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name}は有限の正数で指定してください")
    if not np.isfinite(max_speedup) or max_speedup < 1:
        raise ValueError("max_speedupは1以上の有限値で指定してください")
    for name, value in (("gripper_margin", gripper_margin), ("gripper_threshold", gripper_threshold)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name}は0以上の有限値で指定してください")
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("actionsには有限値からなる1フレーム以上の14関節指令が必要です")

    moving = np.any(np.abs(np.diff(actions[:, [6, 13]], axis=0)) > gripper_threshold, axis=1)
    indices = np.arange(len(moving))
    if moving.any():
        # 前後両方向の最寄り動作区間をO(N)で求める。
        previous = np.maximum.accumulate(np.where(moving, indices, -np.inf))
        following = np.minimum.accumulate(np.where(moving, indices, np.inf)[::-1])[::-1]
        # 区間同士の端点間距離。隣接する区間も等速にして境界を保護する。
        distance = np.maximum(0.0, np.minimum(indices - previous, following - indices) - 1) / fps
        ramp = np.clip((distance - gripper_margin) / speedup_distance, 0.0, 1.0)
    else:
        ramp = np.ones(len(moving))
    speedup = 1.0 + (max_speedup - 1.0) * ramp**2 * (3.0 - 2.0 * ramp)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        intervals = (1.0 / fps / base_speed) / speedup
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("速度設定によるフレーム間隔が表現可能な範囲を超えています")
    return intervals


def load_episode(dataset_path: str, episode_index: int) -> tuple[np.ndarray, float]:
    """ロボット接続前にエピソード全体を読み込み、関節順序と値を検証する。"""
    root = Path(dataset_path).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"データセットが見つかりません: {root}")
    if episode_index < 0:
        raise ValueError("episode_indexは0以上で指定してください")
    dataset = LeRobotDataset(
        repo_id=f"local/{root.name}", root=root, episodes=[episode_index],
    )
    feature = dataset.features.get("action", {})
    names = tuple(feature.get("names") or ())
    if tuple(feature.get("shape", ())) != (14,) or len(names) != 14 or set(names) != set(JOINT_NAMES):
        raise ValueError("actionにはjoint_0〜joint_13の14関節が必要です")
    fps = float(dataset.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"不正なデータセットFPS: {fps}")
    # HFの数値列だけを読む。エピソードを跨いだフレームは受け付けない。
    rows = dataset.hf_dataset.select_columns(["action", "episode_index", "frame_index"])
    if len(rows) == 0:
        raise ValueError(f"エピソード{episode_index}にフレームがありません")
    order = [names.index(name) for name in JOINT_NAMES]
    actions = []
    for i, row in enumerate(rows):
        if int(row["episode_index"]) != episode_index or int(row["frame_index"]) != i:
            raise ValueError("エピソード番号またはフレーム順序が不正です")
        action = np.asarray(row["action"], dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f"フレーム{i}のactionが不正です")
        actions.append(aloha_to_iloha(action[order]))
    return np.stack(actions), fps


async def move_to_first_action(robot: Iloha, target: np.ndarray) -> None:
    """記録開始姿勢までは既存の相対変化量制限で移動する。"""
    deadline = time.perf_counter() + 30.0
    while np.max(np.abs(robot.old_action - target)) > 1e-4:
        if time.perf_counter() >= deadline:
            raise TimeoutError("記録開始姿勢への移動がタイムアウトしました")
        await robot.async_send_action(target, use_relative=True, use_filter=False, use_unwrap=False)
        await asyncio.sleep(1.0 / 60.0)


async def replay_episode(
    robot: Iloha, actions: np.ndarray, fps: float, *, base_speed: float = 1.0,
    max_speedup: float = 1.0, gripper_margin: float = 0.5,
    speedup_distance: float = 2.0, gripper_threshold: float = 1e-4,
) -> int:
    """指令値を変えず、送信間隔を調整して全フレームを再生する。"""
    intervals = build_replay_intervals(
        actions, fps, base_speed, max_speedup, gripper_margin, speedup_distance, gripper_threshold,
    )
    await move_to_first_action(robot, actions[0])
    for index, action in enumerate(actions):
        start = time.perf_counter()
        await robot.async_send_action(action, use_relative=False, use_filter=False, use_unwrap=False)
        if (index + 1) % 30 == 0:
            print(f"再生中: {index + 1}/{len(actions)}フレーム")
        # 処理が遅れた場合もフレームを飛ばしたり、まとめて送信したりしない。
        # 最終指令も最後の周期分保持する（1フレームのみならベース周期）。
        interval = intervals[min(index, len(intervals) - 1)] if len(intervals) else 1.0 / fps / base_speed
        await asyncio.sleep(max(0.0, interval - (time.perf_counter() - start)))
    return len(actions)


async def reset_robot_to_home(robot: Iloha, init=True):
    """
    ロボットを初期位置に戻す（iloha_server.pyのhandle_reset_requestと同じロジック）
    """
    print("ロボットを初期位置に戻しています...")

    home_action = robot.old_action.copy()
    home_action[3:7] = 0.0
    home_action[10:14] = 0.0
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)

    home_action = np.zeros_like(home_action)
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)
    
    print("初期位置復帰完了")


class SingleArmTeleoperation(asyncio.DatagramProtocol):
    """catch_serverと同じUnity通信形式で、指定した片腕だけを操作する。"""

    def __init__(self, robot: Iloha, color: str):
        if color not in ("red", "blue"):
            raise ValueError("colorにはredまたはblueを指定してください")
        self.robot = robot
        self.active_arm = slice(0, 7) if color == "red" else slice(7, 14)
        self.latest_action = None
        self.first_action_time = None
        self.connected = False

    def datagram_received(self, data, addr):
        # mode=1 + little-endian float32 x 14。モードと座標変換は既存サーバーと共通。
        if len(data) < 57 or data[0] != 1:
            return
        angles = np.array(struct.unpack_from("<14f", data, 1), dtype=np.float32)
        if not np.isfinite(angles).all():
            return
        angles[0] += np.pi / 2
        angles[7] -= np.pi / 2
        for offset in (0, 7):
            angles[offset + 1] -= np.pi / 2
            angles[offset + 2] = -angles[offset + 2] - np.pi / 2
            angles[offset + 3:offset + 6] *= -1
        action = np.zeros(14, dtype=np.float32)
        action[self.active_arm] = angles[self.active_arm]
        self.latest_action = action

    async def control_robot(self):
        while True:
            start = time.perf_counter()
            if self.latest_action is not None:
                if self.first_action_time is None:
                    self.first_action_time = start
                action = self.latest_action.copy()
                # 既存サーバー同様、開始3秒間と大きな目標変化では相対制限を使う。
                use_relative = (
                    start - self.first_action_time < 3.0
                    or np.max(np.abs(action - self.robot.old_action)) > 0.2
                )
                await self.robot.async_send_action(
                    action, use_relative=use_relative, use_filter=not use_relative,
                )
            await asyncio.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - start)))

    async def websocket_handler(self, websocket):
        if self.connected:
            await websocket.close(code=1013, reason="別のクライアントが操作中です")
            return
        self.connected = True
        transport = None
        control_task = None
        try:
            data = json.loads(await websocket.recv())
            port = data.get("joint_send_port")
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("joint_send_portには1〜65535の整数が必要です")
            transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                lambda: self, local_addr=("0.0.0.0", port),
            )
            async def control_connection():
                try:
                    await self.control_robot()
                except Exception as exc:
                    print(f"ロボット制御エラー: {exc}")
                    await websocket.close(code=1011, reason="ロボット制御エラー")

            control_task = asyncio.create_task(control_connection())
            await websocket.send(json.dumps({"status": "connected", "message": "接続情報受信完了"}))
            async for message in websocket:
                command = json.loads(message).get("command")
                if command == "reset_robot":
                    control_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await control_task
                    transport.close()
                    transport = None
                    await reset_robot_to_home(self.robot)
                    self.latest_action = None
                    self.first_action_time = None
                    transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                        lambda: self, local_addr=("0.0.0.0", port),
                    )
                    control_task = asyncio.create_task(control_connection())
                    response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
                elif command == "teleoperation":
                    response = {"status": "teleoperation_mode", "message": "片腕の遠隔操作が有効です"}
                else:
                    response = {"status": "error", "message": f"未対応のコマンド: {command}"}
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if transport is not None:
                transport.close()
            if control_task is not None:
                control_task.cancel()
                with suppress(asyncio.CancelledError):
                    await control_task
            try:
                await reset_robot_to_home(self.robot)
            finally:
                self.latest_action = None
                self.first_action_time = None
                self.connected = False

    async def run(self, port: int):
        async with websockets.serve(self.websocket_handler, "0.0.0.0", port):
            print(f"片腕の遠隔操作を開始: WebSocketポート{port}。Unityからの接続待機中 (Ctrl+Cで終了)")
            await asyncio.Future()


async def main(args):
    actions, fps = load_episode(args.dataset_path, args.episode_index)
    speed_options = dict(
        base_speed=args.base_speed, max_speedup=args.max_speedup,
        gripper_margin=args.gripper_margin, speedup_distance=args.speedup_distance,
        gripper_threshold=args.gripper_threshold,
    )
    intervals = build_replay_intervals(actions, fps, **speed_options)
    print(f"エピソード{args.episode_index}: {len(actions)}フレーム、{fps:g} FPS")
    duration = intervals.sum() + (intervals[-1] if len(intervals) else 1.0 / fps / args.base_speed)
    print(f"ベース速度: {args.base_speed:g}倍、追加倍率上限: {args.max_speedup:g}倍")
    print(f"再生予定時間: {duration:.2f}秒（開始姿勢への移動・原点復帰・通信遅延を除く）")
    if args.dry_run:
        print("データセット検証完了（ロボット接続なし）")
        return

    # 接続時の原点復帰から、根元の原点を左ID1は-90度、右ID4は+90度に設定。
    left_settings.Robstride01Constants.DEFAULT_OFFSET = -np.pi / 2
    right_settings.Robstride01Constants.DEFAULT_OFFSET = np.pi / 2
    # 2026-09-11に全開状態で実測した位置。グリッパー指令0を全開にする。
    left_settings.Dynamixel04Constants.CONTROL_PARAMS.offset = 2394  # ID4: 210.41015625度
    right_settings.Dynamixel04Constants.CONTROL_PARAMS.offset = 2365  # ID8: 207.861328125度
    config = IlohaConfig(
        left_robstride_port="auto",
        left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
        right_robstride_port="auto",
        right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
        max_relative_target_1=0.03,
        max_relative_target_2=0.01,
        max_relative_target_3=0.01,
        max_relative_target_4=0.03,
        max_relative_target_5=0.01,
        max_relative_target_6=0.03,
        current_limit_robstride={1: 4.0, 2: 16.0, 3: 4.0, 4: 4.0, 5: 16.0, 6: 4.0},
        current_limit_gripper_R=0.5, # 0.3
        current_limit_gripper_L=0.5, # 0.3
    )
    robot = Iloha(config, debug=False)
    try:
        await robot.connect()
        await reset_robot_to_home(robot)
        count = await replay_episode(robot, actions, fps, **speed_options)
        print(f"再生完了: {count}フレーム")
        await reset_robot_to_home(robot, init=False)
        print(f"操作対象: {args.color}（{'左' if args.color == 'red' else '右'}アーム）")
        await SingleArmTeleoperation(robot, args.color).run(args.websocket_port)
    finally:
        await robot.disconnect()
        print("ロボット切断完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="記録済みエピソードをIlohaで再生後、片腕の遠隔操作を開始")
    parser.add_argument("--color", required=True, choices=("red", "blue"), help="遠隔操作するアーム: red=左、blue=右")
    parser.add_argument("--websocket_port", type=int, default=8080, help="遠隔操作のWebSocketポート（既定: 8080）")
    parser.add_argument("--dataset_path", required=True, help="記録データセットのパス（例: datasets/iloha-0）")
    parser.add_argument("--episode_index", type=int, default=0, help="再生するエピソード番号（0始まり）")
    parser.add_argument("--dry_run", action="store_true", help="ロボットに接続せずデータセットを検証")
    parser.add_argument("--base_speed", type=float, default=1.0, help="全体のベース速度倍率（正数、既定: 1）")
    parser.add_argument("--max_speedup", type=float, default=2.0,
                        help="グリッパー動作から離れた区間の追加倍率上限（1以上、既定: 1=無効）")
    parser.add_argument("--gripper_margin", type=float, default=0.5,
                        help="グリッパー動作前後でベース速度を保つ記録時間の秒数（既定: 0.5）")
    parser.add_argument("--speedup_distance", type=float, default=1.0,
                        help="余白の外側から追加倍率上限に達するまでの記録時間の秒数（正数、既定: 2）")
    parser.add_argument("--gripper_threshold", type=float, default=1e-4,
                        help="動作と判定するグリッパー指令のフレーム間変化量の閾値（0以上、既定: 1e-4）")
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        print("\nゲームを終了しました")

# uv run iloha_catch_game.py --dataset_path datasets/iloha-best-edited2 --color blue
