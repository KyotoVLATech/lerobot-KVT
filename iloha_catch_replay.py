#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iloha_catch_server.pyで記録したエピソードの送信済みアクションを再生する。"""
import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

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
        # RobstrideのIDは左腕が1・2・3、右腕が4・5・6。根元から先端の順。
        # 機種はID1・4がRS03（定格20Nm/13Apk、ピーク60Nm/43Apk）、
        # ID2・5がRS06（定格11Nm/14.3Apk、ピーク36Nm/57Apk）、
        # ID3・6がRS00（定格5Nm/4.7Apk、ピーク14Nm/15.5Apk）。
        # vel_max・acc_setの工場出荷値は3機種とも10rad/s・10rad/s^2。
        # current_limit_robstride={1: 4.0, 2: 16.0, 3: 4.0, 4: 4.0, 5: 16.0, 6: 4.0},
        current_limit_robstride={1: 10.0, 2: 12.0, 3: 4.0, 4: 10.0, 5: 12.0, 6: 4.0},
        vel_max_robstride={1: np.pi, 2: np.pi, 3: np.pi, 4: np.pi, 5: np.pi, 6: np.pi},
        # acc_set_robstride={1: np.pi/2, 2: np.pi/2, 3: np.pi/2, 4: np.pi/2, 5: np.pi/2, 6: np.pi/2},
        acc_set_robstride={1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10},
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
    finally:
        await robot.disconnect()
        print("ロボット切断完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="記録済みエピソードをIlohaで再生")
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
        print("\n再生を中断しました")

# uv run iloha_catch_replay.py --dataset_path datasets/iloha-best-edited2