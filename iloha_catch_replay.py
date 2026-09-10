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


async def replay_episode(robot: Iloha, actions: np.ndarray, fps: float) -> int:
    """保存済み指令に再フィルタを掛けず、記録時の周期で全フレーム送信する。"""
    await move_to_first_action(robot, actions[0])
    for index, action in enumerate(actions):
        start = time.perf_counter()
        await robot.async_send_action(action, use_relative=False, use_filter=False, use_unwrap=False)
        if (index + 1) % 30 == 0:
            print(f"再生中: {index + 1}/{len(actions)}フレーム")
        # 処理が遅れた場合もフレームを飛ばしたり、まとめて送信したりしない。
        await asyncio.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - start)))
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
    print(f"エピソード{args.episode_index}: {len(actions)}フレーム、{fps:g} FPS")
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
        current_limit_gripper_R=0.3,
        current_limit_gripper_L=0.3,
    )
    robot = Iloha(config, debug=False)
    try:
        await robot.connect()
        await reset_robot_to_home(robot)
        count = await replay_episode(robot, actions, fps)
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
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        print("\n再生を中断しました")
