#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-Time EXPO-FT オンラインRL用 Iloha ロールアウトクライアント

構成（expo-ft の client/run_client.py と同じサーバ・クライアント方式）:
  - Learner（GPUマシン, libs/expo-ft の .venv）:
      libs/expo-ft/scripts/iloha_towel/run_server.sh
      train_pi_robo.py + RealTimeEXPOFTLearner。RTC-SFT 済み pi0.5 を読み込み、
      推論・リプレイバッファ・更新をすべて担当し、client_port で待ち受ける。
  - このスクリプト（ロボットPC, lerobot-KVT の環境）:
      Learner に接続し、環境操作（create_env / reset / step / get_observation /
      get_info_for_step / render）をロボット実機で実行して返す。

観測は RTC-SFT の学習データ（libs/expo-ft/scripts/iloha/export_lerobot_to_hdf5.py）と同じ形式:
  cam_*_image: uint8 [3, 224, 224]（cam_high は iloha_eval.py と同じ 640x480 クロップ後、
               OpenPI と同じ resize_with_pad）
  state:       iloha_to_aloha(モータ実測の関節角)（ALOHA座標 14次元）。
               --state_source command で直前の指令値(old_action)に切り替え可能
  prompt:      タスク指示文
行動は ALOHA 座標の絶対関節角 14次元。

成功・失敗判定はキーボード入力（エピソード中に s+Enter で成功、f+Enter で失敗）。
--episode_time_s を超えると失敗として終了する。

使い方:
  # 1. GPUマシンで Learner を起動（libs/expo-ft で）
  bash scripts/iloha_towel/run_server.sh
  # 2. ロボットPCでこのクライアントを起動
  uv run --extra pi iloha_rl.py --host <GPUマシンのIP> --port 8104
"""

import argparse
import asyncio
import functools
import logging
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import msgpack
import numpy as np
import websockets
import websockets.asyncio.client as ws_client

from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from iloha_mapping import JOINT_NAMES, aloha_to_iloha, iloha_to_aloha
from iloha_eval import (
    ABSOLUTE_MODE_DELTA_THRESHOLD,
    CAMERA_CONFIGS,
    RELATIVE_WARMUP_SECONDS,
    STANDBY_MOTOR_DISABLE_DELAY_SECONDS,
    TASK,
    capture_observation,
    get_next_dataset_number,
    initialize_cameras,
    reset_robot_to_home,
    resume_motors_from_standby,
    stop_motors_for_standby,
)


# RTC-SFT 学習時と同じ（Quality ラベルは学習時に除去済み）
DEFAULT_PROMPT = TASK
MODEL_IMAGE_SIZE = 224
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


# ---------------------------------------------------------------------------
# openpi_client.msgpack_numpy と互換のシリアライザ（ロボットPCに OpenPI を入れずに済むよう同梱）
# ---------------------------------------------------------------------------
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# OpenPI (openpi_client.image_tools.resize_with_pad) と同一のリサイズ
# 学習データのエクスポートもこの処理なので、実機観測と学習分布が一致する
# ---------------------------------------------------------------------------
def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    cur_h, cur_w = img.shape[0], img.shape[1]
    if cur_h == height and cur_w == width:
        return np.asarray(img, dtype=np.uint8)
    ratio = max(cur_w / width, cur_h / height)
    resized_w = int(cur_w / ratio)
    resized_h = int(cur_h / ratio)
    resized = cv2.resize(np.asarray(img, dtype=np.uint8), (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = int((width - resized_w) / 2)
    pad_h = int((height - resized_h) / 2)
    out = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    out[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized
    return out


# ---------------------------------------------------------------------------
# キーボード入力（標準入力を1スレッドで読み、エピソード判定とEnter待ちで共有する）
# ---------------------------------------------------------------------------
class KeyboardInput:
    def __init__(self):
        self._lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in sys.stdin:
            self._lines.put(line.strip().lower())

    def clear(self):
        while not self._lines.empty():
            self._lines.get_nowait()

    def poll(self) -> Optional[str]:
        """未処理の入力行を1つ返す（無ければ None）。ブロックしない"""
        try:
            return self._lines.get_nowait()
        except queue.Empty:
            return None

    async def wait_line(self, prompt: str) -> str:
        print(prompt, flush=True)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._lines.get)


# ---------------------------------------------------------------------------
# 実機環境
# ---------------------------------------------------------------------------
class IlohaRLEnv:
    """expo-ft の DroidEnv と同じ契約（reset / step / get_observation / get_info_for_step）を
    Iloha 実機で提供する。reward は成功時のみ 1、done 時は mask=0。"""

    def __init__(self, robot: Iloha, keyboard: KeyboardInput, args, dataset: Optional[LeRobotDataset] = None):
        self.robot = robot
        self.keyboard = keyboard
        self.args = args
        self.dataset = dataset
        self.prompt = args.task
        self.max_episode_steps = int(round(args.episode_time_s * args.fps))
        self.state_names = JOINT_NAMES
        self.ds_features = self._dataset_features()

        self.episode_idx = 0
        self.num_success = 0
        self.motors_in_standby = False
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.steps = 0
        self.episode_start_t = None
        self.verdict: Optional[bool] = None  # True=成功, False=失敗, None=継続中
        self.last_obs_raw = None
        self.frames_saved = 0

    @staticmethod
    def _dataset_features() -> dict:
        features = {
            "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        }
        for key in CAMERA_CONFIGS.keys():
            features[f"observation.images.{key}"] = {
                "dtype": "video", "shape": (480, 640, 3), "names": ("height", "width", "channels"),
            }
        return features

    # --- episode lifecycle ------------------------------------------------
    async def reset(self) -> dict:
        if self.episode_idx > 0:
            self._finish_episode()
            await stop_motors_for_standby(self.robot, motor_disable_delay_s=self.args.standby_motor_disable_delay_s)
            self.motors_in_standby = True
        # 初回も含め、毎エピソード人が環境（タオル配置）を整えてから開始する
        self.keyboard.clear()
        await self.keyboard.wait_line(
            "環境をリセットしてください。準備ができたらEnterでエピソードを開始します（Ctrl+Cで終了）"
        )
        if self.motors_in_standby:
            await resume_motors_from_standby(self.robot)
            self.motors_in_standby = False
        await reset_robot_to_home(self.robot, init=self.episode_idx == 0)
        await asyncio.sleep(2.0)

        self._reset_episode_state()
        self.keyboard.clear()
        self.episode_idx += 1
        print(f"\n--- エピソード {self.episode_idx} 開始（最大{self.args.episode_time_s:.0f}秒）---")
        print("  判定: s+Enter=成功 / f+Enter=失敗（時間切れは失敗）")
        return await self.get_observation()

    def _finish_episode(self):
        result = "成功" if self.verdict else "失敗"
        if self.verdict:
            self.num_success += 1
        print(f"エピソード {self.episode_idx}: {result}（{self.steps}ステップ）"
              f" 累計成功 {self.num_success}/{self.episode_idx}")
        if self.dataset is not None:
            if self.frames_saved > 0:
                self.dataset.save_episode()
            else:
                self.dataset.clear_episode_buffer()

    # --- observation ------------------------------------------------------
    async def read_state(self) -> np.ndarray:
        """ALOHA座標の14次元状態。measured=モータ実測値、command=直前の指令値(old_action)"""
        if self.args.state_source == "measured":
            joints_iloha = await self.robot.async_read_joint_state()
        else:
            joints_iloha = self.robot.old_action
        return iloha_to_aloha(joints_iloha).astype(np.float32)

    async def get_observation(self) -> dict:
        raw = capture_observation(self.robot, self.state_names)
        state = await self.read_state()
        # 保存用の生観測も実際に Policy に渡した状態にそろえる
        for i, joint_name in enumerate(self.state_names):
            raw[joint_name] = float(state[i])
        self.last_obs_raw = raw
        obs = {
            f"{cam}_image": np.ascontiguousarray(
                resize_with_pad(raw[cam], MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE).transpose(2, 0, 1)
            )
            for cam in CAMERA_NAMES
        }
        obs["state"] = state
        obs["prompt"] = self.prompt
        return obs

    def get_info_for_step(self):
        if self.verdict is None:
            key = self.keyboard.poll()
            while key is not None and self.verdict is None:
                if key in ("s", "success"):
                    self.verdict = True
                elif key in ("f", "fail", "failure"):
                    self.verdict = False
                key = self.keyboard.poll()
            if self.verdict is None and self.steps >= self.max_episode_steps:
                print(f"エピソード時間（{self.args.episode_time_s}秒）に達しました → 失敗")
                self.verdict = False
        done = self.verdict is not None
        success = bool(self.verdict)
        reward = 1.0 if success else 0.0
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    # --- action -----------------------------------------------------------
    async def step(self, action) -> dict:
        action_aloha = np.asarray(action, dtype=np.float32)[:14]
        # Learner はプランが無いとき全0、無効時は全-1を送る。DROIDの速度指令なら無害だが、
        # 絶対関節角では遠い姿勢への急移動になるため現在姿勢を保持する。
        hold = (not np.isfinite(action_aloha).all()
                or np.all(action_aloha == 0.0)
                or np.allclose(action_aloha, -1.0))
        if hold:
            return {"executed_action": iloha_to_aloha(self.robot.old_action).astype(np.float64)}

        if self.episode_start_t is None:
            self.episode_start_t = time.perf_counter()
        elapsed = time.perf_counter() - self.episode_start_t

        action_iloha = aloha_to_iloha(action_aloha)
        max_delta = float(np.max(np.abs(action_iloha - self.robot.old_action)))
        use_relative = not self.args.disable_robot_relative_safety and (
            elapsed < self.args.relative_warmup_seconds or max_delta > self.args.absolute_mode_delta_threshold
        )
        await self.robot.async_send_action(action_iloha, use_relative=use_relative, use_filter=not use_relative)
        # 安全制限後に実際に送った値を学習に使う（データ収集時の action と同じ定義）
        executed = iloha_to_aloha(self.robot.old_action)
        self.steps += 1

        if self.dataset is not None and self.last_obs_raw is not None:
            observation_frame = build_dataset_frame(self.ds_features, self.last_obs_raw, prefix="observation")
            action_frame = build_dataset_frame(
                self.ds_features, {n: float(executed[i]) for i, n in enumerate(JOINT_NAMES)}, prefix="action"
            )
            self.dataset.add_frame({**observation_frame, **action_frame, "task": self.prompt})
            self.frames_saved += 1

        if self.steps % (self.args.fps * 5) == 0:
            print(f"  ステップ {self.steps}, 経過 {elapsed:.1f}秒")
        return {"executed_action": executed.astype(np.float64)}

    async def render(self) -> np.ndarray:
        if self.last_obs_raw is None:
            await self.get_observation()
        return np.asarray(self.last_obs_raw["cam_high"], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Learner との通信（expo-ft client/run_client.py と同じプロトコル）
# ---------------------------------------------------------------------------
async def serve_learner(websocket, env: IlohaRLEnv, args):
    logger = logging.getLogger(__name__)
    packer = Packer()
    try:
        sock = websocket.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.warning("TCP_NODELAY を設定できませんでした: %s", e)

    env_id = f"iloha_rl_{int(time.time())}"
    while True:
        try:
            request = unpackb(await websocket.recv())
        except websockets.exceptions.ConnectionClosed:
            return
        op = request.get("operation")
        try:
            if op == "create_env":
                # ロボットは起動時に接続済み。Learner の再接続時も同じ実機を使い回す
                response = {"status": "success", "env_id": env_id, "task_description": env.prompt}
            elif op == "reset":
                response = {"status": "success", "observation": await env.reset(), "done": False}
            elif op == "step":
                t0 = time.perf_counter()
                result = await env.step(request["action"])
                step_ms = (time.perf_counter() - t0) * 1000.0
                if step_ms >= args.step_timing_threshold_ms:
                    logger.warning("[timing] step %.1fms", step_ms)
                response = {"status": "success", "action": result["executed_action"].tolist(), "action_type": "policy"}
            elif op == "get_observation":
                obs = await env.get_observation()
                done, success, reward, mask = env.get_info_for_step()
                response = {"status": "success", "observation": obs, "done": bool(done),
                            "success": bool(success), "reward": float(reward), "mask": float(mask)}
            elif op == "get_info_for_step":
                done, success, reward, mask = env.get_info_for_step()
                response = {"status": "success", "done": bool(done), "success": bool(success),
                            "reward": float(reward), "mask": float(mask)}
            elif op == "render":
                response = {"status": "success", "frame": await env.render()}
            else:
                response = {"status": "error", "message": f"Unknown operation: {op}"}
        except Exception as e:
            logger.error("操作 %s でエラー: %s", op, e, exc_info=True)
            response = {"status": "error", "message": str(e)}
        await websocket.send(packer.pack(response))


async def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    keyboard = KeyboardInput()

    print("=" * 60)
    print("ロボットを初期化中...")
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
    await robot.connect()
    print("ロボット接続完了")
    await reset_robot_to_home(robot)

    print("=" * 60)
    print("カメラを初期化中...")
    cameras = initialize_cameras()
    if not cameras:
        print("エラー: カメラの初期化に失敗しました")
        await robot.disconnect()
        return
    robot.cameras = cameras

    dataset, video_encoding_manager = None, None
    if args.save_data:
        dataset_root = Path(args.output_root)
        dataset_name = f"iloha-rl-{get_next_dataset_number(dataset_root, prefix='iloha-rl-')}"
        dataset = LeRobotDataset.create(
            f"local/{dataset_name}", args.fps, root=dataset_root / dataset_name, robot_type="aloha",
            features=IlohaRLEnv._dataset_features(), use_videos=True, image_writer_processes=0,
            image_writer_threads=len(cameras), video_backend="pyav",
        )
        video_encoding_manager = VideoEncodingManager(dataset)
        video_encoding_manager.__enter__()
        print(f"ロールアウトを保存します: {dataset_root / dataset_name}")

    env = IlohaRLEnv(robot, keyboard, args, dataset=dataset)
    uri = f"ws://{args.host}:{args.port}"
    try:
        while True:
            try:
                print(f"Learner に接続中: {uri}")
                async with ws_client.connect(
                    uri, compression=None, max_size=None, ping_interval=None, ping_timeout=None, close_timeout=100,
                ) as websocket:
                    print("Learner に接続しました。Learner からの指示を待っています")
                    await serve_learner(websocket, env, args)
                print("Learner との接続が切れました。5秒後に再接続します")
            except (OSError, websockets.exceptions.WebSocketException) as e:
                print(f"Learner に接続できません（{e}）。5秒後に再試行します")
            await asyncio.sleep(5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n中断されました")
    finally:
        print("=" * 60)
        print("クリーンアップ中...")
        if video_encoding_manager:
            video_encoding_manager.__exit__(None, None, None)
        if dataset:
            dataset.finalize()
        for name, camera in cameras.items():
            try:
                camera.disconnect()
            except Exception as e:
                print(f"{name} 切断エラー: {e}")
        if not env.motors_in_standby:
            await reset_robot_to_home(robot, init=False)
        await robot.disconnect()
        print("ロボット切断完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time EXPO-FT オンラインRL用 Iloha ロールアウトクライアント")
    parser.add_argument("--host", type=str, required=True, help="Learner（GPUマシン）のホスト名/IP")
    parser.add_argument("--port", type=int, default=8104, help="Learner の client_port（デフォルト: 8104）")
    parser.add_argument("--task", type=str, default=DEFAULT_PROMPT,
                        help="タスク指示文。RTC-SFT の学習時と同じ文にすること")
    parser.add_argument("--fps", type=int, default=30, help="制御周波数（Learner の control_hz と一致させる）")
    parser.add_argument("--state_source", type=str, default="measured", choices=["measured", "command"],
                        help="Policyに渡すロボット状態。measured=モータ実測値（デフォルト）、"
                             "command=直前の指令値(old_action)。学習データ(iloha_server.py)は command で記録されている")
    parser.add_argument("--episode_time_s", type=float, default=30.0,
                        help="1エピソードの最大時間（秒）。超えると失敗扱い（デフォルト: 30）")
    parser.add_argument("--save_data", action="store_true", help="ロールアウトを LeRobotDataset として保存する")
    parser.add_argument("--output_root", type=str, default="datasets/rl", help="保存先ルート（デフォルト: datasets/rl）")
    parser.add_argument("--disable_robot_relative_safety", action="store_true",
                        help="初動・急変時のIloha相対制限安全制御を無効にする")
    parser.add_argument("--relative_warmup_seconds", type=float, default=RELATIVE_WARMUP_SECONDS)
    parser.add_argument("--absolute_mode_delta_threshold", type=float, default=ABSOLUTE_MODE_DELTA_THRESHOLD)
    parser.add_argument("--standby_motor_disable_delay_s", type=float, default=STANDBY_MOTOR_DISABLE_DELAY_SECONDS)
    parser.add_argument("--step_timing_threshold_ms", type=float, default=30.0,
                        help="step処理がこの時間を超えたら警告する（ms）")
    asyncio.run(main(parser.parse_args()))
