#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iloha 実機での オンラインRL(RL-100 diffusion-PPO) デプロイスクリプト.

`iloha_eval.py`（実機の観測取得・アクション送出・安全制御）と RL-100 の
オンライン policy-gradient 更新（`rl_100.unidpg.uni_ppo.BehaviorProximalPolicyOptimization`）
を組み合わせ、オフラインRLで学習したチェックポイントから開始して、実機ロールアウトで
policy を追加学習する。2D RGB(3カメラ) 経路のみ。深度は使わない。

パイプライン全体での位置づけ:
    (1) iloha_to_rl100_zarr.py    LeRobot v3.0 → RL-100 zarr へ変換
    (2) scripts/iloha/train_sushi_offline.sh   BC(模倣学習) + オフラインRL
    (3) 本スクリプト              実機オンラインRL / ロールアウト収集(データフライホイール)

報酬設計（人手ラベル・疎な終端報酬）:
    タオル折り(sushi)には AprilTag 等の自動報酬源が無い。各エピソード終端で
    オペレータがキー入力し、成功なら +terminal_reward、失敗なら 0 を終端遷移に与える。
    エピソード途中の報酬は 0（疎な終端報酬）。--reward_fn で外部成功判定器も差し込める。

実行環境:
    RL-100 の学習環境(conda: rl100 等) かつ lerobot(Iloha実機ドライバ) の両方が import できる
    環境で実行する。RL-100 を PYTHONPATH に通しておくこと:
        export PYTHONPATH=/path/to/libs/RL-100/RL-100:$PYTHONPATH

安全のためのモード:
    --collect_only  … PPO更新を行わず、offlineRL policy を実機で回して人手ラベル付き
                       ロールアウトを .npz 保存するだけ（RL-100 のデータフライホイール相当）。
                       実機ループの検証はまずこのモードで行うこと。
    （既定）        … ロールアウトを batch_size 個ためるごとに diffusion-PPO 更新を実行。

使用例:
    # まず収集のみで実機ループを検証
    python iloha_online_rl.py --collect_only \
        --offline_ckpt libs/RL-100/RL-100/data/outputs_2d_chunk/sushi-rl100-run0_seed100/.../best \
        --zarr libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr

    # オンラインRL 本番
    python iloha_online_rl.py \
        --offline_ckpt <best_dir> --zarr <zarr> --task "fold the towel"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- RL-100 側 ----
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

# ---- lerobot / Iloha 実機側 ----
from iloha_camera_utils import capture_camera_observation
from iloha_mapping import JOINT_NAMES, aloha_to_iloha, iloha_to_aloha
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.iloha import Iloha, IlohaConfig

# =====================================================================================
# 定数（iloha_eval.py と揃える。深度は使わない -> use_depth=False）
# =====================================================================================
CAMERA_MAX_FRAME_AGE_MS = 250
CAM_HIGH_CROP_SIZE = (480, 640)   # height, width（cam_high の中央下部を切り出す）
IMAGE_HW = (240, 320)             # zarr / RL-100 cloth スキーマに合わせた保存解像度 (H, W)
RELATIVE_WARMUP_SECONDS = 3.0
ABSOLUTE_MODE_DELTA_THRESHOLD = 0.2

# LeRobot カメラ名 -> RL-100 cloth スキーマの obs キー
CAM_TO_RGBKEY = {
    "cam_high": "rgb_head",
    "cam_left_wrist": "rgb_left_hand",
    "cam_right_wrist": "rgb_right_hand",
}
RGB_KEYS = list(CAM_TO_RGBKEY.values())

CAMERA_CONFIGS = {
    "cam_high": {"serial_number_or_name": "146222252104", "width": 1280, "height": 720, "fps": 30, "use_depth": False},
    "cam_left_wrist": {"serial_number_or_name": "341522301205", "width": 640, "height": 480, "fps": 30, "use_depth": False},
    "cam_right_wrist": {"serial_number_or_name": "029522250086", "width": 640, "height": 480, "fps": 30, "use_depth": False},
}

# オフラインRL(chunk) と一致させる必要があるアーキ関連の override。
# train_policy_image_unet_chunk_two_stage.sh と同じ値にしてある。ズレるとチェックポイントの
# 重みが正しく load できないので、オフライン学習側を変えたらここも合わせること。
ONLINE_ARCH_OVERRIDES = [
    "online=True", "offline=False", "eval=False",
    "n_obs_steps=3", "n_action_steps=16", "horizon=18",
    "num_inference_steps=10",
    "feature_type='2D'", "use_agent_pos=True",
    "policy._target_=rl_100.policy.rl100_2d.RL1002D",
    "policy.use_visual=True", "policy.scheduler_type='ddim'",
    "policy.ddim_noise_scheduler.num_train_timesteps=100",
    "policy.model=dp3", "policy.act=mish", "policy.mlp_policy_depth=3",
    "policy.down_dims=[256,512,1024]", "policy.img_shape=[3,224,224]",
    "policy.use_aug=True",
    "encoder_output_dim=64", "encoder_type='resnet'",
    "encoders.resnet.share_rgb_model=False", "encoders.resnet.rgb_model.weights='r3m'",
    "use_recon=True", "use_vib=True", "dynamics_type='diffusion'",
    "chunk_as_single_action=True",
    "critic.q_hidden_dim=1024", "critic.v_hidden_dim=512",
    "critic.omega=0.9", "critic.gamma=0.997",
    "++task.env_runner.with_pointcloud=False",
    "task.env_runner.fake_env=True",
    # 本スクリプトの value head は policy の latent 特徴を入力にする(モダリティ非依存):
    "ppo.share_encoder=True", "ppo.fix_encoder=True",
]


# =====================================================================================
# 観測取得
# =====================================================================================
def _resize_hwc(img: np.ndarray, hw=IMAGE_HW) -> np.ndarray:
    import cv2
    h, w = hw
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.uint8)


def _crop_cam_high(image: np.ndarray) -> np.ndarray:
    ch, cw = CAM_HIGH_CROP_SIZE
    h, w = image.shape[:2]
    top = h - ch
    left = (w - cw) // 2
    return np.ascontiguousarray(image[top:top + ch, left:left + cw])


def capture_iloha_obs(robot: Iloha) -> dict:
    """3カメラRGB(240x320 HWC uint8) と ALOHA座標の14次元状態を1ステップ分返す。"""
    cam_obs = capture_camera_observation(robot.cameras, CAMERA_MAX_FRAME_AGE_MS)
    out = {}
    for cam, rgbkey in CAM_TO_RGBKEY.items():
        img = cam_obs[cam]
        if cam == "cam_high":
            img = _crop_cam_high(img)
        out[rgbkey] = _resize_hwc(img)
    aloha_state = iloha_to_aloha(robot.get_measured_state(max_age_s=CAMERA_MAX_FRAME_AGE_MS / 1000))
    out["agent_pos"] = np.asarray(aloha_state, dtype=np.float32)
    return out


# =====================================================================================
# 正規化器（zarr の低次元 state/action からのみ構築。画像はエンコーダ側で処理するので不要）
# rl_100.dataset.cloth.Cloth.get_normalizer と同じ（use_velocity=True 相当: state==agent_pos）
# =====================================================================================
def build_lowdim_normalizer_from_zarr(zarr_path: str):
    import zarr
    from rl_100.model.common.normalizer import LinearNormalizer

    root = zarr.open(zarr_path, mode="r")
    data = {
        "action": root["data"]["action"][:],
        "agent_pos": root["data"]["state"][:],
        "next_action": root["data"]["next_action"][:],
        "next_agent_pos": root["data"]["next_state"][:],
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data=data, last_n_dims=1, mode="limits")
    return normalizer


# =====================================================================================
# マルチビュー用オンラインバッファ（RL-100 の online_buffer.ReplayBuffer は image+point_cloud
# 固定のため、rgb_head/left/right + agent_pos に対応した最小版を用意する）
# =====================================================================================
class MultiViewOnlineBuffer:
    def __init__(self, capacity: int, num_inference_steps: int, shape_info: dict, device):
        self.capacity = capacity
        self.device = device
        self.count = 0
        act_shape = tuple(shape_info["action"])                    # (n_action_steps, action_dim)
        self.obs_keys = list(shape_info["obs"].keys())             # rgb_* + agent_pos
        self.obs = {k: np.zeros((capacity,) + tuple(shape_info["obs"][k]), dtype=np.float32) for k in self.obs_keys}
        self.next_obs = {k: np.zeros((capacity,) + tuple(shape_info["obs"][k]), dtype=np.float32) for k in self.obs_keys}
        self.action = np.zeros((capacity, num_inference_steps + 1) + act_shape, dtype=np.float32)     # all_x
        self.a_logprob = np.zeros((capacity, num_inference_steps) + act_shape, dtype=np.float32)      # all_logprob
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.dw = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, obs, all_x, a_logprob, reward, next_obs, done, dw):
        i = self.count
        for k in self.obs_keys:
            self.obs[k][i] = np.asarray(obs[k], dtype=np.float32)
            self.next_obs[k][i] = np.asarray(next_obs[k], dtype=np.float32)
        self.action[i] = np.asarray(all_x, dtype=np.float32)
        self.a_logprob[i] = np.asarray(a_logprob, dtype=np.float32)
        self.reward[i] = reward
        self.done[i] = float(done)
        self.dw[i] = float(dw)
        self.count += 1

    def reset(self):
        self.count = 0

    def numpy_to_tensor(self):
        n = self.count
        t = lambda x: torch.from_numpy(x[:n]).float().to(self.device)
        obs = {k: t(self.obs[k]) for k in self.obs_keys}
        next_obs = {k: t(self.next_obs[k]) for k in self.obs_keys}
        return obs, t(self.action), t(self.a_logprob), t(self.reward), next_obs, t(self.dw), t(self.done)


# =====================================================================================
# Iloha 実機 gym 環境（RL-100 の MultiStepWrapper でラップして使う）
# =====================================================================================
class IlohaGymEnv:
    """gym 互換の最小実機環境。

    reset() -> obs dict {rgb_head, rgb_left_hand, rgb_right_hand (H,W,3 uint8), agent_pos (14,)}
    step(action) -> (obs, reward, done, info)
        action: 1タイムステップの 14次元 ALOHA アクション
        reward: 途中は 0、終端(操作者ラベル)で terminal_reward or 0
        done  : max_steps 到達で True（そのとき人手ラベルを取得）
        info  : {'is_success': bool}
    """

    def __init__(self, robot: Iloha, loop, max_steps: int, fps: int,
                 terminal_reward: float, reward_fn=None,
                 use_relative_safety: bool = True):
        import gym
        from gym import spaces

        self.robot = robot
        self.loop = loop
        self.max_steps = max_steps
        self.fps = fps
        self.terminal_reward = terminal_reward
        self.reward_fn = reward_fn
        self.use_relative_safety = use_relative_safety

        self._step = 0
        self._t0 = 0.0
        self._last_obs = None

        h, w = IMAGE_HW
        obs_spaces = {k: spaces.Box(0, 255, (h, w, 3), dtype=np.uint8) for k in RGB_KEYS}
        obs_spaces["agent_pos"] = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)

    # --- helpers ---
    def _send(self, action_aloha: np.ndarray, elapsed: float):
        action_iloha = aloha_to_iloha(np.asarray(action_aloha, dtype=np.float32))
        prev = self.robot.old_action.copy()
        max_delta = float(np.max(np.abs(action_iloha - prev)))
        use_rel = self.use_relative_safety and (
            elapsed < RELATIVE_WARMUP_SECONDS or max_delta > ABSOLUTE_MODE_DELTA_THRESHOLD
        )
        self.loop.run_until_complete(
            self.robot.async_send_action(action_iloha, use_relative=use_rel, use_filter=not use_rel)
        )

    def reset(self):
        self._step = 0
        self._t0 = time.perf_counter()
        self._last_obs = capture_iloha_obs(self.robot)
        return self._last_obs

    def step(self, action):
        loop_start = time.perf_counter()
        elapsed = loop_start - self._t0
        self._send(np.asarray(action, dtype=np.float32), elapsed)

        # FPS 制御
        dt = time.perf_counter() - loop_start
        sleep_s = 1.0 / self.fps - dt
        if sleep_s > 0:
            time.sleep(sleep_s)

        obs = capture_iloha_obs(self.robot)
        self._last_obs = obs
        self._step += 1

        done = self._step >= self.max_steps
        reward = 0.0
        is_success = False
        if done:
            is_success = self._label_terminal(obs)
            reward = self.terminal_reward if is_success else 0.0
        return obs, reward, done, {"is_success": is_success}

    def _label_terminal(self, obs) -> bool:
        if self.reward_fn is not None:
            try:
                return bool(self.reward_fn(obs))
            except Exception as e:
                print(f"[warn] reward_fn 失敗 -> 人手ラベルにフォールバック: {e}")
        ans = input("エピソード終了。成功なら [数字], 失敗なら [文字] を入力: ").strip()
        return len(ans) > 0 and ans[0].isdigit()


# =====================================================================================
# RL-100 のロード
# =====================================================================================
def compose_cfg(config_dir: str, config_name: str, task: str, extra_overrides: list[str]):
    overrides = [f"task={task}"] + ONLINE_ARCH_OVERRIDES + list(extra_overrides)
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    # TrainDP3Workspace.__init__ と同じく inference steps を揃える
    with open_dict(cfg):
        cfg.ppo.num_inference_steps = cfg.policy.num_inference_steps
    return cfg


class ValueHead(nn.Module):
    """policy.obs2latent の潜在特徴を入力にとる状態価値 V(s) の MLP ヘッド。"""

    def __init__(self, in_dim: int, hidden: int = 512, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_shape_info(cfg):
    h, w = IMAGE_HW
    n_obs = int(cfg.n_obs_steps)
    return {
        "obs": {
            "rgb_head": (n_obs, h, w, 3),
            "rgb_left_hand": (n_obs, h, w, 3),
            "rgb_right_hand": (n_obs, h, w, 3),
            "agent_pos": (n_obs, 14),
        },
        "action": (int(cfg.n_action_steps), 14),
    }


def load_rl100(cfg, offline_ckpt: str, normalizer, device):
    from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization

    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    model.to(device)
    model.eval()

    unio4 = BehaviorProximalPolicyOptimization(
        policy=model, device=torch.device(device),
        policy_lr=cfg.unio4.bppo_lr, clip_ratio=cfg.unio4.clip_ratio,
        entropy_weight=cfg.unio4.entropy_weight, decay=cfg.unio4.decay,
        omega=cfg.unio4.omega, batch_size=cfg.unio4.bppo_batch_size,
        is_iql=cfg.critic.is_iql, temperature=cfg.unio4.temperature,
        ratio_strategy=cfg.unio4.ratio_strategy, top_k=cfg.unio4.top_k,
        num_inference_steps=cfg.policy.num_inference_steps,
        fix_encoder=cfg.unio4.fix_encoder, cfg=cfg,
    )
    # オフラインRLの最良チェックポイント(model.pt / encoder.pt)を読み込む
    print(f"loading offline checkpoint: {offline_ckpt}")
    unio4.load(offline_ckpt)
    unio4._policy.set_normalizer(normalizer)
    unio4._policy.to(device)
    return model, unio4


# =====================================================================================
# メイン
# =====================================================================================
def make_iloha_config() -> IlohaConfig:
    return IlohaConfig(
        left_robstride_port="auto",
        left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
        right_robstride_port="auto",
        right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
        max_relative_target_1=0.03, max_relative_target_2=0.01, max_relative_target_3=0.01,
        max_relative_target_4=0.03, max_relative_target_5=0.01, max_relative_target_6=0.03,
        current_limit_robstride={1: 4.0, 2: 16.0, 3: 4.0, 4: 4.0, 5: 16.0, 6: 4.0},
        current_limit_gripper_R=0.3, current_limit_gripper_L=0.3,
    )


async def reset_to_home(robot: Iloha):
    home = robot.old_action.copy()
    home[3:7] = 0.0
    home[10:14] = 0.0
    await robot.async_send_action(home, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)
    await robot.async_send_action(np.zeros_like(home), use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)


def init_cameras() -> dict:
    cam_cfgs = {name: RealSenseCameraConfig(**cfg) for name, cfg in CAMERA_CONFIGS.items()}
    cams = make_cameras_from_configs(cam_cfgs)
    for name, cam in cams.items():
        print(f"connecting {name} ...")
        cam.connect(warmup=True)
        time.sleep(1.0)
    return cams


def obs_to_policy_input(stacked_obs: dict, device) -> dict:
    """MultiStepWrapper が返す (n_obs_steps, ...) obs にバッチ次元を足して policy 入力にする。"""
    out = {}
    for k in RGB_KEYS:
        x = torch.from_numpy(np.asarray(stacked_obs[k])).float().to(device)
        out[k] = x.unsqueeze(0)          # (1, To, H, W, 3)
    ap = torch.from_numpy(np.asarray(stacked_obs["agent_pos"])).float().to(device)
    out["agent_pos"] = ap.unsqueeze(0)   # (1, To, 14)
    return out


def run(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 1) RL-100 config を合成
    cfg = compose_cfg(args.config_dir, args.config_name, args.task_name, args.override or [])
    n_action_steps = int(cfg.n_action_steps)
    n_obs_steps = int(cfg.n_obs_steps)
    num_inf = int(cfg.policy.num_inference_steps)
    max_steps = max(1, int(args.episode_time_s * args.fps) // n_action_steps)  # チャンク数
    print(f"n_obs={n_obs_steps} n_action={n_action_steps} num_inference={num_inf} chunks/episode={max_steps}")

    # 2) 正規化器 + policy + unio4 を用意
    normalizer = build_lowdim_normalizer_from_zarr(args.zarr)
    model, unio4 = load_rl100(cfg, args.offline_ckpt, normalizer, device)

    # 3) 実機・カメラを接続（同期スクリプトから使うので専用 event loop を持つ）
    loop = asyncio.new_event_loop()
    robot = Iloha(make_iloha_config(), debug=False)
    loop.run_until_complete(robot.connect())
    loop.run_until_complete(reset_to_home(robot))
    robot.cameras = init_cameras()
    print("robot + cameras ready")

    # 4) 環境を MultiStepWrapper でラップ（step_online のチャンク実行・報酬集約を使う）
    from rl_100.gym_util.multistep_wrapper_real import MultiStepWrapper
    base_env = IlohaGymEnv(
        robot, loop, max_steps=max_steps, fps=args.fps,
        terminal_reward=args.terminal_reward, reward_fn=None,
        use_relative_safety=not args.disable_relative_safety,
    )
    env = MultiStepWrapper(
        base_env, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        max_episode_steps=max_steps, reward_agg_method="sum", gamma=cfg.gamma,
    )

    # 5) オンラインRL用の value head を構築し transfer2online（collect_only では不要）
    online = not args.collect_only
    if online:
        with torch.no_grad():
            dummy = env.reset()
            feat = unio4._policy.obs2latent(obs_to_policy_input(dummy, device), training=False)
        latent_dim = int(feat.shape[-1])
        value_net = ValueHead(latent_dim, hidden=int(cfg.critic.v_hidden_dim), depth=2).to(device)
        print(f"value head input dim = {latent_dim}")
        unio4.transfer2online(critic=value_net, dynamics=None, cfg=cfg)

    shape_info = build_shape_info(cfg)
    buffer = MultiViewOnlineBuffer(int(cfg.ppo.batch_size), num_inf, shape_info, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=True)

    total_steps = 0
    update_num = 0
    ep_idx = 0
    try:
        while total_steps < args.max_train_steps:
            ep_idx += 1
            print(f"\n===== episode {ep_idx} (total chunk-steps {total_steps}) =====")
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_records = []
            while not done:
                obs_in = obs_to_policy_input(obs, device)
                with torch.no_grad():
                    if args.collect_only:
                        # 決定論的評価ロールアウト（データ収集用）
                        out = unio4._policy.predict_action(obs_in, deterministic=True)
                        action = out["action"]
                        all_x = torch.zeros((num_inf + 1, action.shape[1], action.shape[2]), device=device)
                        a_logprob = torch.zeros((num_inf, action.shape[1], action.shape[2]), device=device)
                    else:
                        # 確率的サンプリング + 各ノイズ除去ステップの logprob（PPO用）
                        action, all_x, a_logprob = unio4._policy.all_step_action_logprob(
                            obs_in, fix_encoder=cfg.ppo.fix_encoder
                        )

                act_np = action.squeeze(0).detach().cpu().numpy()      # (n_action_steps, 14) ALOHA
                next_obs, reward, done_step, info = env.step_online(act_np, gamma=cfg.gamma)
                done = bool(done_step)
                ep_reward += float(reward)

                dw = bool(done)   # 実機は最終チャンクを真の終端とみなす
                if not args.collect_only:
                    buffer.store(
                        {k: obs[k] for k in shape_info["obs"].keys()},
                        all_x.squeeze(1).detach().cpu().numpy() if all_x.dim() == 4 else all_x.detach().cpu().numpy(),
                        a_logprob.squeeze(1).detach().cpu().numpy() if a_logprob.dim() == 4 else a_logprob.detach().cpu().numpy(),
                        float(reward),
                        {k: next_obs[k] for k in shape_info["obs"].keys()},
                        float(done), float(dw),
                    )
                ep_records.append({"reward": float(reward), "info": info})
                obs = next_obs
                total_steps += 1

                # batch_size たまったら PPO 更新
                if (not args.collect_only) and buffer.count >= int(cfg.ppo.batch_size):
                    print(f"[update {update_num}] running dp_align_update_no_share (buffer={buffer.count})")
                    actor_loss, critic_loss, bc_loss, distill_loss = unio4.dp_align_update_no_share(buffer, total_steps)
                    print(f"  actor={actor_loss:.4f} critic={critic_loss:.4f} bc={bc_loss:.4f}")
                    buffer.reset()
                    update_num += 1
                    if update_num % args.save_every == 0:
                        save_dir = out_dir / f"online_{update_num}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        unio4.save(str(save_dir))
                        print(f"  saved online checkpoint -> {save_dir}")

            print(f"episode {ep_idx} done: reward={ep_reward:.3f} success={ep_records[-1]['info'].get('is_success')}")
            # ロールアウト保存（データフライホイール用）
            np.savez_compressed(
                rollout_dir / f"ep_{ep_idx:04d}.npz",
                rewards=np.array([r["reward"] for r in ep_records], dtype=np.float32),
                is_success=bool(ep_records[-1]["info"].get("is_success", False)),
            )

            # 次エピソードまで実機を初期姿勢へ
            loop.run_until_complete(reset_to_home(robot))
            input("Enter で次のエピソードを開始（Ctrl+C で終了）...")

    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        if not args.collect_only and update_num >= 0:
            final = out_dir / "online_last"
            final.mkdir(parents=True, exist_ok=True)
            try:
                unio4.save(str(final))
                print(f"final checkpoint -> {final}")
            except Exception as e:
                print(f"[warn] final save 失敗: {e}")
        try:
            loop.run_until_complete(reset_to_home(robot))
            loop.run_until_complete(robot.disconnect())
        except Exception as e:
            print(f"[warn] disconnect: {e}")
        for name, cam in getattr(robot, "cameras", {}).items():
            try:
                cam.disconnect()
            except Exception:
                pass
        print("done.")


def main():
    ap = argparse.ArgumentParser(description="Iloha 実機オンラインRL (RL-100 diffusion-PPO)")
    ap.add_argument("--offline_ckpt", required=True, help="オフラインRLの best チェックポイントdir (model.pt/encoder.pt を含む)")
    ap.add_argument("--zarr", required=True, help="正規化器構築用の RL-100 zarr (iloha_to_rl100_zarr.py の出力)")
    ap.add_argument("--config_dir", default="libs/RL-100/RL-100/rl_100/config", help="RL-100 hydra config ディレクトリ")
    ap.add_argument("--config_name", default="rl100_2d_epsilon", help="RL-100 config 名")
    ap.add_argument("--task_name", default="sushi", help="RL-100 task 名 (config/task/<name>.yaml)")
    ap.add_argument("--task", default="fold the towel", help="言語タスク指示（ログ用）")
    ap.add_argument("--output_dir", default="outputs/iloha_online", help="オンラインチェックポイント/ロールアウト保存先")
    ap.add_argument("--episode_time_s", type=float, default=30.0, help="1エピソードの時間(秒)")
    ap.add_argument("--fps", type=int, default=30, help="制御周波数(Hz)")
    ap.add_argument("--terminal_reward", type=float, default=1.0, help="成功時の終端報酬")
    ap.add_argument("--max_train_steps", type=int, default=100000, help="打ち切りチャンクステップ数")
    ap.add_argument("--save_every", type=int, default=5, help="PPO更新 N 回ごとにチェックポイント保存")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--collect_only", action="store_true", help="PPO更新せず、offlineRL policy でロールアウト収集のみ")
    ap.add_argument("--disable_relative_safety", action="store_true", help="初動/急変時の相対制限安全制御を無効化")
    ap.add_argument("--override", action="append", default=[], help="追加の hydra override (複数指定可)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
