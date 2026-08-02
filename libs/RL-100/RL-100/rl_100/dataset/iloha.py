"""Iloha (ALOHA互換 bimanual, 3xRGB) 用の RL-100 2D データセット.

`rl_100.dataset.cloth.Cloth` を継承し、以下を変える:

1. state をそのまま agent_pos として使う（Cloth は 26次元 state を切り出して 14次元にするが、
   Iloha/sushi の zarr は最初から 14次元の ALOHA joint 状態を保存している）。
   → `use_velocity=True` 相当で Cloth 側の切り出しを無効化（get_normalizer/_sample_to_data 等は継承）。

2. `sequence_stride` を受け取り SequenceSampler に渡す（chunk オフラインRLの critic/finetune 用）。

3. **ディスクバックエンド読み込み（既定 in_memory=False）**。RL-100 標準の Cloth は zarr 全体を
   RAM に載せる（fast_parallel_load_zarr）。sushi 全 296 エピソードを 240x320x3 の 6 画像配列で
   持つと ~170GB になり 31GB RAM では OOM する。`ReplayBuffer.create_from_path(mode='r')` で
   zarr を遅延読みし、DataLoader のバッチ毎に必要窓だけディスクから読む。低次元の正規化は
   state/action のみ（画像には触れない）なので追加RAMはほぼ不要。

zarr は `iloha_to_rl100_zarr.py` が生成したものを想定。
"""

import os

import numpy as np

try:
    # DataLoader worker(fork)内での blosc スレッド起因のクラッシュを避ける
    import numcodecs
    numcodecs.blosc.use_threads = False
except Exception:
    pass

from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.dataset.cloth import Cloth

_REQUIRED_KEYS = [
    "state", "action", "rgb_head", "rgb_left_hand", "rgb_right_hand",
    "next_rgb_head", "next_rgb_right_hand", "next_rgb_left_hand",
    "next_state", "next_action", "reward", "done", "timeout", "return",
]


class Iloha(Cloth):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        scale_strategy=None,
        pre_image_norm=False,
        rgb_head_shape=(3, 240, 320),
        rgb_right_hand_shape=(3, 240, 320),
        rgb_left_hand_shape=(3, 240, 320),
        sequence_stride=1,
        in_memory=False,
    ):
        # Cloth.__init__ は呼ばない（fast_parallel_load_zarr で全RAM読みするため）。
        BaseDataset.__init__(self)
        self.task_name = task_name
        self.use_velocity = True  # 14次元 state をそのまま agent_pos に
        self.rgb_head_shape = rgb_head_shape
        self.rgb_right_hand_shape = rgb_right_hand_shape
        self.rgb_left_hand_shape = rgb_left_hand_shape
        self.img_shape = {
            "rgb_head": rgb_head_shape, "rgb_right_hand": rgb_right_hand_shape,
            "rgb_left_hand": rgb_left_hand_shape, "next_rgb_head": rgb_head_shape,
            "next_rgb_right_hand": rgb_right_hand_shape, "next_rgb_left_hand": rgb_left_hand_shape,
        }
        self.sequence_stride = sequence_stride

        self._zarr_path = zarr_path
        self._in_memory = in_memory
        self._pid = os.getpid()
        if in_memory:
            from rl_100.common.fast_replay_buffer_parallel import fast_parallel_load_zarr
            print("Iloha: loading zarr into RAM (in_memory=True)")
            data = fast_parallel_load_zarr(zarr_path, num_workers=128, keys=_REQUIRED_KEYS)
            self.replay_buffer = ReplayBuffer(root=data)
        else:
            print(f"Iloha: disk-backed zarr (lazy read) from {zarr_path}")
            self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # 画像は encoder が先頭 n_obs_steps フレームしか使わない（obs2latent の [:n_obs_steps]）。
        # ディスクバックエンドで窓全体(horizon)を読むと無駄なので、画像キーだけ先頭 n_obs_steps
        # フレームに限定して読む（key_first_k）。低次元(state/action/reward/...)は全窓読む。
        n_obs_steps = pad_before + 1
        image_keys = [
            "rgb_head", "rgb_left_hand", "rgb_right_hand",
            "next_rgb_head", "next_rgb_left_hand", "next_rgb_right_hand",
        ]
        key_first_k = {k: n_obs_steps for k in image_keys if k in self.replay_buffer.keys()}

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            sequence_stride=sequence_stride,
            key_first_k=key_first_k,
        )
        self._key_first_k = key_first_k
        self._n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _sample_to_data(self, sample):
        # 画像は encoder が先頭 n_obs_steps フレームしか使わないので、そこだけ返す。
        # かつ uint8 のまま返す（/255・float化は encoder 側）。これで
        #   ・horizon(18)×float32 の巨大なサンプル(≈400MB)→ n_obs(3)×uint8(≈4MB) に圧縮し
        #   ・DataLoader worker の OOM(SIGKILL) を防ぐ。
        k = self._n_obs_steps
        agent_pos = sample["state"].astype(np.float32)           # (horizon, 14) 低次元なので全窓保持
        next_agent_pos = sample["next_state"].astype(np.float32)
        data = {
            "obs": {
                "rgb_head": sample["rgb_head"][:k],
                "rgb_right_hand": sample["rgb_right_hand"][:k],
                "rgb_left_hand": sample["rgb_left_hand"][:k],
                "agent_pos": agent_pos,
            },
            "next_obs": {
                "rgb_head": sample["next_rgb_head"][:k],
                "rgb_right_hand": sample["next_rgb_right_hand"][:k],
                "rgb_left_hand": sample["next_rgb_left_hand"][:k],
                "agent_pos": next_agent_pos,
            },
            "reward": sample["reward"].astype(np.float32),
            "not_done": 1.0 - sample["done"].astype(np.bool_),
            "return": sample["return"].astype(np.float32),
            "action": sample["action"].astype(np.float32),
            "next_action": sample["next_action"].astype(np.float32),
        }
        return data

    def _ensure_process_local_buffer(self):
        # DataLoader worker(fork)は親の開いた zarr ハンドルを共有すると壊れるので、
        # プロセスが変わったら（disk-backed のときだけ）このプロセス用に開き直す。
        if self._in_memory:
            return
        if getattr(self, "_pid", None) != os.getpid():
            self.replay_buffer = ReplayBuffer.create_from_path(self._zarr_path, mode="r")
            self.sampler.replay_buffer = self.replay_buffer
            self._pid = os.getpid()

    def __getitem__(self, idx):
        self._ensure_process_local_buffer()
        return super().__getitem__(idx)

    def get_validation_dataset(self):
        import copy as _copy

        val_set = _copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            sequence_stride=self.sequence_stride,
            key_first_k=getattr(self, "_key_first_k", {}),
        )
        val_set.train_mask = ~self.train_mask
        return val_set
