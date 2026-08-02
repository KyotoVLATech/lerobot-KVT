#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LeRobot v3.0 データセット → RL-100 2D(RGB) 用 zarr 変換スクリプト.

`datasets/iloha-dataset-sushi` のような LeRobot v3.0 データセット（parquet + av1 mp4,
14次元 state / 14次元 action, 3台のRGBカメラ）を、RL-100 の 2D 画像パイプライン
(`rl_100.dataset.cloth.Cloth` / `rl_100.dataset.iloha.Iloha`) が読み込める zarr に変換する。

出力 zarr レイアウト（zarr v2 フォーマット, RL-100 の ReplayBuffer 互換）:

    <out>.zarr/
    ├── data/
    │   ├── state              (N, 14)        float32   ALOHA座標の観測状態
    │   ├── action             (N, 14)        float32   ALOHA座標の目標action
    │   ├── rgb_head           (N, H, W, 3)   uint8     cam_high         (HWC, RGB)
    │   ├── rgb_left_hand       (N, H, W, 3)   uint8     cam_left_wrist   (HWC, RGB)
    │   ├── rgb_right_hand      (N, H, W, 3)   uint8     cam_right_wrist  (HWC, RGB)
    │   ├── next_state / next_action / next_rgb_*         上記の1ステップ先(エピソード末尾は自己複製)
    │   ├── reward             (N, 1)         float32   エピソード終端のみ terminal_reward, 他は0
    │   ├── done               (N, 1)         bool      エピソード終端 True
    │   ├── timeout            (N, 1)         bool      エピソード終端 True (= done)
    │   └── return             (N, 1)         float32   γ で後ろ向きに計算した割引リターン
    └── meta/
        └── episode_ends       (n_episodes,)  int64     累積終端インデックス (最後 == N)

画像は uint8(0-255) のまま保存する（/255 正規化はエンコーダ側で行う）。state/action も
生値のまま保存する（正規化は dataset.get_normalizer() が学習時に行う）。座標系はデータセットの
ALOHA 座標のまま保存し、実機デプロイ側 (iloha_online_rl.py) で iloha<->aloha 変換する。

使い方:
    uv run python iloha_to_rl100_zarr.py \
        --dataset datasets/iloha-dataset-sushi \
        --out libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr

依存: pandas, pyarrow, numpy, av (PyAV), opencv-python(cv2), zarr>=2 (v3 でも zarr_format=2 で書ける).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

# LeRobot v3.0 のカメラキー -> RL-100 cloth スキーマの画像キー
DEFAULT_CAM_MAP = {
    "cam_high": "rgb_head",
    "cam_left_wrist": "rgb_left_hand",
    "cam_right_wrist": "rgb_right_hand",
}


def compute_return(reward: np.ndarray, not_done: np.ndarray, gamma: float) -> np.ndarray:
    """cloth.py の compute_return と同じ後ろ向き割引リターン計算 (N,1)."""
    size = len(reward)
    out = np.zeros((size, 1), dtype=np.float32)
    pre = 0.0
    r = reward.reshape(-1)
    nd = not_done.reshape(-1)
    for i in reversed(range(size)):
        out[i, 0] = r[i] + gamma * pre * nd[i]
        pre = out[i, 0]
    return out


def load_episode_meta(dataset_root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(dataset_root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no episode meta parquet under {dataset_root/'meta'/'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def load_data_frame(dataset_root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(dataset_root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dataset_root/'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # グローバル index 順に整列（video のフレーム順と一致させる）
    return df.sort_values("index").reset_index(drop=True)


def stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype(np.float32)


def video_path(dataset_root: Path, cam_key: str, chunk_idx: int, file_idx: int) -> Path:
    return dataset_root / "videos" / f"observation.images.{cam_key}" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def decode_camera_into_zarr(
    dataset_root: Path,
    ep: pd.DataFrame,
    cam_key: str,
    zarr_out_key: str,
    data_group: zarr.Group,
    total_rows: int,
    H: int,
    W: int,
    fps: float,
    chunk_len: int,
    compressor,
) -> None:
    """1台のカメラ動画を復号し、行順で zarr 配列 data/<zarr_out_key> に書き込む。

    LeRobot v3.0 の動画は複数エピソードを連結した1本のストリーム。各エピソードのフレーム位置は
    from_timestamp から求める（round(from_ts*fps) がそのエピソードの動画内開始フレーム）。
    タイムスタンプの丸め誤差でエピソード境界は data 行の累積長から最大±数十フレームずれるため、
    各エピソードごとに自分の開始フレームから length フレームだけ取り出す（境界の余剰フレームは捨てる）。
    出力行は 0..N-1 で厳密に連続するので、chunk_len 単位でバッファリングして書き込む。
    """
    arr = data_group.create_array(
        name=zarr_out_key,
        shape=(total_rows, H, W, 3),
        chunks=(chunk_len, H, W, 3),
        dtype="uint8",
        compressor=compressor,
        overwrite=True,
    )

    tf_col = f"videos/observation.images.{cam_key}/from_timestamp"
    ci_col = f"videos/observation.images.{cam_key}/chunk_index"
    fi_col = f"videos/observation.images.{cam_key}/file_index"

    # (chunk_idx, file_idx) ごとにエピソードをまとめる
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(len(ep)):
        key = (int(ep[ci_col].iloc[i]), int(ep[fi_col].iloc[i]))
        groups.setdefault(key, []).append(i)

    write_buf: list[np.ndarray] = []
    write_cursor = 0  # 次に書き込む zarr の行

    def flush():
        nonlocal write_buf, write_cursor
        if not write_buf:
            return
        block = np.stack(write_buf)
        arr[write_cursor : write_cursor + len(block)] = block
        write_cursor += len(block)
        write_buf = []

    for (ci, fi), ep_indices in sorted(groups.items()):
        # このファイル内のブロックを開始フレーム順に並べる
        blocks = []
        for i in ep_indices:
            vstart = int(round(float(ep[tf_col].iloc[i]) * fps))
            length = int(ep["length"].iloc[i])
            drow = int(ep["dataset_from_index"].iloc[i])
            blocks.append((vstart, length, drow))
        blocks.sort(key=lambda b: b[0])

        path = video_path(dataset_root, cam_key, ci, fi)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")

        container = av.open(str(path))
        ptr = 0
        j = 0
        for frame in container.decode(video=0):
            while ptr + 1 < len(blocks) and j >= blocks[ptr + 1][0]:
                ptr += 1
            vstart, length, drow = blocks[ptr]
            off = j - vstart
            if 0 <= off < length:
                out_row = drow + off
                if out_row != write_cursor + len(write_buf):
                    raise RuntimeError(
                        f"non-sequential write for {cam_key}: expected {write_cursor+len(write_buf)}, got {out_row}"
                    )
                img = frame.to_ndarray(format="rgb24")  # HWC uint8 RGB
                if (img.shape[0], img.shape[1]) != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                write_buf.append(np.ascontiguousarray(img, dtype=np.uint8))
                if len(write_buf) >= chunk_len:
                    flush()
            j += 1
        container.close()

    flush()
    if write_cursor != total_rows:
        raise RuntimeError(
            f"{cam_key}: wrote {write_cursor} rows, expected {total_rows} (video/frame misalignment)"
        )
    print(f"  [{cam_key} -> {zarr_out_key}] wrote {write_cursor} frames {H}x{W}")


def build_next_images(data_group: zarr.Group, key: str, next_key: str, bounds: list[tuple[int, int]], chunk_len: int, H: int, W: int, compressor) -> None:
    """data/<key> から1ステップ先の画像 data/<next_key> を作る（エピソード末尾は自己複製）。"""
    src = data_group[key]
    n = src.shape[0]
    dst = data_group.create_array(
        name=next_key,
        shape=(n, H, W, 3),
        chunks=(chunk_len, H, W, 3),
        dtype="uint8",
        compressor=compressor,
        overwrite=True,
    )
    for (s, e) in bounds:
        if e - s >= 2:
            dst[s : e - 1] = src[s + 1 : e]
        dst[e - 1] = src[e - 1]  # 末尾フレームは自分自身
    print(f"  [{next_key}] built from {key}")


def build_next_lowdim(x: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    nx = np.empty_like(x)
    for (s, e) in bounds:
        if e - s >= 2:
            nx[s : e - 1] = x[s + 1 : e]
        nx[e - 1] = x[e - 1]
    return nx


def main():
    ap = argparse.ArgumentParser(description="LeRobot v3.0 -> RL-100 2D zarr converter")
    ap.add_argument("--dataset", required=True, help="LeRobot v3.0 データセットのルート (例: datasets/iloha-dataset-sushi)")
    ap.add_argument("--out", required=True, help="出力 zarr パス (例: libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr)")
    ap.add_argument("--image_h", type=int, default=240, help="保存する画像の高さ (既定 240, cloth と同じ)")
    ap.add_argument("--image_w", type=int, default=320, help="保存する画像の幅 (既定 320, cloth と同じ)")
    ap.add_argument("--terminal_reward", type=float, default=1.0, help="各エピソード終端に与える疎な報酬 (teleopは全て成功とみなす)")
    ap.add_argument("--gamma", type=float, default=0.99, help="return 計算の割引率")
    ap.add_argument("--chunk_len", type=int, default=100, help="zarr の時間方向チャンク長")
    ap.add_argument("--max_episodes", type=int, default=0, help=">0 の場合、先頭からこのエピソード数だけ変換（検証用）")
    ap.add_argument(
        "--cam_map",
        type=str,
        default="",
        help='カメラ対応の上書き JSON 例: \'{"cam_high":"rgb_head"}\' 省略時は既定マップ',
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    out_path = args.out
    H, W = args.image_h, args.image_w
    cam_map = json.loads(args.cam_map) if args.cam_map else DEFAULT_CAM_MAP

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    print(f"dataset={dataset_root} fps={fps} out={out_path} image={H}x{W}")

    ep = load_episode_meta(dataset_root)
    df = load_data_frame(dataset_root)

    if args.max_episodes and args.max_episodes > 0:
        ep = ep.iloc[: args.max_episodes].reset_index(drop=True)
        last_row = int(ep["dataset_to_index"].iloc[-1])
        df = df.iloc[:last_row].reset_index(drop=True)
        print(f"[subset] {len(ep)} episodes, {last_row} frames")

    total_rows = int(ep["dataset_to_index"].iloc[-1])
    assert len(df) == total_rows, f"data rows {len(df)} != dataset_to_index end {total_rows}"

    # エピソード境界 (start,end) と episode_ends
    bounds = [(int(ep["dataset_from_index"].iloc[i]), int(ep["dataset_to_index"].iloc[i])) for i in range(len(ep))]
    episode_ends = ep["dataset_to_index"].to_numpy().astype(np.int64)

    # low-dim
    state = stack_col(df, "observation.state")  # (N,14)
    action = stack_col(df, "action")            # (N,14)
    assert state.shape[1] == 14 and action.shape[1] == 14, f"expected 14-dim, got {state.shape}, {action.shape}"
    next_state = build_next_lowdim(state, bounds)
    next_action = build_next_lowdim(action, bounds)

    # reward / done / timeout / return
    reward = np.zeros((total_rows, 1), dtype=np.float32)
    done = np.zeros((total_rows, 1), dtype=bool)
    timeout = np.zeros((total_rows, 1), dtype=bool)
    for (s, e) in bounds:
        reward[e - 1, 0] = args.terminal_reward
        done[e - 1, 0] = True
        timeout[e - 1, 0] = True
    ret = compute_return(reward, 1.0 - done.astype(np.float32), args.gamma)

    # zarr v2 で書く (RL-100 の zarr v2 スタックと互換)
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} already exists; remove it first")
    root = zarr.open_group(out_path, mode="w", zarr_format=2)
    data_g = root.create_group("data")
    meta_g = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    def put(name, arr2d):
        a = data_g.create_array(
            name=name,
            shape=arr2d.shape,
            dtype=arr2d.dtype,
            chunks=(min(args.chunk_len * 100, len(arr2d)), arr2d.shape[1]),
            compressor=compressor,
            overwrite=True,
        )
        a[:] = arr2d

    print("writing low-dim arrays...")
    put("state", state)
    put("action", action)
    put("next_state", next_state)
    put("next_action", next_action)
    put("reward", reward)
    put("done", done)
    put("timeout", timeout)
    put("return", ret)
    ee = meta_g.create_array(
        name="episode_ends",
        shape=episode_ends.shape,
        dtype=episode_ends.dtype,
        chunks=(len(episode_ends),),
        compressor=compressor,
        overwrite=True,
    )
    ee[:] = episode_ends

    print("decoding videos -> current rgb arrays...")
    for cam_key, zkey in cam_map.items():
        decode_camera_into_zarr(dataset_root, ep, cam_key, zkey, data_g, total_rows, H, W, fps, args.chunk_len, compressor)

    print("building next rgb arrays...")
    for zkey in cam_map.values():
        build_next_images(data_g, zkey, f"next_{zkey}", bounds, args.chunk_len, H, W, compressor)

    print("done.")
    print(f"episodes={len(ep)} frames={total_rows}")
    print(f"zarr written to: {out_path}")


if __name__ == "__main__":
    main()
