#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory Studioの画面上の軌道を`trajectory_settings.json`から再構成する。

`tools/iloha_trajectory_web/core.mjs` のプレビュー計算をPythonへ移植したもので、
各クリップの速度設定・使用区間・境界接続をブラウザと同じ順序で適用する。
データセットの書き出しは不要で、設定ファイルと元データセットがあれば
画面と同じフレーム列を実機へ送信できる。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi
# グリッパー（6, 13）は角度ではないため巻き戻しと範囲の扱いを分ける。
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_JOINTS = [6, 13]
TRANSITIONS = ("cut", "crossfade", "linear", "smooth")
SUPPORTED_SCHEMAS = (2, 3)
MAX_CLIPS = 20
MAX_FRAMES = 150000
DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
DEFAULT_REPLAY = dict(base_speed=1.0, max_speedup=2.0, gripper_margin=0.5,
                      speedup_distance=1.0, gripper_threshold=1e-4)


def _round(value: float) -> int:
    """JavaScriptのMath.roundと同じ丸め（.5は正の無限大方向）。"""
    return int(math.floor(value + 0.5))


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}には有限の数値が必要です")
    return float(value)


def replay_intervals(
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

    moving = np.any(np.abs(np.diff(actions[:, GRIPPER_JOINTS], axis=0)) > gripper_threshold, axis=1)
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
    # 括弧の付け方までcore.mjsと揃える。丸め1つの違いが軌道の差になる。
    speedup = 1.0 + (max_speedup - 1.0) * ramp * ramp * (3.0 - 2.0 * ramp)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        intervals = (1.0 / fps / base_speed) / speedup
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("速度設定によるフレーム間隔が表現可能な範囲を超えています")
    return intervals


def unwrap(actions: np.ndarray) -> np.ndarray:
    """腕の関節角だけを最短経路で連続化する。グリッパーは変更しない。"""
    result = np.array(actions, dtype=np.float64)
    for i in range(1, len(result)):
        previous, current = result[i - 1, ARM_JOINTS], result[i, ARM_JOINTS]
        result[i, ARM_JOINTS] = current + TAU * np.floor((previous - current) / TAU + 0.5)
    return result


def prepare_clip(actions: np.ndarray, source_fps: float, start: float, end: float,
                 replay: dict, fps: float) -> np.ndarray:
    """速度設定で時間を変更してから使用区間を切り出し、出力FPSで再標本化する。"""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2 or not np.isfinite(actions).all():
        raise ValueError("2フレーム以上の有限な14関節指令が必要です")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("元データのFPSは有限の正数で指定してください")
    max_time = (len(actions) - 1) / source_fps
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 \
            or end > max_time + 1e-6 or end - start < 1 / fps:
        raise ValueError(f"使用区間は0〜{max_time:g}秒の範囲で1フレーム以上にしてください")
    source = unwrap(actions)
    intervals = replay_intervals(source, source_fps, **replay)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    def warped(seconds: float) -> float:
        x = seconds * source_fps
        index = min(int(math.floor(x)), len(times) - 2)
        return float(times[index] + (x - index) * intervals[index])

    begin, finish = warped(start), warped(end)
    count = max(1, _round((finish - begin) * fps))
    if count > MAX_FRAMES:
        raise ValueError("クリップが長すぎます。区間または速度を調整してください")
    sampled = begin + (finish - begin) * np.arange(count + 1) / count
    # times[index] <= t < times[index + 1]。core.mjsのlowerSegmentと同じ区間を選ぶ。
    index = np.clip(np.searchsorted(times, sampled, side="right") - 1, 0, len(times) - 2)
    weight = np.clip((sampled - times[index]) / intervals[index], 0.0, 1.0)
    return source[index] + (source[index + 1] - source[index]) * weight[:, None]


def stitch(clips: list[dict], mode: str = "crossfade", blend: float = 1.0, fps: float = 60.0) -> dict:
    """速度変更済みのクリップを指定の接続方式でつなぎ、1本の軌道にする。"""
    if not clips:
        raise ValueError("クリップがありません")
    if not math.isfinite(fps) or not 1 <= fps <= 240 or not math.isfinite(blend) or not 0 <= blend <= 60:
        raise ValueError("接続時間またはFPSが不正です")
    if mode not in TRANSITIONS:
        raise ValueError(f"接続方式が不正です: {mode}")
    frames: list[np.ndarray] = []
    segments: list[dict] = []
    boundaries: list[dict] = []
    for order, clip in enumerate(clips):
        following = list(prepare_clip(clip["actions"], clip["fps"], clip["start"], clip["end"],
                                      clip["replay"], fps))
        start_frame = len(frames)
        if order == 0:
            frames = following
            start_frame = 0
        else:
            last = frames[-1]
            # 境界の角度差を2πの整数倍で詰めてから接続する。
            shift = TAU * np.floor((last[ARM_JOINTS] - following[0][ARM_JOINTS]) / TAU + 0.5)
            for row in following:
                row[ARM_JOINTS] += shift
            if mode == "crossfade" and blend > 0:
                count = _round(blend * fps)
                if count < 1 or count >= len(frames) or count >= len(following) \
                        or count / fps > segments[-1]["end"] - segments[-1]["start"] + 1e-6:
                    raise ValueError("クロスディゾルブ時間を各クリップの使用区間より短くしてください")
                start_frame = len(frames) - 1 - count
                for k in range(count + 1):
                    u = k / count
                    weight = u * u * (3 - 2 * u)
                    current = frames[start_frame + k]
                    frames[start_frame + k] = current + (following[k] - current) * weight
                frames.extend(following[count + 1:])
                boundaries.append(dict(start=start_frame / fps, end=(start_frame + count) / fps, mode=mode))
            elif mode in ("linear", "smooth") and blend > 0:
                count = max(1, _round(blend * fps))
                begin = len(frames) - 1
                if mode == "smooth" and len(frames) < 2:
                    raise ValueError("滑らかな補間には2フレーム以上の先行クリップが必要です")
                before, after = (frames[-2] if mode == "smooth" else None), following[1]
                for k in range(1, count):
                    u = k / count
                    if mode == "linear":
                        frames.append(last + (following[0] - last) * u)
                    else:  # 三次Hermite補間。両端の関節速度に合わせて接続する。
                        h00, h10 = 2 * u**3 - 3 * u * u + 1, u**3 - 2 * u * u + u
                        h01, h11 = -2 * u**3 + 3 * u * u, u**3 - u * u
                        frames.append(h00 * last + h10 * count * (last - before)
                                      + h01 * following[0] + h11 * count * (after - following[0]))
                start_frame = len(frames)
                frames.extend(following)
                boundaries.append(dict(start=begin / fps, end=start_frame / fps, mode=mode))
            else:
                frames.extend(following)
                boundaries.append(dict(start=(start_frame - 1) / fps, end=start_frame / fps, mode="cut"))
        segments.append(dict(name=clip.get("name", ""), sourceStart=clip["start"], sourceEnd=clip["end"],
                             start=start_frame / fps, end=(len(frames) - 1) / fps,
                             synthetic=bool(clip.get("synthetic"))))
    actions = np.stack(frames)
    # グリッパーは角度ではないため、補間結果を有効範囲へ戻す。
    actions[:, GRIPPER_JOINTS] = np.clip(actions[:, GRIPPER_JOINTS], 0.0, 1.0)
    return dict(actions=actions, fps=fps, duration=(len(actions) - 1) / fps,
                segments=segments, boundaries=boundaries)


def parse_settings(document: dict) -> dict:
    """Trajectory Studioの設定JSONを検証し、再構成に必要な値だけを取り出す。"""
    if not isinstance(document, dict):
        raise ValueError("設定ファイルの形式が不正です")
    schema = document.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"対応していない設定ファイルです（schema_version={schema!r}）")
    raw_clips = document.get("clips")
    if not isinstance(raw_clips, list) or not 1 <= len(raw_clips) <= MAX_CLIPS:
        raise ValueError(f"clipsには1〜{MAX_CLIPS}個のクリップが必要です")
    clips = []
    for index, clip in enumerate(raw_clips):
        if not isinstance(clip, dict):
            raise ValueError(f"クリップ{index + 1}の形式が不正です")
        if clip.get("synthetic"):
            raise ValueError(f"クリップ{index + 1}は動作確認用の人工データです。実機では再生できません")
        name = clip.get("dataset")
        if not isinstance(name, str) or not DATASET_NAME.fullmatch(name):
            raise ValueError(f"クリップ{index + 1}のデータセット名が不正です: {name!r}")
        episode = clip.get("episode", 0)
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValueError(f"クリップ{index + 1}のエピソード番号は0以上の整数で指定してください")
        start = _number(clip.get("start", 0), f"クリップ{index + 1}の開始時刻")
        end = _number(clip.get("end"), f"クリップ{index + 1}の終了時刻")
        if start < 0 or end <= start:
            raise ValueError(f"クリップ{index + 1}の使用区間が不正です（{start}〜{end}）")
        replay = dict(DEFAULT_REPLAY)
        for key, value in (clip.get("replay") or {}).items():
            if key not in DEFAULT_REPLAY:
                continue  # 旧版が追加した未知の速度設定は無視する。
            replay[key] = _number(value, f"クリップ{index + 1}の{key}")
        clips.append(dict(dataset=name, episode=episode, start=start, end=end, replay=replay))
    edit = document.get("edit")
    if not isinstance(edit, dict):
        raise ValueError("editに接続方式と接続時間が必要です")
    mode = edit.get("mode", "crossfade")
    if mode not in TRANSITIONS:
        raise ValueError(f"接続方式が不正です: {mode!r}")
    blend = _number(edit.get("blend", 0), "接続時間")
    expected = document.get("trajectory") if isinstance(document.get("trajectory"), dict) else {}
    # schema 3は再構成に使うFPSをtrajectoryに記録する。旧版はeditのFPSを使う。
    fps = _number(expected.get("fps", edit.get("fps", 60)), "FPS")
    if not 1 <= fps <= 240:
        raise ValueError("FPSは1〜240で指定してください")
    frames = expected.get("frames")
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 2):
        raise ValueError("trajectory.framesが不正です")
    return dict(schema_version=schema, clips=clips, mode=mode, blend=blend, fps=fps,
                speed_applied=bool(document.get("speed_applied", schema >= 3)),
                expected_frames=frames, task=document.get("task") or "")


def load_settings(path: str | Path) -> dict:
    """設定JSONを読み込んで検証する。"""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    if path.stat().st_size > 20_000_000:
        raise ValueError("設定ファイルが大きすぎます")
    return parse_settings(json.loads(path.read_text(encoding="utf-8")))


def build_trajectory(settings: dict, load_source) -> dict:
    """設定と元データセットから画面と同じフレーム列を再構成する。

    load_source(dataset_name, episode)は(Iloha座標のactions, fps)を返す。
    """
    clips = []
    for clip in settings["clips"]:
        actions, fps = load_source(clip["dataset"], clip["episode"])
        clips.append(dict(name=clip["dataset"], actions=actions, fps=fps, start=clip["start"],
                          end=clip["end"], replay=clip["replay"], synthetic=False))
    trajectory = stitch(clips, settings["mode"], settings["blend"], settings["fps"])
    expected = settings.get("expected_frames")
    if expected is not None and expected != len(trajectory["actions"]):
        raise ValueError(
            f"設定ファイルと再構成結果のフレーム数が一致しません（設定 {expected} / 再構成 "
            f"{len(trajectory['actions'])}）。元データセットが書き出し時と異なります。"
        )
    return trajectory
