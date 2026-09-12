"""Write edited, unretimed ideal motion as a standalone LeRobot v3 dataset."""
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path

JOINT_NAMES = [f"joint_{i}" for i in range(14)]
OFFSET = [0, -1, 1.15, 0, 0, 0, 1] * 2
SCALE = [-1, -1, 1, -1, 1, -1, -1] * 2


def column_stats(rows: list[list[float]]) -> dict:
    count = len(rows)
    result = {key: [] for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}
    for column in zip(*rows, strict=True):
        values = sorted(column)
        mean = math.fsum(values) / count
        result["min"].append(values[0])
        result["max"].append(values[-1])
        result["mean"].append(mean)
        result["std"].append(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / count))
        for key, percentile in (("q01", .01), ("q10", .1), ("q50", .5), ("q90", .9), ("q99", .99)):
            at = (count - 1) * percentile
            lo = int(at)
            result[key].append(values[lo] + (values[min(lo + 1, count - 1)] - values[lo]) * (at - lo))
    result["count"] = [count]
    return result


def export_dataset(root: Path, payload: dict) -> dict:
    """Only create a new child of root; never overwrite an existing dataset.

    The frontend rebuilds these ideal actions from source clips at base speed 1,
    without simulation. The separate settings document is saved but never used
    to transform actions here.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    name = payload.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("保存名は英数字・ハイフン・アンダースコアの1〜80文字にしてください")
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        raise ValueError("その保存名は使用できません")
    root = root.resolve()
    target = root / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{name} は既にあります。別の保存名を指定してください。")
    if payload.get("coordinates") != "iloha":
        raise ValueError("Iloha座標の理想軌道が必要です")
    fps = payload.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or not 1 <= fps <= 240 or int(fps) != fps:
        raise ValueError("書き出しFPSは1〜240の整数にしてください")
    fps = int(fps)
    actions = payload.get("actions")
    if not isinstance(actions, list) or not 2 <= len(actions) <= 150000:
        raise ValueError("2〜150000フレームの理想軌道が必要です")
    for row in actions:
        if not isinstance(row, list) or len(row) != 14 or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or abs(v) > 1000 for v in row
        ):
            raise ValueError("関節値は14要素の有限数値で指定してください")
        if any(not 0 <= row[j] <= 1 for j in (6, 13)):
            raise ValueError("グリッパー指令は0〜1にしてください")
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("設定JSONの形式が不正です")
    document = {**settings, "schema_version": 1, "dataset": name, "speed_applied": False,
                "actuator_limits_applied": False, "observation_state": "ideal_action_copy_not_measured",
                "fps": fps, "frames": len(actions)}
    settings_json = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    task = payload.get("task") or "Merged ideal Iloha trajectory"
    if not isinstance(task, str) or len(task) > 10000:
        raise ValueError("タスク名が不正です")

    # Export in the same ALOHA convention as the source datasets / replay loader.
    aloha = [[OFFSET[j] + SCALE[j] * q[j] for j in range(14)] for q in actions]
    count = len(aloha)
    joint_array = pa.array(aloha, type=pa.list_(pa.float32(), 14))
    table = pa.table({
        "observation.state": joint_array,
        "action": joint_array,
        "timestamp": pa.array([i / fps for i in range(count)], type=pa.float32()),
        "frame_index": pa.array(range(count), type=pa.int64()),
        "episode_index": pa.array([0] * count, type=pa.int64()),
        "index": pa.array(range(count), type=pa.int64()),
        "task_index": pa.array([0] * count, type=pa.int64()),
    })
    features = {
        key: {"dtype": "float32", "shape": [14], "names": JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    stats = {}
    for key in features:
        values = table[key].to_pylist()
        stats[key] = column_stats(values if key in ("action", "observation.state") else [[v] for v in values])
    episode = {"episode_index": 0, "tasks": [task], "length": count,
               "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": 0, "dataset_to_index": count,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
    for feature, values in stats.items():
        for key, value in values.items():
            episode[f"stats/{feature}/{key}"] = value
    info = dict(codebase_version="v3.0", robot_type="aloha", total_episodes=1, total_frames=count,
                total_tasks=1, chunks_size=1000, data_files_size_in_mb=100, video_files_size_in_mb=200,
                fps=fps, splits={"train": "0:1"},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path=None, features=features)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trajectory-export-", dir=root)).resolve()
    try:
        (staging / "data" / "chunk-000").mkdir(parents=True)
        (staging / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        pq.write_table(table, staging / "data" / "chunk-000" / "file-000.parquet")
        pq.write_table(pa.Table.from_pylist([episode]), staging / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        # Match pandas' task-indexed DataFrame without requiring pandas at runtime.
        tasks = pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([task])})
        pandas_metadata = {"index_columns": ["task"], "column_indexes": [], "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ], "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.0.0"}
        tasks = tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode()})
        pq.write_table(tasks, staging / "meta" / "tasks.parquet")
        for filename, content in (("info.json", info), ("stats.json", stats)):
            (staging / "meta" / filename).write_text(json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        (staging / "trajectory_settings.json").write_text(settings_json, encoding="utf-8")
        # Reserve target before moving files; even concurrent exports cannot overwrite.
        target.mkdir(exist_ok=False)
        try:
            for child in staging.iterdir():
                child.rename(target / child.name)
        except BaseException:
            # This directory was created exclusively by this export.
            shutil.rmtree(target)
            raise
    finally:
        if staging.is_dir() and staging.parent == root and staging.name.startswith(".trajectory-export-"):
            shutil.rmtree(staging)
    return dict(name=name, path=str(target), frames=count, fps=fps, duration=(count - 1) / fps,
                settings_path=str(target / "trajectory_settings.json"))
