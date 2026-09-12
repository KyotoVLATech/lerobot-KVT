#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow>=18,<26"]
# ///
"""Local, read-only trajectory editor. Run: uv run iloha_trajectory_web.py"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "tools" / "iloha_trajectory_web"
JOINT_NAMES = [f"joint_{i}" for i in range(14)]
OFFSET = [0, -1, 1.15, 0, 0, 0, 1] * 2
SCALE = [-1, -1, 1, -1, 1, -1, -1] * 2


def dataset_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root.resolve() or not (path / "meta" / "info.json").is_file():
        raise ValueError("データセットが見つかりません")
    return path


def data_files(path: Path) -> list[Path]:
    # Only the data directory contains frame data; meta/episodes is not motion data.
    files = sorted((path / "data").rglob("*.parquet"))
    if any(not f.resolve().is_relative_to(path.resolve()) for f in files):
        raise ValueError("データセット外を参照するファイルは読み込めません")
    return files


def catalog(root: Path) -> list[dict]:
    entries = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not (path / "meta" / "info.json").is_file():
            continue
        try:
            path = dataset_path(root, path.name)
            info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
            available = bool(data_files(path))
            entries.append(dict(
                name=path.name, fps=info["fps"], frames=info["total_frames"],
                episodes=info["total_episodes"], available=available,
                error=None if available else "軌道本体 data/**/*.parquet がありません（メタ情報のみ）",
            ))
        except (ValueError, KeyError, OSError) as exc:
            entries.append(dict(name=path.name, available=False, error=str(exc)))
    return entries


def load_episode(root: Path, name: str, episode: int) -> dict:
    path = dataset_path(root, name)
    info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0 or episode < 0:
        raise ValueError("FPSまたはエピソード番号が不正です")
    names = info.get("features", {}).get("action", {}).get("names", [])
    if len(names) != 14 or set(names) != set(JOINT_NAMES):
        raise ValueError("actionにjoint_0〜joint_13の14関節が必要です")
    files = data_files(path)
    if not files:
        raise FileNotFoundError(f"{name}: data/**/*.parquet がありません。実データを配置してください。")
    import pyarrow.parquet as pq

    rows = []
    for file in files:
        table = pq.read_table(file, columns=["action", "episode_index", "frame_index"],
                              filters=[("episode_index", "=", episode)])
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: row["frame_index"])
    if not rows:
        raise ValueError(f"エピソード {episode} にフレームがありません")
    order = [names.index(name) for name in JOINT_NAMES]
    actions = []
    for index, row in enumerate(rows):
        if row["frame_index"] != index or row["episode_index"] != episode:
            raise ValueError("フレーム番号に欠損または重複があります")
        raw = row["action"]
        if len(raw) != 14 or any(not math.isfinite(float(v)) for v in raw):
            raise ValueError(f"フレーム {index} の関節値が不正です")
        actions.append([(float(raw[order[j]]) - OFFSET[j]) / SCALE[j] for j in range(14)])
    tasks = []
    for file in sorted((path / "meta" / "episodes").rglob("*.parquet")):
        episode_rows = pq.read_table(file, columns=["tasks"], filters=[("episode_index", "=", episode)]).to_pylist()
        for row in episode_rows:
            tasks.extend(row.get("tasks") or [])
    return dict(name=name, episode=episode, fps=fps, actions=actions, coordinates="iloha", task=" → ".join(dict.fromkeys(tasks)),
                duration=(len(actions) - 1) / fps, synthetic=False)


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript"}

    def __init__(self, *args, datasets_root: Path, **kwargs):
        self.datasets_root = datasets_root
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def json_response(self, value, status=HTTPStatus.OK):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        request = urlsplit(self.path)
        try:
            if request.path == "/api/catalog":
                self.json_response({"datasets": catalog(self.datasets_root)})
            elif request.path == "/api/episode":
                query = parse_qs(request.query)
                self.json_response(load_episode(self.datasets_root, query.get("dataset", [""])[0],
                                                int(query.get("episode", ["0"])[0])))
            elif request.path.startswith("/api/"):
                self.json_response({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
            elif request.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                # Serve only bundled web assets, never workspace or dataset files.
                if request.path not in ("/", "/index.html", "/style.css", "/app.mjs", "/core.mjs", "/worker.mjs"):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                super().do_GET()
        except ImportError:
            self.json_response({"error": "PyArrowが必要です。uv run iloha_trajectory_web.py で起動してください。"},
                               HTTPStatus.SERVICE_UNAVAILABLE)
        except (ValueError, KeyError, OSError) as exc:
            self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, datasets_root=args.datasets_root.resolve()))
    print(f"Iloha Trajectory Studio: http://{args.host}:{server.server_port}", flush=True)
    print(f"Dataset root: {args.datasets_root.resolve()} (read-only; datasets are never modified)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
