"""Local, non-image LeRobot action timeline editor. Run with --help."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def statistics(a):
    result = {k: getattr(np, k)(a, axis=0).tolist() for k in ('min', 'max', 'mean', 'std')}
    result['count'] = [len(a)]
    for q in (1, 10, 50, 90, 99):
        result[f'q{q:02}'] = np.quantile(a, q / 100, axis=0).tolist()
    return result


class Editor:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if any(f.get('dtype') in ('image', 'video') for f in self.info['features'].values()):
            raise ValueError('画像・動画のないデータセットを指定してください。')
        self.tables = {p.relative_to(self.root): pq.read_table(p) for p in sorted((self.root / 'data').rglob('*.parquet'))}
        if not self.tables:
            raise ValueError('data 以下に Parquet がありません。')
        self.rows = {}
        for path, table in self.tables.items():
            for row, ep in enumerate(table['episode_index'].to_pylist()):
                self.rows.setdefault(ep, []).append((path, row))
        self.original, self.actions, self.states = {}, {}, {}
        for ep, rows in self.rows.items():
            rows.sort(key=lambda r: self.tables[r[0]]['frame_index'][r[1]].as_py())
            a = np.array([self.tables[p]['action'][i].as_py() for p, i in rows], dtype=np.float32)
            if a.ndim != 2 or not np.isfinite(a).all():
                raise ValueError('Action は有限値の二次元配列である必要があります。')
            self.original[ep] = a.copy()
            self.actions[ep] = a
            self.states[ep] = ([self.tables[p]['observation.state'][i].as_py() for p, i in rows]
                               if 'observation.state' in next(iter(self.tables.values())).column_names else None)
        self.undo, self.redo = [], []

    def payload(self, ep):
        a = self.actions[ep]
        names = self.info['features']['action'].get('names')
        if not isinstance(names, list) or len(names) != a.shape[1]:
            names = [f'action {i}' for i in range(a.shape[1])]
        return dict(episode=ep, episodes=sorted(self.rows), fps=self.info['fps'], names=names,
                    action=a.tolist(), original=self.original[ep].tolist(), state=self.states[ep],
                    dirty=[e for e in self.actions if not np.array_equal(self.actions[e], self.original[e])],
                    undo=bool(self.undo), redo=bool(self.redo), root=str(self.root))

    def edit(self, req):
        ep = int(req['episode'])
        op = req['op']
        if op in ('undo', 'redo'):
            source, dest = (self.undo, self.redo) if op == 'undo' else (self.redo, self.undo)
            if source:
                e, a = source.pop()
                dest.append((e, self.actions[e].copy()))
                self.actions[e] = a
                ep = e
            return self.payload(ep)
        a = self.actions[ep].copy()
        start, end, target = (int(req[k]) for k in ('start', 'end', 'target'))
        dims = list(dict.fromkeys(int(d) for d in req['dims']))
        length = int(req.get('length', end - start)) if op == 'stretch' else end - start
        if not dims or min(dims) < 0 or max(dims) >= a.shape[1]:
            raise ValueError('編集するレーンを選択してください。')
        if not 0 <= start < end <= len(a) or length < 1 or not 0 <= target <= len(a) - length:
            raise ValueError('範囲または移動先がエピソードの外です。終了フレームは範囲に含みません。')
        clip = a[start:end, dims].copy()
        if op in ('move', 'stretch'):
            # Fill the vacated interval by interpolation between untouched neighbours.
            left, right = a[max(0, start - 1), dims], a[min(len(a) - 1, end), dims]
            a[start:end, dims] = np.linspace(left, right, end - start + 2)[1:-1]
        if op == 'stretch':
            clip = np.stack([np.interp(np.linspace(0, len(clip) - 1, length),
                                       np.arange(len(clip)), clip[:, d]) for d in range(len(dims))], axis=1)
        if op not in ('copy', 'move', 'stretch'):
            raise ValueError('不明な操作です。')
        a[target:target + length, dims] = clip
        self.undo.append((ep, self.actions[ep].copy()))
        self.undo = self.undo[-50:]
        self.redo.clear()
        self.actions[ep] = a
        return self.payload(ep)

    def save(self, destination):
        dest = Path(destination).expanduser().resolve()
        if dest.exists() or dest == self.root or self.root in dest.parents:
            raise ValueError('元データセットの外にある、未使用の保存先を指定してください。')
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.action-editor-', dir=dest.parent))
        try:
            shutil.copytree(self.root, staging, dirs_exist_ok=True)
            replacements = {p: t['action'].to_pylist() for p, t in self.tables.items()}
            for ep, rows in self.rows.items():
                for j, (p, i) in enumerate(rows):
                    replacements[p][i] = self.actions[ep][j].tolist()
            for p, t in self.tables.items():
                idx = t.schema.get_field_index('action')
                t = t.set_column(idx, t.schema.field(idx), pa.array(replacements[p], type=t.schema.field(idx).type))
                pq.write_table(t, staging / p)
            stats_path = staging / 'meta/stats.json'
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
            stats['action'] = statistics(np.concatenate(list(self.actions.values())))
            stats_path.write_text(json.dumps(stats, indent=2) + '\n')
            per_episode = {ep: statistics(a) for ep, a in self.actions.items()}
            for p in (staging / 'meta/episodes').rglob('*.parquet'):
                t = pq.read_table(p)
                for key in stats['action']:
                    name = f'stats/action/{key}'
                    values = [per_episode[e][key] for e in t['episode_index'].to_pylist()]
                    if name in t.column_names:
                        idx = t.schema.get_field_index(name)
                        t = t.set_column(idx, t.schema.field(idx), pa.array(values, type=t.schema.field(idx).type))
                    else:
                        t = t.append_column(name, pa.array(values))
                pq.write_table(t, p)
            old_stats = staging / 'meta/episodes_stats.jsonl'
            if old_stats.exists():
                records = [json.loads(line) for line in old_stats.read_text().splitlines() if line.strip()]
                for record in records:
                    record['stats']['action'] = per_episode[record['episode_index']]
                old_stats.write_text(''.join(json.dumps(r) + '\n' for r in records))
            staging.rename(dest)
        except Exception:
            shutil.rmtree(staging)
            raise
        return {'saved': str(dest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='datasets/iloha-best')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    editor = Editor(args.dataset)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/':
                raw = Path(__file__).with_name('index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == '/api/load':
                self.reply(editor.payload(0 if 0 in editor.rows else min(editor.rows)))
            else:
                self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            # Only same-origin JSON requests may mutate the local editor.
            origin = self.headers.get('Origin')
            if ((origin and urlparse(origin).netloc != self.headers.get('Host'))
                    or self.headers.get('Content-Type') != 'application/json'):
                self.reply({'error': 'Invalid request origin/content type'}, 403)
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 1_000_000:
                    raise ValueError('Invalid request size')
                req = json.loads(self.rfile.read(size))
                if self.path == '/api/episode':
                    result = editor.payload(int(req['episode']))
                elif self.path == '/api/edit':
                    result = editor.edit(req)
                elif self.path == '/api/save':
                    result = editor.save(req['destination'])
                else:
                    raise ValueError('Unknown endpoint')
                self.reply(result)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.reply({'error': str(exc)}, 400)

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Action editor: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
