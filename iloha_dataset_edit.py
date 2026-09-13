"""Gaussian joint edits and lossless dataset copying for the local web editor."""
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from threading import Lock

_SAVE_LOCK = Lock()


def save_dataset(root, request):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from iloha_trajectory_web import dataset_path, data_files, load_episode, SCALE

    source = dataset_path(root, request['dataset'])
    name = request['name']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('保存名は英数字・ハイフン・アンダースコアで指定してください')
    episode = request['episode']
    if type(episode) is not int:
        raise ValueError('エピソード番号が不正です')
    original = load_episode(root, source.name, episode)
    edits = request['edits']
    if not isinstance(edits, list) or not 1 <= len(edits) <= 1000:
        raise ValueError('編集を1〜1000件指定してください')
    for edit in edits:
        if not isinstance(edit, dict) or type(edit.get('joint')) is not int or edit['joint'] not in (1, 2, 8, 9):
            raise ValueError('左腕 joint_1・joint_2 または右腕 joint_8・joint_9 を指定してください')
        for key in ('center', 'sigma', 'amplitude'):
            if type(edit.get(key)) not in (int, float) or not math.isfinite(edit[key]):
                raise ValueError('編集値には有限の数値が必要です')
        if not 0 <= edit['center'] <= original['duration'] or edit['sigma'] <= 0 or abs(edit['amplitude']) > math.pi:
            raise ValueError('中心時刻・幅・角度が範囲外です（角度は±180度以内）')
    state = request.get('update_state', False)
    if type(state) is not bool:
        raise ValueError('update_state は真偽値で指定してください')
    features = ['action', 'observation.state'] if state else ['action']
    info = json.loads((source / 'meta/info.json').read_text())
    for feature in features:
        names = info['features'][feature]['names']
        if len(names) != 14 or len(set(names)) != 14 or any(f'joint_{j}' not in names for j in range(14)):
            raise ValueError(f'{feature} の関節名が不正です')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('シンボリックリンクを含むデータセットは保存できません')

    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        result = {k: getattr(a, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')}
        result['count'] = [len(a)]
        result.update({f'q{q:02}': np.quantile(a, q / 100, axis=0).tolist() for q in (1, 10, 50, 90, 99)})
        return result

    destination = root / name
    with _SAVE_LOCK:
        if destination.exists():
            raise ValueError('同名の保存先が存在します。別の名前を指定してください')
        staging = Path(tempfile.mkdtemp(prefix='.iloha-edit-', dir=root))
        try:
            shutil.copytree(source, staging, dirs_exist_ok=True)
            all_values = {f: [] for f in features}
            episode_values = {f: {} for f in features}
            for file in data_files(staging):
                table = pq.read_table(file)
                ids = table['episode_index'].to_pylist()
                frames = table['frame_index'].to_pylist()
                for feature in features:
                    values = table[feature].to_pylist()
                    names = info['features'][feature]['names']
                    for i, (ep, frame) in enumerate(zip(ids, frames, strict=True)):
                        if ep == episode:
                            for edit in edits:
                                j = edit['joint']
                                z = (frame / original['fps'] - edit['center']) / edit['sigma']
                                values[i][names.index(f'joint_{j}')] += SCALE[j] * edit['amplitude'] * math.exp(-0.5 * z * z)
                    column = pa.array(values, type=table.schema.field(feature).type)
                    table = table.set_column(table.schema.get_field_index(feature), table.schema.field(feature), column)
                    # Compute statistics after conversion to the stored dtype.
                    values = column.to_pylist()
                    all_values[feature].extend(values)
                    for ep, value in zip(ids, values, strict=True):
                        episode_values[feature].setdefault(ep, []).append(value)
                pq.write_table(table, file)
            stat_path = staging / 'meta/stats.json'
            totals = json.loads(stat_path.read_text()) if stat_path.exists() else {}
            for feature in features:
                totals[feature] = stats(all_values[feature])
            stat_path.write_text(json.dumps(totals, indent=2, allow_nan=False))
            per_episode = {f: {ep: stats(v) for ep, v in eps.items()} for f, eps in episode_values.items()}
            for file in (staging / 'meta/episodes').rglob('*.parquet'):
                table = pq.read_table(file)
                ids = table['episode_index'].to_pylist()
                for feature in features:
                    for key in totals[feature]:
                        column_name = f'stats/{feature}/{key}'
                        values = [per_episode[feature][ep][key] for ep in ids]
                        if column_name in table.column_names:
                            field = table.schema.field(column_name)
                            table = table.set_column(table.schema.get_field_index(column_name), field, pa.array(values, type=field.type))
                        else:
                            table = table.append_column(column_name, pa.array(values))
                pq.write_table(table, file)
            (staging / 'meta/joint_edits.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {'name': name, 'path': str(destination), 'frames': info['total_frames']}
