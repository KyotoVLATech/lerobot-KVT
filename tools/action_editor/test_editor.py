"""Run: .venv/bin/python -m unittest discover -s tools/action_editor -p 'test_*.py'"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from server import Editor


class EditorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        (self.root / 'meta/episodes/chunk-000').mkdir(parents=True)
        (self.root / 'data/chunk-000').mkdir(parents=True)
        (self.root / 'meta/info.json').write_text(json.dumps(dict(
            fps=30, features={'action': {'dtype': 'float32', 'shape': [2], 'names': ['a', 'b']}})))
        # Episodes split across files, including a file containing two episodes.
        for file, indices in enumerate(([0, 1, 2], [3, 4, 5, 6, 7])):
            t = pa.table({'episode_index': [i // 4 for i in indices],
                          'frame_index': [i % 4 for i in indices],
                          'timestamp': [i % 4 / 30 for i in indices],
                          'action': pa.array([[float(i), float(i + 10)] for i in indices],
                                             type=pa.list_(pa.float32(), 2))})
            pq.write_table(t, self.root / f'data/chunk-000/file-{file:03}.parquet')
        pq.write_table(pa.table({'episode_index': [0, 1], 'length': [4, 4]}),
                       self.root / 'meta/episodes/chunk-000/file-000.parquet')
        self.editor = Editor(self.root)

    def edit(self, op, **kw):
        return self.editor.edit(dict(episode=0, op=op, start=0, end=2, target=1, dims=[0], **kw))

    def test_overlap_move_and_history(self):
        original = self.editor.actions[0].copy()
        self.edit('move')
        np.testing.assert_array_equal(self.editor.actions[0][1:3, 0], original[:2, 0])
        np.testing.assert_array_equal(self.editor.actions[0][:, 1], original[:, 1])
        modified = self.editor.actions[0].copy()
        self.edit('undo')
        np.testing.assert_array_equal(self.editor.actions[0], original)
        self.edit('redo')
        np.testing.assert_array_equal(self.editor.actions[0], modified)

    def test_stretch_and_invalid_range(self):
        self.edit('stretch', length=3)
        np.testing.assert_array_equal(self.editor.actions[0][1:, 0], [0, .5, 1])
        with self.assertRaises(ValueError):
            self.edit('stretch', length=5)

    def test_export_preserves_columns_and_updates_statistics(self):
        self.edit('copy')
        dest = Path(self.tmp.name) / 'edited'
        self.editor.save(dest)
        saved = Editor(dest)
        for ep in (0, 1):
            np.testing.assert_array_equal(saved.actions[ep], self.editor.actions[ep])
        for p, t in self.editor.tables.items():
            exported = pq.read_table(dest / p)
            self.assertEqual(t.schema, exported.schema)
            for column in ('episode_index', 'frame_index', 'timestamp'):
                self.assertEqual(t[column], exported[column])
        stats = json.loads((dest / 'meta/stats.json').read_text())['action']
        np.testing.assert_allclose(stats['mean'], np.concatenate(list(saved.actions.values())).mean(axis=0))
        meta = pq.read_table(dest / 'meta/episodes/chunk-000/file-000.parquet')
        np.testing.assert_allclose(meta['stats/action/mean'][0].as_py(), saved.actions[0].mean(axis=0))
        np.testing.assert_array_equal(Editor(self.root).actions[0][:, 0], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            self.editor.save(dest)
        with self.assertRaises(ValueError):
            self.editor.save(self.root / 'nested')


if __name__ == '__main__':
    unittest.main()
