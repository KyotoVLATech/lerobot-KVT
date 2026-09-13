import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from iloha_dataset_edit import save_dataset
from iloha_trajectory_web import JOINT_NAMES, load_episode


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        (self.source / 'meta/episodes').mkdir(parents=True)
        (self.source / 'data').mkdir()
        names = list(reversed(JOINT_NAMES))
        info = dict(fps=2, total_frames=6, total_episodes=2,
                    features={f: {'names': names} for f in ['action', 'observation.state']})
        (self.source / 'meta/info.json').write_text(json.dumps(info))
        rows = [dict(action=[0.] * 14, **{'observation.state': [1.] * 14},
                     episode_index=ep, frame_index=i, timestamp=i / 2, index=ep*3+i)
                for ep in range(2) for i in range(3)]
        pq.write_table(pa.Table.from_pylist(rows), self.source / 'data/file.parquet')
        pq.write_table(pa.Table.from_pylist([{'episode_index': ep, 'tasks': ['task']} for ep in range(2)]),
                       self.source / 'meta/episodes/file.parquet')
        self.request = dict(dataset='source', name='edited', episode=0, update_state=False,
                            edits=[dict(joint=1, center=.5, sigma=.5, amplitude=.2),
                                   dict(joint=2, center=.5, sigma=.5, amplitude=-.1)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_preservation(self):
        before = (self.source / 'data/file.parquet').read_bytes()
        save_dataset(self.root, self.request)
        self.assertEqual(before, (self.source / 'data/file.parquet').read_bytes())
        original = load_episode(self.root, 'source', 0)['actions']
        edited = load_episode(self.root, 'edited', 0)['actions']
        for i in range(3):
            for j in range(14):
                delta = (.2 if j == 1 else -.1 if j == 2 else 0) * math.exp(-.5*((i/2-.5)/.5)**2)
                self.assertAlmostEqual(edited[i][j], original[i][j]+delta)
        self.assertEqual(load_episode(self.root, 'source', 1)['actions'], load_episode(self.root, 'edited', 1)['actions'])
        a = pq.read_table(self.source / 'data/file.parquet')
        b = pq.read_table(self.root / 'edited/data/file.parquet')
        for key in a.column_names:
            if key != 'action':
                self.assertEqual(a[key], b[key])
        stats = json.loads((self.root / 'edited/meta/stats.json').read_text())
        np.testing.assert_allclose(stats['action']['mean'], np.mean(b['action'].to_pylist(), axis=0))
        ep = pq.read_table(self.root / 'edited/meta/episodes/file.parquet').to_pylist()[0]
        np.testing.assert_allclose(ep['stats/action/mean'], np.mean(b['action'].to_pylist()[:3], axis=0))
        with self.assertRaises(ValueError):
            save_dataset(self.root, self.request)

    def test_right_arm_roundtrip_and_left_arm_preserved(self):
        self.request['edits'] = [dict(joint=8, center=.5, sigma=.5, amplitude=.3),
                                 dict(joint=9, center=.5, sigma=.5, amplitude=-.2)]
        self.request['update_state'] = True
        save_dataset(self.root, self.request)
        original = load_episode(self.root, 'source', 0)['actions']
        edited = load_episode(self.root, 'edited', 0)['actions']
        for i in range(3):
            for j in range(14):
                delta = {8: .3, 9: -.2}.get(j, 0) * math.exp(-.5*((i/2-.5)/.5)**2)
                self.assertAlmostEqual(edited[i][j], original[i][j] + delta)
        self.assertEqual(load_episode(self.root, 'source', 1)['actions'],
                         load_episode(self.root, 'edited', 1)['actions'])
        table = pq.read_table(self.root / 'edited/data/file.parquet')
        np.testing.assert_allclose(np.array(table['observation.state'].to_pylist()) - 1,
                                   table['action'].to_pylist(), atol=1e-12)

    def test_state_delta_and_validation(self):
        self.request['update_state'] = True
        save_dataset(self.root, self.request)
        table = pq.read_table(self.root / 'edited/data/file.parquet')
        np.testing.assert_allclose(np.array(table['observation.state'].to_pylist())-1, table['action'].to_pylist(), atol=1e-12)
        for key, value in [('sigma', 0), ('center', 10), ('amplitude', float('nan')), ('joint', 7)]:
            req = {**self.request, 'name': 'invalid', 'edits': [{**self.request['edits'][0], key: value}]}
            with self.assertRaises(ValueError):
                save_dataset(self.root, req)
            self.assertFalse((self.root / 'invalid').exists())
        for name in ['../escape', 'source', '/tmp/escape']:
            with self.assertRaises(ValueError):
                save_dataset(self.root, {**self.request, 'name': name})


if __name__ == '__main__':
    unittest.main()
