"""Standalone tests: uv run --no-project --with pyarrow python -m unittest discover -s tests -p test_iloha_trajectory_web.py"""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iloha_trajectory_web import JOINT_NAMES, OFFSET, SCALE, catalog, dataset_path, load_episode


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "fixture"
        (self.path / "meta").mkdir(parents=True)
        (self.path / "data" / "chunk-000").mkdir(parents=True)
        self.info = dict(fps=30, total_frames=4, total_episodes=2,
                         features={"action": {"names": list(reversed(JOINT_NAMES))}})
        (self.path / "meta" / "info.json").write_text(json.dumps(self.info), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, indices=(0, 1)):
        rows = [dict(action=list(reversed([OFFSET[j] + SCALE[j] * i for j in range(14)])),
                     episode_index=episode, frame_index=i) for episode in (0, 1) for i in indices]
        pq.write_table(pa.Table.from_pylist(list(reversed(rows))), self.path / "data" / "chunk-000" / "file-000.parquet")

    def test_missing_motion_is_reported(self):
        self.assertFalse(catalog(self.root)[0]["available"])
        with self.assertRaises(FileNotFoundError):
            load_episode(self.root, "fixture", 0)

    def test_episode_filter_order_and_coordinate_mapping(self):
        self.write()
        data = load_episode(self.root, "fixture", 1)
        self.assertEqual(len(data["actions"]), 2)
        self.assertEqual(data["actions"][0], [0] * 14)
        for value in data["actions"][1]:
            self.assertAlmostEqual(value, 1)
        self.assertEqual(data["coordinates"], "iloha")

    def test_missing_and_duplicate_frames_are_rejected(self):
        for indices in ((0, 2), (0, 0)):
            self.write(indices)
            with self.assertRaises(ValueError):
                load_episode(self.root, "fixture", 0)

    def test_path_escape_rejected(self):
        for name in ("../fixture", "..", "/"):
            with self.assertRaises(ValueError):
                dataset_path(self.root, name)


if __name__ == "__main__":
    unittest.main()
