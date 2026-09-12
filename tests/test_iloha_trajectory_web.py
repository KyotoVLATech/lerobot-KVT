"""Standalone tests: uv run --no-project --with pyarrow python -m unittest discover -s tests -p test_iloha_trajectory_web.py"""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iloha_trajectory_web import JOINT_NAMES, OFFSET, SCALE, catalog, dataset_path, load_episode
from iloha_trajectory_dataset import export_dataset


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

    def test_export_roundtrip_v3_metadata_and_separate_settings(self):
        actions = [[i / 10] * 14 for i in range(10)]
        payload = dict(name="merged", coordinates="iloha", fps=30, actions=actions,
                       task="Test merged ideal motion", settings={"actuator": {"current": 0}, "replay": {"base_speed": 8}})
        output = export_dataset(self.root, payload)
        loaded = load_episode(self.root, "merged", 0)
        self.assertEqual(output["frames"], 10)
        self.assertEqual(loaded["task"], payload["task"])
        for actual, expected in zip(loaded["actions"], actions, strict=True):
            for a, b in zip(actual, expected, strict=True):
                self.assertAlmostEqual(a, b, places=6)
        path = self.root / "merged"
        saved = json.loads((path / "trajectory_settings.json").read_text())
        self.assertFalse(saved["speed_applied"])
        self.assertFalse(saved["actuator_limits_applied"])
        self.assertEqual(saved["replay"]["base_speed"], 8)
        self.assertEqual(saved["actuator"]["current"], 0)
        frame = pq.read_table(path / "data/chunk-000/file-000.parquet").to_pylist()[-1]
        self.assertAlmostEqual(frame["timestamp"], .3, places=6)
        self.assertEqual(frame["action"], frame["observation.state"])
        episode = pq.read_table(path / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
        self.assertEqual(episode["dataset_to_index"], 10)
        self.assertEqual(episode["stats/action/count"], [10])
        self.assertEqual(json.loads((path / "meta/info.json").read_text())["codebase_version"], "v3.0")
        original = (path / "data/chunk-000/file-000.parquet").read_bytes()
        with self.assertRaises(FileExistsError):
            export_dataset(self.root, payload)
        self.assertEqual(original, (path / "data/chunk-000/file-000.parquet").read_bytes())

    def test_export_rejects_invalid_targets_and_nonfinite_actions(self):
        payload = dict(name="../escape", coordinates="iloha", fps=30, actions=[[0] * 14] * 2)
        with self.assertRaises(ValueError):
            export_dataset(self.root, payload)
        payload["name"] = "invalid"
        payload["actions"] = [[float("nan")] * 14] * 2
        with self.assertRaises(ValueError):
            export_dataset(self.root, payload)
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
