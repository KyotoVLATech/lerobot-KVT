"""Standalone tests: uv run --no-project --with numpy python -m unittest discover -s tests -p test_iloha_trajectory_replay.py

The parity test additionally needs Node to run the browser implementation.
"""
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iloha_trajectory_replay import (build_trajectory, load_settings, parse_settings,
                                     prepare_clip, replay_intervals, stitch, unwrap)

REPO = Path(__file__).resolve().parents[1]
CORE = (REPO / "tools" / "iloha_trajectory_web" / "core.mjs").as_uri()
MODES = ("cut", "crossfade", "linear", "smooth")


def constant(value, frames=61, fps=30.0):
    return dict(name="fixture", fps=fps, actions=np.full((frames, 14), float(value)))


def clip(source, start=None, end=None, **replay):
    settings = dict(base_speed=1.0, max_speedup=1.0, gripper_margin=0.5,
                    speedup_distance=1.0, gripper_threshold=1e-4)
    settings.update(replay)
    end = (len(source["actions"]) - 1) / source["fps"] if end is None else end
    return dict(name=source["name"], actions=source["actions"], fps=source["fps"],
                start=0.0 if start is None else start, end=end, replay=settings)


def document(clips, mode="crossfade", blend=1.0, fps=30, **extra):
    return dict(schema_version=3, speed_applied=True, clips=clips,
                edit=dict(mode=mode, blend=blend, fps=fps, time_basis="speed_adjusted_seconds"),
                trajectory=dict(fps=fps), **extra)


def clip_settings(dataset="fixture", start=0.0, end=2.0, **replay):
    return dict(dataset=dataset, episode=0, start=start, end=end, replay=replay or None,
                synthetic=False)


class SpeedTests(unittest.TestCase):
    def test_gripper_activity_and_neighbours_keep_the_base_speed(self):
        rows = constant(0, 301)["actions"].copy()
        rows[151:, 6] = 1
        intervals = replay_intervals(rows, 30, base_speed=2, max_speedup=3,
                                     gripper_margin=.5, speedup_distance=1)
        for index in (149, 150, 151):
            self.assertAlmostEqual(intervals[index], 1 / 60)
        self.assertAlmostEqual(intervals[0], 1 / 180)
        self.assertAlmostEqual(intervals[-1], 1 / 180)
        self.assertTrue(1 / 180 < intervals[120] < 1 / 60)

    def test_without_gripper_activity_everything_uses_the_maximum(self):
        intervals = replay_intervals(constant(0)["actions"], 30, base_speed=2, max_speedup=3)
        self.assertTrue(np.allclose(intervals, 1 / 180))

    def test_invalid_speed_settings_are_rejected(self):
        rows = constant(0)["actions"]
        for key, value in (("fps", 0), ("base_speed", -1), ("max_speedup", .5),
                           ("gripper_margin", -1), ("speedup_distance", float("inf")),
                           ("gripper_threshold", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay_intervals(rows, **{"fps": 30, key: value})

    def test_arm_angles_are_unwrapped_but_grippers_are_not(self):
        rows = constant(0, 2)["actions"].copy()
        rows[0][0], rows[1][0], rows[1][6] = math.pi - .1, -math.pi + .1, 5
        result = unwrap(rows)
        self.assertAlmostEqual(result[1][0] - result[0][0], .2)
        self.assertAlmostEqual(result[1][6], 5)


class StitchTests(unittest.TestCase):
    def test_each_clip_keeps_its_own_speed(self):
        result = stitch([clip(constant(0), base_speed=2), clip(constant(.2))], "cut", 0, 60)
        self.assertAlmostEqual(result["segments"][0]["end"] - result["segments"][0]["start"], 1)
        self.assertAlmostEqual(result["segments"][1]["end"] - result["segments"][1]["start"], 2)

    def test_crossfade_overlaps_in_time_and_keeps_the_endpoints(self):
        clips = [clip(constant(0, 121), 1, 3), clip(constant(1, 121), .5, 3.5)]
        result = stitch(clips, "crossfade", 1, 60)
        self.assertAlmostEqual(result["duration"], 4)
        self.assertAlmostEqual(result["actions"][0][0], 0)
        self.assertAlmostEqual(result["actions"][-1][0], 1)
        self.assertAlmostEqual(result["actions"][90][0], .5)
        self.assertAlmostEqual(result["boundaries"][0]["start"], 1)
        self.assertAlmostEqual(result["boundaries"][0]["end"], 2)
        with self.assertRaises(ValueError):
            stitch(clips, "crossfade", 3, 60)

    def test_linear_and_smooth_transitions_insert_the_blend(self):
        clips = [clip(constant(0)), clip(constant(1))]
        linear = stitch(clips, "linear", 1, 60)
        self.assertAlmostEqual(linear["duration"], 5)
        self.assertAlmostEqual(linear["actions"][150][0], .5)
        self.assertAlmostEqual(linear["actions"][120][0], 0)
        self.assertAlmostEqual(linear["actions"][180][0], 1)
        smooth = stitch(clips, "smooth", 1, 60)
        self.assertAlmostEqual(smooth["actions"][150][0], .5)
        self.assertLess(smooth["actions"][121][0], linear["actions"][121][0])

    def test_grippers_stay_inside_their_valid_range(self):
        source = constant(0, 61)
        source["actions"] = source["actions"].copy()
        source["actions"][:, 6] = 1.0
        result = stitch([clip(source), clip(constant(0))], "smooth", 1, 60)
        self.assertTrue(np.all(result["actions"][:, [6, 13]] >= 0))
        self.assertTrue(np.all(result["actions"][:, [6, 13]] <= 1))

    def test_invalid_edits_are_rejected(self):
        clips = [clip(constant(0))]
        for mode, blend, fps in (("bogus", 1, 60), ("cut", -1, 60), ("cut", 1, 0), ("cut", 1, 500)):
            with self.subTest(mode=mode, blend=blend, fps=fps), self.assertRaises(ValueError):
                stitch(clips, mode, blend, fps)
        with self.assertRaises(ValueError):
            prepare_clip(constant(0)["actions"], 30, 0, 99, clip(constant(0))["replay"], 60)


class SettingsTests(unittest.TestCase):
    def test_schema_2_files_are_replayed_with_their_speed_settings(self):
        legacy = dict(schema_version=2, speed_applied=False,
                      clips=[clip_settings(end=1.0, base_speed=2.0)],
                      edit=dict(mode="cut", blend=0, fps=30, time_basis="original_recording_seconds"))
        settings = parse_settings(legacy)
        self.assertEqual(settings["fps"], 30)
        self.assertEqual(settings["clips"][0]["replay"]["base_speed"], 2.0)
        # Unspecified speed settings fall back to the studio defaults.
        self.assertEqual(settings["clips"][0]["replay"]["max_speedup"], 2.0)
        self.assertIsNone(settings["expected_frames"])

    def test_invalid_documents_are_rejected(self):
        for broken in (
            dict(schema_version=1, clips=[clip_settings()], edit=dict(mode="cut", blend=0)),
            dict(schema_version=3, clips=[], edit=dict(mode="cut", blend=0)),
            dict(schema_version=3, clips=[clip_settings(dataset="../escape")], edit=dict(mode="cut", blend=0)),
            dict(schema_version=3, clips=[clip_settings(end=0.0)], edit=dict(mode="cut", blend=0)),
            dict(schema_version=3, clips=[clip_settings()], edit=dict(mode="bogus", blend=0)),
            dict(schema_version=3, clips=[clip_settings()], edit=dict(mode="cut", blend=float("nan"))),
            dict(schema_version=3, clips=[clip_settings()], edit=dict(mode="cut", blend=0, fps=0)),
            dict(schema_version=3, clips=[clip_settings()]),
        ):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                parse_settings(broken)

    def test_demo_clips_are_refused_for_the_robot(self):
        broken = document([{**clip_settings(), "synthetic": True}])
        with self.assertRaises(ValueError):
            parse_settings(broken)

    def test_settings_rebuild_the_recorded_frame_count(self):
        source = np.stack([np.full(14, i / 30) for i in range(61)])
        loaded = []

        def load_source(name, episode):
            loaded.append((name, episode))
            return source, 30.0

        settings = parse_settings(document([clip_settings(end=2.0), clip_settings(end=2.0)],
                                           mode="cut", blend=0, fps=30))
        trajectory = build_trajectory(settings, load_source)
        self.assertEqual(len(trajectory["actions"]), 122)
        self.assertEqual(loaded, [("fixture", 0), ("fixture", 0)])
        # A dataset that no longer matches the export must not move the robot.
        settings["expected_frames"] = 999
        with self.assertRaises(ValueError):
            build_trajectory(settings, load_source)

    def test_files_are_read_and_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trajectory_settings.json"
            path.write_text(json.dumps(document([clip_settings()])), encoding="utf-8")
            self.assertEqual(load_settings(path)["mode"], "crossfade")
            with self.assertRaises(FileNotFoundError):
                load_settings(Path(temp) / "missing.json")


@unittest.skipUnless(shutil.which("node"), "Node is required to run the browser implementation")
class BrowserParityTests(unittest.TestCase):
    """The robot must move exactly like the preview, so both must agree bit for bit."""

    def sources(self):
        generator = np.random.default_rng(7)
        sources = {}
        for name, frames, fps in (("a", 211, 30.0), ("b", 173, 30.0)):
            time = np.arange(frames)[:, None] / fps
            actions = 2.5 * np.sin(time * generator.uniform(.5, 3, 14) + generator.uniform(0, 6, 14))
            # Gripper commands drive the speed schedule: hold, then move, then hold.
            for column in (6, 13):
                actions[:, column] = np.clip(np.abs(np.sin(time[:, 0] * .7)) - .2, 0, 1)
                actions[:frames // 3, column] = actions[frames // 3, column]
            sources[name] = dict(name=name, fps=fps, coordinates="iloha", actions=actions)
        return sources

    def javascript(self, sources, clips, mode, blend, fps, temp):
        job = dict(sources={k: {**v, "actions": v["actions"].tolist()} for k, v in sources.items()},
                   clips=clips, mode=mode, blend=blend, fps=fps)
        (temp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (temp / "run.mjs").write_text(
            "import fs from 'node:fs';\n"
            f"import {{stitch}} from '{CORE}';\n"
            "const job=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));\n"
            "const clips=job.clips.map(c=>({source:job.sources[c.dataset],start:c.start,end:c.end,replay:c.replay}));\n"
            "const t=stitch(clips,{mode:job.mode,blend:job.blend,fps:job.fps});\n"
            "fs.writeFileSync(process.argv[3],JSON.stringify({actions:t.actions,duration:t.duration,"
            "segments:t.segments,boundaries:t.boundaries}));\n", encoding="utf-8")
        subprocess.run(["node", str(temp / "run.mjs"), str(temp / "job.json"), str(temp / "out.json")],
                       check=True, capture_output=True)
        return json.loads((temp / "out.json").read_text(encoding="utf-8"))

    def test_every_transition_matches_the_browser_exactly(self):
        sources = self.sources()
        clips = [
            clip_settings("a", 0.7, 5.5, base_speed=1.3, max_speedup=2.5,
                          gripper_margin=.4, speedup_distance=1.1, gripper_threshold=1e-4),
            clip_settings("b", 0.0, 4.0, base_speed=.8, max_speedup=3.0,
                          gripper_margin=.2, speedup_distance=.7, gripper_threshold=1e-3),
        ]
        with tempfile.TemporaryDirectory() as name:
            temp = Path(name)
            for mode, fps in ((mode, fps) for mode in MODES for fps in (30, 60)):
                with self.subTest(mode=mode, fps=fps):
                    expected = self.javascript(sources, clips, mode, 1.2, fps, temp)
                    settings = parse_settings(document(clips, mode, 1.2, fps))
                    actual = build_trajectory(
                        settings, lambda n, e: (sources[n]["actions"], sources[n]["fps"]))
                    reference = np.asarray(expected["actions"], dtype=np.float64)
                    self.assertEqual(reference.shape, actual["actions"].shape)
                    self.assertEqual(np.max(np.abs(reference - actual["actions"])), 0.0)
                    self.assertEqual(expected["duration"], actual["duration"])
                    self.assertEqual(expected["boundaries"], actual["boundaries"])
                    for a, b in zip(expected["segments"], actual["segments"], strict=True):
                        self.assertEqual({k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                                          for k, v in a.items()},
                                         {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                                          for k, v in b.items()})


if __name__ == "__main__":
    unittest.main()
