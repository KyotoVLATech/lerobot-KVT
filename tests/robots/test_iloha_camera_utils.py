import numpy as np

from iloha_camera_utils import capture_camera_observation, make_camera_dataset_features


class FakeRGBDCamera:
    use_rgb = True
    use_depth = True

    def __init__(self) -> None:
        self.color = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.full((480, 640, 1), 1000, dtype=np.uint16)

    def read_latest(self, max_age_ms: int) -> np.ndarray:
        return self.color

    def read_latest_depth(self, max_age_ms: int) -> np.ndarray:
        return self.depth


def test_capture_camera_observation_includes_rgb_and_depth() -> None:
    camera = FakeRGBDCamera()

    observation = capture_camera_observation({"wrist": camera}, max_age_ms=250)

    assert observation["wrist"] is camera.color
    assert observation["wrist_depth"] is camera.depth


def test_make_camera_dataset_features_marks_depth_stream() -> None:
    features = make_camera_dataset_features(
        {"wrist": {"use_depth": True}},
        height=480,
        width=640,
    )

    rgb_feature = features["observation.images.wrist"]
    depth_feature = features["observation.images.wrist_depth"]
    assert rgb_feature["shape"] == (480, 640, 3)
    assert rgb_feature["info"]["is_depth_map"] is False
    assert depth_feature["shape"] == (480, 640, 1)
    assert depth_feature["info"]["is_depth_map"] is True
