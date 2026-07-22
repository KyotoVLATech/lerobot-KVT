#!/usr/bin/env python3

from collections.abc import Mapping
from typing import Any

import numpy as np

from lerobot.utils.feature_utils import hw_to_dataset_features

DEPTH_KEY_SUFFIX = "_depth"


def read_camera_frame(camera: Any, max_age_ms: int, *, depth: bool = False) -> np.ndarray:
    """Read the latest RGB or depth frame, falling back to blocking camera APIs."""
    try:
        latest_method = camera.read_latest_depth if depth else camera.read_latest
        return latest_method(max_age_ms=max_age_ms)
    except Exception:
        try:
            async_method = camera.async_read_depth if depth else camera.async_read
            return async_method(timeout_ms=max_age_ms)
        except Exception:
            sync_method = camera.read_depth if depth else camera.read
            return sync_method()


def capture_camera_observation(cameras: Mapping[str, Any], max_age_ms: int) -> dict[str, np.ndarray]:
    """Capture all enabled RGB and depth streams using LeRobot's camera key convention."""
    observation = {}
    for name, camera in cameras.items():
        if getattr(camera, "use_rgb", True):
            observation[name] = read_camera_frame(camera, max_age_ms)
        if getattr(camera, "use_depth", False):
            observation[f"{name}{DEPTH_KEY_SUFFIX}"] = read_camera_frame(
                camera, max_age_ms, depth=True
            )
    return observation


def make_camera_dataset_features(
    camera_configs: Mapping[str, Mapping[str, Any]],
    *,
    height: int,
    width: int,
) -> dict[str, dict]:
    """Build canonical LeRobot RGB/depth video features for the configured cameras."""
    hardware_features = {}
    for name, config in camera_configs.items():
        if config.get("use_rgb", True):
            hardware_features[name] = (height, width, 3)
        if config.get("use_depth", False):
            hardware_features[f"{name}{DEPTH_KEY_SUFFIX}"] = (height, width, 1)

    return hw_to_dataset_features(hardware_features, prefix="observation", use_video=True)
