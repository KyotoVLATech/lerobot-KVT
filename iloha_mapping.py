#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint naming and coordinate mapping between Iloha hardware and Aloha datasets."""

from __future__ import annotations

import numpy as np


JOINT_NAMES = tuple(f"joint_{i}" for i in range(14))

# Values are ordered as joint_0 ... joint_13.
# The dataset/policy side follows Aloha coordinates:
#   aloha = ALOHA_OFFSET + ALOHA_FROM_ILOHA_SCALE * iloha
ALOHA_OFFSET = np.array(
    [
        0.0,
        -1.0,
        1.15,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        1.15,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)
ALOHA_FROM_ILOHA_SCALE = np.array(
    [
        -1.0, # left_waist/joint_0
        -1.0, # left_shoulder/joint_1
        1.0,  # left_elbow/joint_2
        -1.0, # left_forearm_roll/joint_3
        1.0,  # left_wrist_angle/joint_4
        -1.0, # left_wrist_rotate/joint_5
        -1.0, # left_gripper/joint_6
        -1.0, # right_waist/joint_7
        -1.0, # right_shoulder/joint_8
        1.0,  # right_elbow/joint_9
        -1.0, # right_forearm_roll/joint_10
        1.0,  # right_wrist_angle/joint_11
        -1.0, # right_wrist_rotate/joint_12
        -1.0, # right_gripper/joint_13
    ],
    dtype=np.float32,
)


def _as_joint_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (14,):
        raise ValueError(f"Expected a 14-element joint vector, got shape {array.shape}")
    return array


def iloha_to_aloha(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert an Iloha joint vector to Aloha dataset/policy coordinates."""
    array = _as_joint_array(values)
    return ALOHA_OFFSET + ALOHA_FROM_ILOHA_SCALE * array


def aloha_to_iloha(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert an Aloha dataset/policy joint vector to Iloha hardware coordinates."""
    array = _as_joint_array(values)
    return (array - ALOHA_OFFSET) / ALOHA_FROM_ILOHA_SCALE
