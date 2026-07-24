#!/usr/bin/env python3
"""Interactive launcher recipe selector for RL-100."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


ROUTE_AXES: List[Tuple[str, str]] = [
    ("family", "Policy family"),
    ("stage", "Training stage"),
    ("modality", "Observation modality"),
    ("control", "Control mode"),
]

ADVISORY_AXES: List[Tuple[str, str, List[str]]] = [
    ("extraction", "Policy extraction", ["PG", "PG + IDQL"]),
    ("distill", "One-step distillation", ["No", "Yes"]),
]

PREFERRED_ORDER = {
    "family": ["DDIM", "Flow"],
    "stage": ["Offline", "Online"],
    "modality": ["3D", "2D"],
    "control": ["Chunk action", "Single action"],
}


@dataclass(frozen=True)
class Recipe:
    tags: Dict[str, str]
    command: str
    note: str


RECIPES: List[Recipe] = [
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Offline",
            "modality": "3D",
            "control": "Single action",
        },
        command="bash scripts/Diffusion/Offline/3D/train_policy.sh rl100 adroit_door_medium 0112 100",
        note="3D DDIM offline policy-gradient training.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Offline",
            "modality": "3D",
            "control": "Single action",
        },
        command="bash scripts/Flow/Offline/3D/train_policy_flow.sh rl100 adroit_door_medium 0112 100",
        note="3D flow offline policy-gradient training.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Offline",
            "modality": "3D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Diffusion/Offline/3D/train_policy_chunk_two_stage.sh "
            "rl100 adroit_door_medium 0112 100 4"
        ),
        note="3D DDIM chunk-action offline training.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Offline",
            "modality": "3D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Flow/Offline/3D/train_policy_chunk_two_stage_flow.sh "
            "rl100 adroit_door_medium 0112 100 4"
        ),
        note="3D flow chunk-action offline training.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Offline",
            "modality": "2D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D DDIM chunk-action offline training.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Offline",
            "modality": "2D",
            "control": "Single action",
        },
        command=(
            "bash scripts/Diffusion/Offline/2D/train_policy_image_unet_two_stage.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D DDIM single-action offline training (horizon=3, n_action_steps=1).",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Offline",
            "modality": "2D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Flow/Offline/2D/train_policy_image_unet_flow_chunk_two_stage.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D flow chunk-action offline training.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Offline",
            "modality": "2D",
            "control": "Single action",
        },
        command=(
            "bash scripts/Flow/Offline/2D/train_policy_image_unet_two_stage_flow.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D flow single-action offline training (horizon=3, n_action_steps=1).",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Online",
            "modality": "3D",
            "control": "Single action",
        },
        command="bash scripts/Diffusion/Online/3D/train_policy_online_cm_vec.sh rl100 adroit_door_medium 0112 100",
        note="3D DDIM online RL.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Online",
            "modality": "3D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Diffusion/Online/3D/train_policy_online_cm_chunk.sh "
            "rl100 adroit_door_medium 0112 100 8"
        ),
        note="3D DDIM chunk-action online RL.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Online",
            "modality": "3D",
            "control": "Chunk action",
        },
        command="bash scripts/Flow/Online/3D/train_policy_online_flow_chunk.sh rl100 adroit_door_medium 0112 100",
        note="3D flow chunk-action online RL.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Online",
            "modality": "3D",
            "control": "Single action",
        },
        command=(
            "bash scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh "
            "rl100 adroit_door_medium 0112 100 8"
        ),
        note="Verified 3D flow online recipe; configure online distillation inside the launcher.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Online",
            "modality": "2D",
            "control": "Single action",
        },
        command=(
            "bash scripts/Diffusion/Online/2D/train_policy_image_unet_online_vec.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D DDIM online RL.",
    ),
    Recipe(
        tags={
            "family": "DDIM",
            "stage": "Online",
            "modality": "2D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Diffusion/Online/2D/train_policy_image_unet_online_chunk_vec.sh "
            "rl100 adroit_door_medium 0112 100 8"
        ),
        note="2D DDIM chunk-action online RL.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Online",
            "modality": "2D",
            "control": "Single action",
        },
        command=(
            "bash scripts/Flow/Online/2D/train_policy_image_unet_online_flow_vec.sh "
            "rl100 adroit_door_medium 0112 100"
        ),
        note="2D flow online RL.",
    ),
    Recipe(
        tags={
            "family": "Flow",
            "stage": "Online",
            "modality": "2D",
            "control": "Chunk action",
        },
        command=(
            "bash scripts/Flow/Online/2D/train_policy_image_unet_online_flow_chunk_vec.sh "
            "rl100 adroit_door_medium 0112 100 8"
        ),
        note="2D flow chunk-action online RL.",
    ),
]


def available_values(recipes: List[Recipe], axis: str) -> List[str]:
    values = {recipe.tags[axis] for recipe in recipes}
    preferred = PREFERRED_ORDER.get(axis, [])
    return [value for value in preferred if value in values]


def choose(axis_label: str, values: List[str]) -> str:
    print(f"\n{axis_label}:")
    for i, value in enumerate(values, 1):
        print(f"  {i}. {value}")
    while True:
        raw = input("Select number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(values):
            return values[int(raw) - 1]
        print(f"Please enter a number from 1 to {len(values)}.")


def main() -> None:
    print("RL-100 interactive command selector")
    print("Only valid remaining options are shown at each step.")

    remaining = RECIPES
    selected: Dict[str, str] = {}

    for axis, label in ROUTE_AXES:
        values = available_values(remaining, axis)
        value = choose(label, values)
        selected[axis] = value
        remaining = [recipe for recipe in remaining if recipe.tags[axis] == value]

        if len(remaining) == 1:
            # Still show the remaining axes as fixed choices for clarity.
            for next_axis, next_label in ROUTE_AXES[ROUTE_AXES.index((axis, label)) + 1 :]:
                selected[next_axis] = remaining[0].tags[next_axis]
                print(f"\n{next_label}:")
                print(f"  1. {selected[next_axis]}")
                print("Selected number: 1")
            break

    for axis, label, values in ADVISORY_AXES:
        selected[axis] = choose(label, values)

    print("\nSelected recipe:")
    for axis, label in ROUTE_AXES:
        print(f"  {label}: {selected[axis]}")
    for axis, label, _ in ADVISORY_AXES:
        print(f"  {label}: {selected[axis]}")

    print("\nCommand:")
    print(remaining[0].command)
    print(f"\nNote: {remaining[0].note}")
    print("\nLauncher setting suggestions:")
    if selected["extraction"] == "PG + IDQL":
        print("  Use the launcher flags for PG + IDQL-style extraction.")
    else:
        print("  Use the launcher flags for PG policy-gradient extraction.")

    if selected["distill"] == "Yes":
        if selected["stage"] == "Offline":
            print(
                "  For offline distillation, first finish the offline sweep and parameter "
                "selection, then set distill_phase='after_offline'."
            )
        else:
            print("  For online distillation, set distill_phase='online'.")
    else:
        print("  Keep one-step distillation disabled in the launcher.")


if __name__ == "__main__":
    main()
