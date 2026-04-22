import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_INDEX_STATS = (
    "min",
    "max",
    "mean",
    "std",
    "count",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
)


@dataclass
class TaskChangePlan:
    root: Path
    original_episode_tasks: dict[int, str]
    new_episode_tasks: dict[int, str]
    task_to_index: dict[str, int]

    @property
    def changed_episodes(self) -> list[int]:
        return [
            episode_index
            for episode_index, task in self.new_episode_tasks.items()
            if self.original_episode_tasks[episode_index] != task
        ]


def find_dataset_root(dataset: str) -> Path:
    candidate = Path(dataset)
    if candidate.exists():
        return candidate.resolve()

    dataset_candidate = Path("datasets") / dataset
    if dataset_candidate.exists():
        return dataset_candidate.resolve()

    raise FileNotFoundError(f"Dataset not found: {dataset}")


def parse_episode_indices(value: str) -> list[int]:
    """Parse comma-separated episode indices and ranges such as '0,2,5-8'."""
    episode_indices: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            episode_indices.update(range(start, end + 1))
        else:
            episode_indices.add(int(part))

    if not episode_indices:
        raise ValueError("No episode indices were provided")

    return sorted(episode_indices)


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text())


def write_info(root: Path, info: dict[str, Any]) -> None:
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def load_tasks(root: Path) -> pd.DataFrame:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks.parquet: {tasks_path}")

    tasks = pd.read_parquet(tasks_path)
    tasks.index.name = "task"
    if "task_index" not in tasks.columns:
        raise ValueError(f"{tasks_path} does not contain a task_index column")
    return tasks


def write_tasks(root: Path, task_to_index: dict[str, int]) -> None:
    tasks = pd.DataFrame(
        {"task_index": list(task_to_index.values())},
        index=pd.Index(task_to_index.keys(), name="task"),
    )
    tasks.to_parquet(root / "meta" / "tasks.parquet")


def episode_parquet_paths(root: Path) -> list[Path]:
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(episodes_dir.rglob("*.parquet"))


def data_parquet_paths(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    return sorted(data_dir.rglob("*.parquet"))


def normalize_task_cell(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        if not value:
            raise ValueError("Encountered an episode with an empty tasks list")
        return str(value[0])

    if isinstance(value, tuple):
        if not value:
            raise ValueError("Encountered an episode with an empty tasks tuple")
        return str(value[0])

    if pd.isna(value):
        raise ValueError("Encountered an episode with a null task")

    return str(value)


def load_episode_tasks(root: Path, tasks: pd.DataFrame) -> dict[int, str]:
    episode_tasks: dict[int, str] = {}
    episode_files = episode_parquet_paths(root)

    if episode_files:
        for parquet_path in episode_files:
            df = pd.read_parquet(parquet_path, columns=["episode_index", "tasks"])
            for row in df.itertuples(index=False):
                episode_index = int(row.episode_index)
                task = normalize_task_cell(row.tasks)
                if episode_index in episode_tasks:
                    raise ValueError(f"Duplicate episode_index in metadata: {episode_index}")
                episode_tasks[episode_index] = task
        return dict(sorted(episode_tasks.items()))

    index_to_task = {int(row.task_index): str(task) for task, row in tasks.iterrows()}
    for parquet_path in data_parquet_paths(root):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for episode_index, task_index in df.groupby("episode_index")["task_index"].first().items():
            task = index_to_task.get(int(task_index))
            if task is None:
                raise ValueError(f"Unknown task_index {task_index} in {parquet_path}")
            episode_tasks[int(episode_index)] = task

    if not episode_tasks:
        raise ValueError("No episodes found in dataset")

    return dict(sorted(episode_tasks.items()))


def ordered_task_mapping(episode_tasks: dict[int, str]) -> dict[str, int]:
    task_to_index: dict[str, int] = {}
    for episode_index in sorted(episode_tasks):
        task = episode_tasks[episode_index]
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
    return task_to_index


def resolve_source_task(tasks: pd.DataFrame, from_task: str | None, from_task_index: int | None) -> str:
    if from_task is not None and from_task_index is not None:
        raise ValueError("Specify only one of --from-task or --from-task-index")

    if from_task is not None:
        if from_task not in tasks.index:
            available = "\n".join(f"  [{int(row.task_index)}] {task}" for task, row in tasks.iterrows())
            raise ValueError(f"Task not found: {from_task}\nAvailable tasks:\n{available}")
        return from_task

    if from_task_index is None:
        raise ValueError("--mode task requires --from-task or --from-task-index")

    matches = [str(task) for task, row in tasks.iterrows() if int(row.task_index) == from_task_index]
    if not matches:
        available = "\n".join(f"  [{int(row.task_index)}] {task}" for task, row in tasks.iterrows())
        raise ValueError(f"Task index not found: {from_task_index}\nAvailable tasks:\n{available}")
    return matches[0]


def build_plan(args: argparse.Namespace) -> TaskChangePlan:
    root = find_dataset_root(args.dataset)
    tasks = load_tasks(root)
    original_episode_tasks = load_episode_tasks(root, tasks)
    new_episode_tasks = dict(original_episode_tasks)

    if args.mode == "all":
        for episode_index in new_episode_tasks:
            new_episode_tasks[episode_index] = args.new_task

    elif args.mode == "episode":
        if args.episodes is None:
            raise ValueError("--mode episode requires --episodes")
        selected_episodes = parse_episode_indices(args.episodes)
        valid_episodes = set(original_episode_tasks)
        invalid_episodes = sorted(set(selected_episodes) - valid_episodes)
        if invalid_episodes:
            raise ValueError(f"Invalid episode indices: {invalid_episodes}")

        for episode_index in selected_episodes:
            new_episode_tasks[episode_index] = args.new_task

    elif args.mode == "task":
        source_task = resolve_source_task(tasks, args.from_task, args.from_task_index)
        selected_episodes = [
            episode_index
            for episode_index, task in original_episode_tasks.items()
            if task == source_task
        ]
        if not selected_episodes:
            raise ValueError(f"No episodes use task: {source_task}")

        for episode_index in selected_episodes:
            new_episode_tasks[episode_index] = args.new_task

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return TaskChangePlan(
        root=root,
        original_episode_tasks=original_episode_tasks,
        new_episode_tasks=new_episode_tasks,
        task_to_index=ordered_task_mapping(new_episode_tasks),
    )


def weighted_quantile(values: list[int], weights: list[int], quantile: float) -> float:
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights, strict=False))
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return 0.0

    threshold = quantile * total
    cumulative = 0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def task_index_stats(values: list[int], weights: list[int]) -> dict[str, list[int | float]]:
    total = sum(weights)
    if total <= 0:
        return {
            "min": [0],
            "max": [0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
            "q01": [0.0],
            "q10": [0.0],
            "q50": [0.0],
            "q90": [0.0],
            "q99": [0.0],
        }

    mean = sum(value * weight for value, weight in zip(values, weights, strict=False)) / total
    variance = sum(weight * ((value - mean) ** 2) for value, weight in zip(values, weights, strict=False)) / total

    return {
        "min": [int(min(values))],
        "max": [int(max(values))],
        "mean": [float(mean)],
        "std": [float(variance**0.5)],
        "count": [int(total)],
        "q01": [weighted_quantile(values, weights, 0.01)],
        "q10": [weighted_quantile(values, weights, 0.10)],
        "q50": [weighted_quantile(values, weights, 0.50)],
        "q90": [weighted_quantile(values, weights, 0.90)],
        "q99": [weighted_quantile(values, weights, 0.99)],
    }


def task_index_stat_cell(stat_name: str, task_index: int, episode_length: int) -> list[int | float]:
    if stat_name in {"min", "max"}:
        return [int(task_index)]
    if stat_name == "count":
        return [int(episode_length)]
    if stat_name == "std":
        return [0.0]
    return [float(task_index)]


def backup_files(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = root / "backups" / "change_task" / timestamp
    paths = [root / "meta" / "info.json", root / "meta" / "tasks.parquet"]
    paths.extend(episode_parquet_paths(root))
    paths.extend(data_parquet_paths(root))

    for src_path in paths:
        if not src_path.exists():
            continue
        dst_path = backup_root / src_path.relative_to(root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    return backup_root


def update_data_files(plan: TaskChangePlan) -> None:
    episode_to_task_index = {
        episode_index: plan.task_to_index[task]
        for episode_index, task in plan.new_episode_tasks.items()
    }

    for parquet_path in data_parquet_paths(plan.root):
        df = pd.read_parquet(parquet_path)
        mapped = df["episode_index"].map(episode_to_task_index)
        if mapped.isna().any():
            unknown = sorted(df.loc[mapped.isna(), "episode_index"].unique().tolist())
            raise ValueError(f"{parquet_path} contains unknown episode indices: {unknown}")

        new_task_index = mapped.astype("int64")
        if not df["task_index"].equals(new_task_index):
            df["task_index"] = new_task_index
            df.to_parquet(parquet_path, index=False)


def update_episode_files(plan: TaskChangePlan) -> list[int]:
    episode_lengths: dict[int, int] = {}
    episode_to_task_index = {
        episode_index: plan.task_to_index[task]
        for episode_index, task in plan.new_episode_tasks.items()
    }

    for parquet_path in episode_parquet_paths(plan.root):
        df = pd.read_parquet(parquet_path)
        if "tasks" in df.columns:
            df["tasks"] = df["episode_index"].apply(
                lambda episode_index: [plan.new_episode_tasks[int(episode_index)]]
            )

        for stat_name in TASK_INDEX_STATS:
            column = f"stats/task_index/{stat_name}"
            if column not in df.columns:
                continue
            df[column] = df.apply(
                lambda row: task_index_stat_cell(
                    stat_name,
                    episode_to_task_index[int(row["episode_index"])],
                    int(row["length"]),
                ),
                axis=1,
            )

        for row in df[["episode_index", "length"]].itertuples(index=False):
            episode_lengths[int(row.episode_index)] = int(row.length)

        df.to_parquet(parquet_path, index=False)

    return [episode_lengths[episode_index] for episode_index in sorted(episode_lengths)]


def update_stats(root: Path, plan: TaskChangePlan, episode_lengths: list[int]) -> None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return

    stats = json.loads(stats_path.read_text())
    task_indices = [
        plan.task_to_index[plan.new_episode_tasks[episode_index]]
        for episode_index in sorted(plan.new_episode_tasks)
    ]

    if len(task_indices) == len(episode_lengths):
        stats["task_index"] = task_index_stats(task_indices, episode_lengths)
    else:
        stats["task_index"] = compute_task_index_stats_from_data(root)

    stats_path.write_text(json.dumps(stats, indent=4) + "\n")


def compute_task_index_stats_from_data(root: Path) -> dict[str, list[int | float]]:
    counts: dict[int, int] = {}
    for parquet_path in data_parquet_paths(root):
        task_counts = pd.read_parquet(parquet_path, columns=["task_index"])["task_index"].value_counts()
        for task_index, count in task_counts.items():
            counts[int(task_index)] = counts.get(int(task_index), 0) + int(count)

    values = sorted(counts)
    weights = [counts[value] for value in values]
    return task_index_stats(values, weights)


def apply_plan(plan: TaskChangePlan) -> None:
    write_tasks(plan.root, plan.task_to_index)
    update_data_files(plan)
    episode_lengths = update_episode_files(plan)
    update_stats(plan.root, plan, episode_lengths)

    info = load_info(plan.root)
    info["total_tasks"] = len(plan.task_to_index)
    write_info(plan.root, info)


def print_summary(plan: TaskChangePlan) -> None:
    changed_episodes = plan.changed_episodes
    print(f"Dataset: {plan.root}")
    print(f"Total episodes: {len(plan.new_episode_tasks)}")
    print(f"Changed episodes: {len(changed_episodes)}")
    if changed_episodes:
        preview = ", ".join(map(str, changed_episodes[:20]))
        suffix = " ..." if len(changed_episodes) > 20 else ""
        print(f"Changed episode indices: {preview}{suffix}")

    print("New task table:")
    for task, task_index in plan.task_to_index.items():
        print(f"  [{task_index}] {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Change task instruction text in a LeRobot dataset in-place.",
    )
    parser.add_argument("dataset", help="Dataset root path, or a name under ./datasets")
    parser.add_argument(
        "--mode",
        choices=("all", "episode", "task"),
        required=True,
        help="all: rewrite all episodes, episode: rewrite selected episodes, task: rewrite one task type",
    )
    parser.add_argument("--new-task", required=True, help="New task instruction text")
    parser.add_argument(
        "--episodes",
        help="Episode indices for --mode episode. Example: '0,2,5-8'",
    )
    parser.add_argument("--from-task", help="Original task text for --mode task")
    parser.add_argument("--from-task-index", type=int, help="Original task_index for --mode task")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned changes without writing")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy changed metadata/data parquet files under backups/change_task before writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    print_summary(plan)

    if args.dry_run:
        print("Dry run: no files were changed.")
        return

    if not plan.changed_episodes:
        print("No task changes needed.")
        return

    if args.backup:
        backup_root = backup_files(plan.root)
        print(f"Backup written to: {backup_root}")

    apply_plan(plan)
    print("Task metadata and task_index columns were updated.")


if __name__ == "__main__":
    main()
