import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.utils import flatten_dict, write_info, write_stats
from lerobot.datasets.video_utils import encode_video_frames


@dataclass
class VideoIssue:
    packet_index: int
    decoded_frames_before_error: int
    error_type: str
    message: str


@dataclass
class VideoCheckResult:
    video_path: Path
    frame_count: int
    expected_frames: int | None
    issues: list[VideoIssue]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


@dataclass
class EpisodeSource:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    data_path: Path
    tasks: list[str]


@dataclass
class VideoSegment:
    video_key: str
    chunk_index: int
    file_index: int
    path: Path
    start_frame: int
    end_frame: int


CODEC_TO_ENCODER = {
    "av1": "libsvtav1",
    "libsvtav1": "libsvtav1",
    "libaom-av1": "libsvtav1",
    "h264": "h264",
    "hevc": "hevc",
    "h265": "hevc",
}


def find_dataset_root(dataset_name: str) -> Path:
    candidate = Path(dataset_name)
    if candidate.exists():
        return candidate.resolve()

    datasets_candidate = Path("datasets") / dataset_name
    if datasets_candidate.exists():
        return datasets_candidate.resolve()

    raise FileNotFoundError(f"Dataset not found: {dataset_name}")


def iter_video_files(dataset_root: Path) -> list[Path]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos directory not found: {videos_dir}")
    return sorted(videos_dir.glob("**/*.mp4"))


def iter_data_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    return sorted(data_dir.glob("chunk-*/*.parquet"))


def parse_chunk_file(path: Path) -> tuple[int, int]:
    chunk_index = int(path.parent.name.removeprefix("chunk-"))
    file_index = int(path.stem.removeprefix("file-"))
    return chunk_index, file_index


def load_info(dataset_root: Path) -> dict:
    with open(dataset_root / "meta" / "info.json") as f:
        return json.load(f)


def video_keys_from_info(info: dict) -> list[str]:
    return [key for key, feature in info["features"].items() if feature["dtype"] == "video"]


def non_video_features(info: dict) -> dict:
    return {
        key: feature
        for key, feature in info["features"].items()
        if feature["dtype"] not in {"image", "video"}
    }


def load_task_by_index(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}

    tasks = pd.read_parquet(tasks_path)
    mapping: dict[int, str] = {}
    for task, row in tasks.iterrows():
        mapping[int(row["task_index"])] = str(task)
    return mapping


def check_video(video_path: Path) -> VideoCheckResult:
    issues: list[VideoIssue] = []
    frame_count = 0

    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        expected_frames = int(stream.frames) if stream.frames else None

        for packet_index, packet in enumerate(container.demux(stream)):
            try:
                frames = packet.decode()
            except Exception as exc:
                issues.append(
                    VideoIssue(
                        packet_index=packet_index,
                        decoded_frames_before_error=frame_count,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            frame_count += len(frames)

    return VideoCheckResult(
        video_path=video_path,
        frame_count=frame_count,
        expected_frames=expected_frames,
        issues=issues,
    )


def _choose_output_codec(source_stream: av.video.stream.VideoStream, preferred_codec: str | None = None) -> str:
    codec_name = preferred_codec or source_stream.codec_context.name
    return CODEC_TO_ENCODER.get(codec_name.lower(), "h264")


def _source_pix_fmt(source_stream: av.video.stream.VideoStream, preferred_pix_fmt: str | None = None) -> str:
    if preferred_pix_fmt:
        return preferred_pix_fmt
    fmt = source_stream.codec_context.format
    return fmt.name if fmt is not None else "yuv420p"


def repair_video(
    video_path: Path,
    keep_backup: bool = True,
    preferred_codec: str | None = None,
    preferred_pix_fmt: str | None = None,
) -> tuple[int, int]:
    temp_fd, temp_name = tempfile.mkstemp(suffix=".mp4", dir=str(video_path.parent))
    os.close(temp_fd)
    Path(temp_name).unlink(missing_ok=True)

    replacements = 0
    written_frames = 0

    with av.open(str(video_path), "r") as container:
        input_stream = container.streams.video[0]
        fps = input_stream.average_rate
        if fps is None:
            fps = Fraction(30, 1)

        output_codec = _choose_output_codec(input_stream, preferred_codec)
        output_pix_fmt = _source_pix_fmt(input_stream, preferred_pix_fmt)

        with tempfile.TemporaryDirectory(dir=str(video_path.parent)) as frames_dir:
            frames_path = Path(frames_dir)
            last_good_image = None

            for packet in container.demux(input_stream):
                try:
                    frames = packet.decode()
                except Exception:
                    if last_good_image is None:
                        continue

                    frame_path = frames_path / f"frame-{written_frames:06d}.png"
                    last_good_image.save(frame_path)
                    replacements += 1
                    written_frames += 1
                    continue

                for frame in frames:
                    image = frame.to_image()
                    frame_path = frames_path / f"frame-{written_frames:06d}.png"
                    image.save(frame_path)
                    last_good_image = image.copy()
                    written_frames += 1

            encode_video_frames(
                frames_path,
                temp_name,
                fps=int(round(float(fps))),
                vcodec=output_codec,
                pix_fmt=output_pix_fmt,
                overwrite=True,
            )

    backup_path = video_path.with_suffix(video_path.suffix + ".bak")
    if keep_backup:
        shutil.move(str(video_path), str(backup_path))
    else:
        video_path.unlink()
    shutil.move(temp_name, str(video_path))

    return replacements, written_frames


def build_episode_sources(dataset_root: Path) -> list[EpisodeSource]:
    task_by_index = load_task_by_index(dataset_root)
    episodes: dict[int, EpisodeSource] = {}

    for data_path in iter_data_files(dataset_root):
        chunk_index, file_index = parse_chunk_file(data_path)
        df = pd.read_parquet(data_path)
        required = {"episode_index", "index", "task_index"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{data_path} is missing required columns: {sorted(missing)}")

        for episode_index in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == episode_index]
            task_indices = list(dict.fromkeys(int(v) for v in ep_df["task_index"].tolist()))
            tasks = [task_by_index.get(task_index, str(task_index)) for task_index in task_indices]
            source = EpisodeSource(
                episode_index=int(episode_index),
                length=int(len(ep_df)),
                dataset_from_index=int(ep_df["index"].min()),
                dataset_to_index=int(ep_df["index"].max() + 1),
                data_chunk_index=chunk_index,
                data_file_index=file_index,
                data_path=data_path,
                tasks=tasks,
            )
            if source.episode_index in episodes:
                raise ValueError(f"Episode {source.episode_index} appears in multiple data files")
            episodes[source.episode_index] = source

    if not episodes:
        raise ValueError(f"No episodes found in {dataset_root / 'data'}")

    return [episodes[idx] for idx in sorted(episodes)]


def build_video_segments(dataset_root: Path, info: dict) -> dict[str, list[VideoSegment]]:
    segments_by_key: dict[str, list[VideoSegment]] = {}

    for video_key in video_keys_from_info(info):
        video_dir = dataset_root / "videos" / video_key
        video_files = sorted(video_dir.glob("chunk-*/*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"No video files found for {video_key}: {video_dir}")

        start_frame = 0
        segments: list[VideoSegment] = []
        for video_path in video_files:
            chunk_index, file_index = parse_chunk_file(video_path)
            result = check_video(video_path)
            if result.has_issues:
                raise ValueError(f"Video still has decode issues, repair it first: {video_path}")
            end_frame = start_frame + result.frame_count
            segments.append(
                VideoSegment(
                    video_key=video_key,
                    chunk_index=chunk_index,
                    file_index=file_index,
                    path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
            start_frame = end_frame

        segments_by_key[video_key] = segments

    return segments_by_key


def find_segment_for_episode(segments: list[VideoSegment], episode: EpisodeSource) -> VideoSegment:
    for segment in segments:
        if segment.start_frame <= episode.dataset_from_index and episode.dataset_to_index <= segment.end_frame:
            return segment

    raise ValueError(
        f"Episode {episode.episode_index} frames "
        f"[{episode.dataset_from_index}, {episode.dataset_to_index}) do not fit in one video segment "
        f"for {segments[0].video_key if segments else 'unknown video key'}"
    )


def episode_array(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        return np.stack(series.to_numpy())
    return series.to_numpy()


def compute_video_episode_stats(
    segment: VideoSegment,
    episode: EpisodeSource,
) -> dict[str, np.ndarray]:
    local_start = episode.dataset_from_index - segment.start_frame
    local_end = episode.dataset_to_index - segment.start_frame
    frame_indices = set(local_start + idx for idx in sample_indices(episode.length))

    sampled_frames: list[np.ndarray] = []
    with av.open(str(segment.path), "r") as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index >= local_end:
                break
            if frame_index not in frame_indices:
                continue

            image = frame.to_ndarray(format="rgb24")
            image = np.transpose(image, (2, 0, 1))
            sampled_frames.append(auto_downsample_height_width(image))

    if len(sampled_frames) != len(frame_indices):
        raise ValueError(
            f"Decoded {len(sampled_frames)} sampled frames for {segment.path}, expected {len(frame_indices)}"
        )

    frame_array = np.stack(sampled_frames)
    stats = get_feature_stats(frame_array, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value / 255.0, axis=0) for key, value in stats.items()}


def compute_episode_metadata(
    dataset_root: Path,
    info: dict,
    episode: EpisodeSource,
    video_segments: dict[str, list[VideoSegment]],
    compute_stats: bool,
) -> tuple[dict, dict]:
    row = {
        "episode_index": episode.episode_index,
        "tasks": episode.tasks,
        "length": episode.length,
        "data/chunk_index": episode.data_chunk_index,
        "data/file_index": episode.data_file_index,
        "dataset_from_index": episode.dataset_from_index,
        "dataset_to_index": episode.dataset_to_index,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }

    ep_stats = {}
    for video_key, segments in video_segments.items():
        segment = find_segment_for_episode(segments, episode)
        fps = float(info["fps"])
        local_start = episode.dataset_from_index - segment.start_frame
        local_end = episode.dataset_to_index - segment.start_frame
        row.update(
            {
                f"videos/{video_key}/chunk_index": segment.chunk_index,
                f"videos/{video_key}/file_index": segment.file_index,
                f"videos/{video_key}/from_timestamp": local_start / fps,
                f"videos/{video_key}/to_timestamp": local_end / fps,
            }
        )
        if compute_stats:
            ep_stats[video_key] = compute_video_episode_stats(segment, episode)

    if compute_stats:
        df = pd.read_parquet(episode.data_path)
        ep_df = df[df["episode_index"] == episode.episode_index]
        episode_data = {}
        features = non_video_features(info)
        for key in features:
            if key in ep_df.columns:
                episode_data[key] = episode_array(ep_df[key])
        ep_stats.update(compute_episode_stats(episode_data, features))
        row.update(flatten_dict({"stats": ep_stats}))

    return row, ep_stats


def serialize_parquet_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def backup_path(path: Path) -> Path:
    if not path.exists():
        return path.with_suffix(path.suffix + ".bak")

    candidate = path.with_suffix(path.suffix + ".bak")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        numbered = path.with_suffix(path.suffix + f".bak.{index}")
        if not numbered.exists():
            return numbered
        index += 1


def existing_episode_indices(dataset_root: Path) -> set[int]:
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        return set()

    indices: set[int] = set()
    for parquet_path in sorted(episodes_dir.glob("chunk-*/*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index"])
        indices.update(int(v) for v in df["episode_index"].tolist())
    return indices


def rebuild_episode_metadata(
    dataset_root: Path,
    check_only: bool,
    keep_backup: bool,
    compute_stats_flag: bool,
    force: bool,
) -> None:
    info = load_info(dataset_root)
    episodes = build_episode_sources(dataset_root)
    expected_indices = {episode.episode_index for episode in episodes}
    current_indices = existing_episode_indices(dataset_root)
    missing_indices = sorted(expected_indices - current_indices)

    if current_indices == expected_indices and not force:
        print(f"[OK] meta/episodes contains all {len(expected_indices)} episodes")
        return

    if current_indices == expected_indices and force:
        print(f"[REBUILD] meta/episodes contains all {len(expected_indices)} episodes; forcing recalculation")
    elif current_indices:
        print(
            f"[MISSING] meta/episodes has {len(current_indices)}/{len(expected_indices)} episodes; "
            f"missing={missing_indices}"
        )
    else:
        print(f"[MISSING] meta/episodes is absent or empty; rebuilding {len(expected_indices)} episodes")

    if check_only:
        return

    video_segments = build_video_segments(dataset_root, info)
    rows = []
    stats_list = []
    for episode in episodes:
        print(f"[EPISODE] recalculating episode={episode.episode_index} frames={episode.length}")
        row, ep_stats = compute_episode_metadata(
            dataset_root=dataset_root,
            info=info,
            episode=episode,
            video_segments=video_segments,
            compute_stats=compute_stats_flag,
        )
        rows.append({key: serialize_parquet_value(value) for key, value in row.items()})
        if compute_stats_flag:
            stats_list.append(ep_stats)

    episodes_dir = dataset_root / "meta" / "episodes"
    if keep_backup and episodes_dir.exists():
        dst = backup_path(episodes_dir)
        shutil.move(str(episodes_dir), str(dst))
        print(f"[BACKUP] {episodes_dir.relative_to(dataset_root)} -> {dst.relative_to(dataset_root)}")
    else:
        shutil.rmtree(episodes_dir, ignore_errors=True)

    output_path = episodes_dir / "chunk-000" / "file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)

    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(episode.length for episode in episodes)
    info["splits"] = {"train": f"0:{len(episodes)}"}
    write_info(info, dataset_root)

    if compute_stats_flag:
        write_stats(aggregate_stats(stats_list), dataset_root)

    print(f"[FIXED] rebuilt {output_path.relative_to(dataset_root)}")


def format_result(result: VideoCheckResult, dataset_root: Path) -> str:
    rel_path = result.video_path.relative_to(dataset_root)
    expected = result.expected_frames if result.expected_frames is not None else "unknown"
    if not result.has_issues:
        return f"[OK] {rel_path} frames={result.frame_count}/{expected}"

    parts = [f"[BROKEN] {rel_path} frames={result.frame_count}/{expected} issues={len(result.issues)}"]
    for issue in result.issues:
        parts.append(
            "  "
            f"packet={issue.packet_index} decoded_before_error={issue.decoded_frames_before_error} "
            f"{issue.error_type}: {issue.message}"
        )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and repair LeRobot dataset videos and missing episode metadata."
    )
    parser.add_argument("dataset", help="Dataset path or dataset name under datasets/")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect videos and report issues without modifying files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Replace broken files without keeping .bak backups.",
    )
    parser.add_argument(
        "--skip-episodes",
        action="store_true",
        help="Do not rebuild missing meta/episodes metadata.",
    )
    parser.add_argument(
        "--force-episodes",
        action="store_true",
        help="Rebuild meta/episodes even when all episode indices already exist.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Rebuild meta/episodes without recalculating episode and dataset statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = find_dataset_root(args.dataset)
    info = load_info(dataset_root)
    video_files = iter_video_files(dataset_root)

    print(f"Inspecting dataset: {dataset_root}")
    print(f"Found {len(video_files)} video files")

    broken_results: list[VideoCheckResult] = []
    for video_path in video_files:
        result = check_video(video_path)
        print(format_result(result, dataset_root))
        if result.has_issues:
            broken_results.append(result)

    if not broken_results:
        print("No invalid frames found.")
    elif args.check_only:
        print(f"Found {len(broken_results)} broken videos. No files were modified.")
        return
    else:
        print(f"Repairing {len(broken_results)} broken videos...")
        for result in broken_results:
            rel_path = result.video_path.relative_to(dataset_root)
            parts = rel_path.parts
            video_key = parts[1] if len(parts) >= 4 and parts[0] == "videos" else None
            feature_info = info["features"].get(video_key, {}).get("info", {}) if video_key else {}
            replacements, written_frames = repair_video(
                result.video_path,
                keep_backup=not args.no_backup,
                preferred_codec=feature_info.get("video.codec"),
                preferred_pix_fmt=feature_info.get("video.pix_fmt"),
            )
            repaired = check_video(result.video_path)
            if repaired.has_issues:
                raise RuntimeError(f"Repair failed for {rel_path}: issues remain after rewrite.")
            print(
                f"[FIXED] {rel_path} replacements={replacements} "
                f"frames={written_frames}/{repaired.expected_frames if repaired.expected_frames is not None else 'unknown'}"
            )

    if not args.skip_episodes:
        rebuild_episode_metadata(
            dataset_root=dataset_root,
            check_only=args.check_only,
            keep_backup=not args.no_backup,
            compute_stats_flag=not args.skip_stats,
            force=args.force_episodes,
        )

    print("Repair completed.")


if __name__ == "__main__":
    main()

# 壊れているか確認
# uv run fix_dataset.py iloha-1 --check-only

# 壊れているファイルを修復（--no-backupで.bakバックアップなし）
# uv run fix_dataset.py iloha-1
