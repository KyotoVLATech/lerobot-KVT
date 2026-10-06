"""Export an iLoHa LeRobot v3.0 dataset to expo-ft's per-episode HDF5 layout.

The offline / online learners read demos with `expo_ft.env.droid_utils.process_droid_dataset`,
which expects `<out_dir>/<N>/traj.hdf5` with `saved_observation/` and `action/` groups. This
script writes that layout from a LeRobot v3.0 dataset (parquet + AV1 mp4), without needing a
LeRobot install (OpenPI pins an older lerobot that cannot read v3.0).

Per step it stores:
  saved_observation/cam_high_image, cam_left_wrist_image, cam_right_wrist_image
      uint8 [3, 224, 224]; resized with the same `resize_with_pad` OpenPI applies at train and
      inference time, so pre-resizing here is lossless w.r.t. what the model sees.
  saved_observation/state    float32 [14]  (ALOHA joint coordinates)
  saved_observation/prompt   language instruction
  action/joint_position      float32 [14]  (absolute joint targets)

Usage:
    python scripts/iloha/export_lerobot_to_hdf5.py \
        --dataset_root ../../datasets/iloha-dataset-all \
        --out_dir ./data/iloha_towel/success \
        --task_filter "Quality: High" \
        --prompt "Grab the edge of the towel and fold it twice."
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil

import av
import h5py
import numpy as np
import pandas as pd
import tqdm
from openpi_client.image_tools import resize_with_pad

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_SIZE = 224


def _decode_camera(args):
    """Decode one camera video and write the needed frames (resized, CHW) into a memmap."""
    video_path, frame_to_row, num_rows, fps, out_path, cam = args
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                    shape=(num_rows, 3, IMAGE_SIZE, IMAGE_SIZE))
    written = np.zeros(num_rows, dtype=bool)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in tqdm.tqdm(container.decode(stream), total=stream.frames, desc=cam, position=CAMERAS.index(cam)):
            idx = int(round(float(frame.pts * stream.time_base) * fps))
            row = frame_to_row.get(idx)
            if row is None:
                continue
            img = resize_with_pad(frame.to_ndarray(format="rgb24"), IMAGE_SIZE, IMAGE_SIZE)
            out[row] = img.transpose(2, 0, 1)
            written[row] = True
    out.flush()
    return cam, int(written.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_filter", default="",
                        help="Keep episodes whose task string contains this substring (empty = all).")
    parser.add_argument("--prompt", required=True, help="Language instruction stored with every step.")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all matching episodes.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    fps = info["fps"]
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 dataset, got {info.get('codebase_version')}")

    episodes = pd.concat(
        [pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(root, "meta", "episodes"))
         for f in sorted(fs) if f.endswith(".parquet")]
    ).sort_values("episode_index").reset_index(drop=True)
    keep = episodes["tasks"].apply(lambda ts: any(args.task_filter in t for t in ts))
    episodes = episodes[keep].reset_index(drop=True)
    if args.max_episodes > 0:
        episodes = episodes.iloc[:args.max_episodes]
    print(f"Exporting {len(episodes)} episodes ({int(episodes['length'].sum())} frames) "
          f"matching task_filter={args.task_filter!r}")

    for cam in CAMERAS:
        files = episodes[[f"videos/observation.images.{cam}/chunk_index",
                          f"videos/observation.images.{cam}/file_index"]].drop_duplicates()
        if len(files) != 1:
            raise NotImplementedError(f"{cam}: episodes span multiple video files; not supported yet.")

    data = pd.concat(
        [pd.read_parquet(os.path.join(dp, f), columns=["observation.state", "action", "episode_index", "frame_index"])
         for dp, _, fs in os.walk(os.path.join(root, "data")) for f in sorted(fs) if f.endswith(".parquet")]
    )
    data = data[data["episode_index"].isin(episodes["episode_index"])]
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    assert len(data) == int(episodes["length"].sum()), "data rows do not match episode lengths"

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} is not empty; pass --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    tmp_dir = os.path.join(args.out_dir, "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    # Global row index (into `data`) of each episode's first frame.
    row_starts = np.concatenate([[0], np.cumsum(episodes["length"].values)[:-1]])

    jobs = []
    for cam in CAMERAS:
        frame_to_row = {}
        for (_, ep), row_start in zip(episodes.iterrows(), row_starts):
            first = int(round(ep[f"videos/observation.images.{cam}/from_timestamp"] * fps))
            for i in range(int(ep["length"])):
                frame_to_row[first + i] = int(row_start + i)
        ep0 = episodes.iloc[0]
        video_path = os.path.join(root, info["video_path"].format(
            video_key=f"observation.images.{cam}",
            chunk_index=int(ep0[f"videos/observation.images.{cam}/chunk_index"]),
            file_index=int(ep0[f"videos/observation.images.{cam}/file_index"]),
        ))
        jobs.append((video_path, frame_to_row, len(data), fps, os.path.join(tmp_dir, f"{cam}.npy"), cam))

    with mp.Pool(len(CAMERAS)) as pool:
        for cam, n in pool.imap_unordered(_decode_camera, jobs):
            if n != len(data):
                raise RuntimeError(f"{cam}: decoded {n} of {len(data)} frames")
    print()

    frames = {cam: np.load(os.path.join(tmp_dir, f"{cam}.npy"), mmap_mode="r") for cam in CAMERAS}
    states = np.stack(data["observation.state"].values).astype(np.float32)
    actions = np.stack(data["action"].values).astype(np.float32)

    for out_idx, ((_, ep), row_start) in enumerate(tqdm.tqdm(list(zip(episodes.iterrows(), row_starts)),
                                                             desc="write hdf5")):
        sl = slice(int(row_start), int(row_start + ep["length"]))
        T = sl.stop - sl.start
        ep_dir = os.path.join(args.out_dir, str(out_idx))
        os.makedirs(ep_dir, exist_ok=True)
        with h5py.File(os.path.join(ep_dir, "traj.hdf5"), "w") as f:
            f.attrs["source_episode_index"] = int(ep["episode_index"])
            f.attrs["source_task"] = str(ep["tasks"][0])
            obs = f.create_group("saved_observation")
            for cam in CAMERAS:
                obs.create_dataset(f"{cam}_image", data=np.asarray(frames[cam][sl]),
                                   chunks=(1, 3, IMAGE_SIZE, IMAGE_SIZE), compression="lzf")
            obs.create_dataset("state", data=states[sl])
            obs.create_dataset("prompt", data=np.array([args.prompt.encode()] * T))
            f.create_group("action").create_dataset("joint_position", data=actions[sl])

    # Close the memmaps first: on NFS, open files turn into .nfs* placeholders that block rmtree.
    del frames
    shutil.rmtree(tmp_dir)
    print(f"Wrote {len(episodes)} episodes to {args.out_dir}")


if __name__ == "__main__":
    main()
