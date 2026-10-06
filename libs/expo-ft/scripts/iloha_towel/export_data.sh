#!/usr/bin/env bash
# LeRobot v3.0 iLoHa dataset -> expo-ft traj.hdf5 episodes (Quality: High only).

source .venv/bin/activate

DATASET_ROOT="../../datasets/iloha-dataset-all"
OUT_DIR="./data/iloha_towel/success"

python scripts/iloha/export_lerobot_to_hdf5.py \
    --dataset_root="$DATASET_ROOT" \
    --out_dir="$OUT_DIR" \
    --task_filter="Quality: High" \
    --prompt="Grab the edge of the towel and fold it twice."
