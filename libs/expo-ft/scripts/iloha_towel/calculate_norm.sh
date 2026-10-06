#!/usr/bin/env bash

source .venv/bin/activate

python scripts/iloha/compute_norm_stats.py \
    --config_name=expo_pi05_iloha_lora_finetune_sft_joint \
    --dataset_path=./data/iloha_towel/success \
    --assets_dir=./assets/expo_pi05_iloha_lora_finetune_sft_joint \
    --asset_id=iloha_towel_high
