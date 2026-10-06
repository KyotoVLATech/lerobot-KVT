#!/usr/bin/env bash
# RTC-SFT (prefix-conditioned pi0.5 LoRA BC) on the iLoHa towel demos.

source .venv/bin/activate

# nvidia-smi numbering (mixed GPU models reorder CUDA's default numbering).
# Two H200s run data-parallel (~3 s/step on one at batch 64).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}

python train_offline_rtc.py \
    --config_task=configs/task/iloha_towel.py \
    --dataset_path=./data/iloha_towel/success \
    --num_data=0 \
    --batch_size=64 \
    --fsdp_devices=1 \
    --config=configs/model/rtc_pi_config.py \
    --config.freeze_pi05_encoder=False \
    --config.p1_use_prefix_conditioning=True \
    --config.p1_max_delay=10 \
    --config.pi05_config_name=expo_pi05_iloha_lora_finetune_sft_joint \
    --config.pi05_assets_dir="./assets/expo_pi05_iloha_lora_finetune_sft_joint" \
    --config.pi05_asset_id="iloha_towel_high" \
    --project_name=pi_sft_iloha_towel \
    --output_dir=./checkpoints/iloha_towel_rtc_offline \
    --max_steps=${MAX_STEPS:-10000} \
    --resume \
    --checkpoint_model \
    --checkpoint_interval=2000 \
    --run_name=${RUN_NAME:-pi_rtc_iloha_towel_high_maxdelay10}
