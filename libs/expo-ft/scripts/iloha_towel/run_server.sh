#!/usr/bin/env bash
# Real-Time EXPO-FT online RL learner for iLoHa towel folding.
# The rollout client is iloha_rl.py at the lerobot-KVT root (run on the robot PC):
#   uv run --extra pi iloha_rl.py --host <this machine> --port 8104

source .venv/bin/activate

# The learner listens here; iloha_rl.py dials in directly, no SSH tunnel.
CLIENT_IP=0.0.0.0

# nvidia-smi numbering (mixed GPU models reorder CUDA's default numbering).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

# RTC-SFT checkpoint from scripts/iloha_towel/offline_train.sh.
SFT_CKPT=${SFT_CKPT:-./checkpoints/iloha_towel_rtc_offline/pi_rtc_iloha_towel_high_maxdelay10/checkpoints/10000/params}

python train_pi_robo.py \
    --config_task=configs/task/iloha_towel.py \
    --dataset_path=./data/iloha_towel/success \
    --num_data=0 \
    --update_type=episode \
    --step_interval=30 \
    --offline_ratio=0 \
    --config=configs/model/realtime_expo_ft_pi_config.py \
    --config.valids_keep_terminal_windows=True \
    --config.N=32 \
    --config.n_edit_samples=32 \
    --config.filter_N=32 \
    --config.filter_n_edit=1 \
    --config.edit_scale=0.1 \
    --config.filter_add_delayed_obs=True \
    --config.pi05_config_name=expo_pi05_iloha_lora_finetune_sft_joint \
    --config.pi05_weight_loader_path="$SFT_CKPT" \
    --config.pi05_assets_dir="./assets/expo_pi05_iloha_lora_finetune_sft_joint" \
    --config.pi05_asset_id="iloha_towel_high" \
    --project_name=expo_ft_iloha_towel \
    --output_dir=./checkpoints/iloha_towel \
    --client_host="$CLIENT_IP" \
    --client_port=8104 \
    --fsdp_devices=1 \
    --delay=5 \
    --resume \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=20000 \
    --run_name=ours_iloha_towel_high_delay5
