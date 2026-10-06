#!/usr/bin/env bash
# Real-Time EXPO-FT online RL learner for iLoHa towel folding.
# The rollout client is iloha_rl.py at the lerobot-KVT root (run on the robot PC):
#   uv run --extra pi iloha_rl.py --host <this machine> --port 8104

set -euo pipefail
# Contain a learner OOM instead of allowing it to kill the desktop/VSCode.
# Fail closed if the user-session memory controller is unavailable.
if [[ ${ILOHA_RL_MEMORY_GUARDED:-0} != 1 ]]; then
    exec systemd-run --user --scope --collect --unit=iloha-rl-learner \
        -p MemoryMax=20G -p MemorySwapMax=1G \
        /usr/bin/env ILOHA_RL_MEMORY_GUARDED=1 \
        /usr/bin/bash "$(realpath "${BASH_SOURCE[0]}")" "$@"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

# The learner listens here; iloha_rl.py dials in directly, no SSH tunnel.
CLIENT_IP=0.0.0.0

# nvidia-smi numbering (mixed GPU models reorder CUDA's default numbering).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
# Release unused allocations between critic/policy stages; avoid BFC pool
# fragmentation when the accumulated policy gradient needs a large buffer.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}

# RTC-SFT checkpoint from scripts/iloha_towel/offline_train.sh.
SFT_CKPT=${SFT_CKPT:-./checkpoints/iloha_towel_rtc_offline/pi_rtc_iloha_towel_high_maxdelay10/checkpoints/10000/params}

# User-selected batch/candidate/UTD counts; keep full demo coverage unchanged.
# Three consecutive real-checkpoint GPU updates and post-update inference pass.
python train_pi_robo.py \
    --config_task=configs/task/iloha_towel.py \
    --dataset_path=./data/iloha_towel/success \
    --num_data=0 \
    --batch_size=16 \
    --utd_ratio=10 \
    --host_update_batches \
    --replay_capacity=0 \
    --replay_prefetch=1 \
    --replay_storage_dir=./replay_cache \
    --noupdate_batch_prefetch \
    --update_type=episode \
    --step_interval=30 \
    --offline_ratio=0 \
    --config=configs/model/realtime_expo_ft_pi_config.py \
    --config.valids_keep_terminal_windows=True \
    --config.N=8 \
    --config.actor_microbatch_size=4 \
    --config.n_edit_samples=8 \
    --config.filter_N=8 \
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
