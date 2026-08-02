#!/usr/bin/env bash
set -euo pipefail
#
# iloha-dataset-sushi の 模倣学習(BC) + オフラインRL を RL-100 で回すラッパ。
#
# 前提:
#   1) RL-100 の学習環境(conda: rl100 など, INSTALL.md 参照)を有効化していること。
#      本スクリプトは lerobot の uv venv ではなく RL-100 環境で実行する。
#   2) 先に zarr 変換を済ませていること:
#        uv run python iloha_to_rl100_zarr.py \
#            --dataset datasets/iloha-dataset-sushi \
#            --out libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr
#      → libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr が出来ている。
#   3) このスクリプトは RL-100 の git ルート(libs/RL-100)から実行する。
#        cd libs/RL-100 && bash scripts/iloha/train_sushi_offline.sh
#
# 2段構成(既存の chunk two-stage ランチャを流用):
#   Stage 1 = BC 初期化 + IQL critic + dynamics(world model) の学習
#             (unio4.bppo_steps=0。BC=模倣学習に相当)
#   Stage 2 = BPPO による オフラインRL fine-tune(1ジョブだけの最小スイープ)
#
# 実機/シムのオンライン評価はここでは行わない(task=sushi は env_runner.fake_env=True)。
# 実評価・オンラインRLは iloha_online_rl.py で別途行う。
#
# 使い方:
#   bash scripts/iloha/train_sushi_offline.sh [tag] [seed] [num_gpus]
# 例:
#   bash scripts/iloha/train_sushi_offline.sh run0 100 1

TAG=${1:-run0}
SEED=${2:-100}
NUM_GPUS=${3:-1}

# --- 最小構成(スイープを1ジョブに絞る)。本格スイープは既存ランチャを直接叩く。---
export LR_VALUES=${LR_VALUES:-"1.42e-6"}
export ROLLOUT_VALUES=${ROLLOUT_VALUES:-"5"}
export CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"null"}
export CHUNK_LOSS_MODE_COMBOS=${CHUNK_LOSS_MODE_COMBOS:-"scalar:scalar_iql"}

# --- action-chunk ジオメトリ(ALOHA系操作の標準)---
export N_OBS_STEPS=${N_OBS_STEPS:-3}
export N_ACTION_STEPS=${N_ACTION_STEPS:-16}

# 初回スモークテストでエポックを絞りたい場合は以下を指定して呼ぶ:
#   TRAIN_NUM_EPOCHS=20 CRITIC_NUM_EPOCHS=20 DYN_MAX_EPOCHS=20 bash scripts/iloha/train_sushi_offline.sh ...
if [ -n "${TRAIN_NUM_EPOCHS:-}" ]; then
  export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-} training.num_epochs=${TRAIN_NUM_EPOCHS}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # libs/RL-100/scripts
LAUNCHER="${HERE}/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh"

echo "==== RL-100 sushi offline (BC + Offline RL) ===="
echo "tag=${TAG} seed=${SEED} num_gpus=${NUM_GPUS}"
echo "launcher=${LAUNCHER}"
echo "N_OBS_STEPS=${N_OBS_STEPS} N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "sweep: LR=${LR_VALUES} ROLLOUT=${ROLLOUT_VALUES} COMBO=${CHUNK_LOSS_MODE_COMBOS}"

bash "${LAUNCHER}" rl100 sushi "${TAG}" "${SEED}" "${NUM_GPUS}"
