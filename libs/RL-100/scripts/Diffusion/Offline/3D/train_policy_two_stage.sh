#!/usr/bin/env bash
set -euo pipefail

# Two-stage launcher for 3D chunk policy:
# 1) Run one stage-1 pass to produce BC / IQL / dynamics artifacts.
# 2) Fan out BPPO sweep jobs, each in its own timestamped run_dir while sharing
#    the same stage-1 artifacts and a global best under the root run dir.
#
# Example:
#   bash scripts/train_policy_chunk_two_stage.sh rl100 adroit_door_medium 0112 100 4

DEBUG=False
save_ckpt=True

alg_name=${1:?alg_name is required}
task_name=${2:?task_name is required}
addition_info=${3:?addition_info is required}
seed=${4:?seed is required}
NUM_GPUS=${5:-4}

config_name='rl100_3d_epsilon'
exp_name=${task_name}-${alg_name}-${addition_info}

RUN_STAGE1=${RUN_STAGE1:-true}
RUN_SWEEP=${RUN_SWEEP:-true}
STAGE1_ENTRY=${STAGE1_ENTRY:-train_ddp.py}

GPU_LIST=${GPU_LIST:-""}
LR_VALUES=${LR_VALUES:-"1e-6 2e-6 4e-6 1e-5"}
ROLLOUT_VALUES=${ROLLOUT_VALUES:-"10 5 15 3 20"}
CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"0.1 0.8"}

STAGE1_BPPO_STEPS=${STAGE1_BPPO_STEPS:-0}
STAGE1_BPPO_LR=${STAGE1_BPPO_LR:-1e-6}
STAGE1_ROLLOUT_LENGTH=${STAGE1_ROLLOUT_LENGTH:-3}
STAGE1_CLIP_STD_MAX=${STAGE1_CLIP_STD_MAX:-0.1}
STAGE1_EVAL_TIMES=${STAGE1_EVAL_TIMES:-30}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "${GPU_LIST}" ]; then
    gpu_list=${GPU_LIST}
else
    if [ "${NUM_GPUS}" -gt 1 ]; then
        gpu_list=$(bash scripts/find_gpus.sh "${NUM_GPUS}")
        if [ $? -ne 0 ]; then
            echo "Failed to find ${NUM_GPUS} available GPUs" >&2
            exit 1
        fi
    else
        gpu_id=$(bash scripts/find_gpu.sh)
        gpu_list=${gpu_id}
    fi
fi

echo "gpu ids (to use): ${gpu_list}"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
else
    wandb_mode=offline
fi

cd RL-100

export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

act='relu'
encoder_type='dp3vib'
model='skipnet'

root_run_dir="data/outputs_two_stage_chunk/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"
stage1_run_dir="${root_run_dir}"
sweep_root_dir="${root_run_dir}"

mkdir -p "${root_run_dir}"

get_common_params() {
    local run_dir=$1
    local lr=$2
    local rollout_length=$3
    local clip_std_max=$4
    local device=$5

    echo "task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=${DEBUG} \
        training.seed=${seed} \
        training.device=${device} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        unio4.bppo_lr=${lr} \
        unio4.rollout_length=${rollout_length} \
        training.resume=True \
        policy._target_=rl_100.policy.rl100_3d.RL1003D \
        policy.ddim_noise_scheduler.num_train_timesteps=100 \
        policy.cm_noise_scheduler.num_train_timesteps=100 \
        use_action_embed=False \
        horizon=3 \
        n_action_steps=1 \
        n_obs_steps=3 \
        ft_all_actions=False \
        num_inference_steps=10 \
        policy.model=${model} \
        policy.encoder_type=${encoder_type} \
        policy.act=${act} \
        unio4.bppo_steps=5000 \
        policy.encoder_output_dim=64 \
        policy.diffusion_step_embed_dim=256 \
        policy.down_dims="[256,512,1024]" \
        task.env_runner.eval_episodes=30 \
        task.env_runner.env_num=1 \
        offline=True \
        policy.scheduler_type='ddim' \
        policy.use_agent_pos=True \
        only_bc=True \
        unio4.idql_eval=True \
        policy.mlp_policy_depth=3 \
        use_wandb=True \
        critic.omega=0.7 \
        critic.gamma=0.997 \
        policy.use_vib=True \
        policy.use_recon=True \
        dynamics_type='mlp' \
        training.num_epochs=600 \
        training.num_critic_epochs=500 \
        dynamics.dynamics_max_epochs=150 \
        dataloader.batch_size=1024 \
        val_dataloader.batch_size=1024 \
        optimizer.lr=2.83e-4 \
        critic.q_lr=2.83e-4 \
        critic.v_lr=2.83e-4 \
        dynamics.dynamics_lr=5.66e-4 \
        distill_phase='after_offlin' \
        clip_std_max=${clip_std_max} \
        kl_annealing=False \
        policy.beta_kl=1e-5"
}

stage1_complete() {
    local run_dir=$1
    if [ ! -f "${run_dir}/checkpoints/latest.ckpt" ]; then
        return 1
    fi
    if [ ! -f "${run_dir}/Q_bc_20.pt" ]; then
        return 1
    fi
    if [ ! -f "${run_dir}/value_20.pt" ]; then
        return 1
    fi
    if ! compgen -G "${run_dir}/saved_models*/dynamics.pth" > /dev/null; then
        return 1
    fi
    return 0
}

run_stage1() {
    local params
    params=$(get_common_params "${stage1_run_dir}" "${STAGE1_BPPO_LR}" "${STAGE1_ROLLOUT_LENGTH}" "${STAGE1_CLIP_STD_MAX}" "cuda:0")
    local primary_gpu
    local stage1_gpu_count

    primary_gpu="$(echo "${gpu_list}" | cut -d',' -f1)"
    IFS=',' read -ra STAGE1_GPU_ARRAY <<< "${gpu_list}"
    stage1_gpu_count=${#STAGE1_GPU_ARRAY[@]}

    echo "=== Stage 1: materialize BC / IQL / dynamics artifacts ==="
    echo "stage1_run_dir=${stage1_run_dir}"
    echo "stage1_entry=${STAGE1_ENTRY}"
    if [ "${STAGE1_ENTRY}" = "train_ddp.py" ]; then
        echo "stage1_gpu_count=${stage1_gpu_count} (${gpu_list})"
        export CUDA_VISIBLE_DEVICES="${gpu_list}"
        torchrun --standalone --nproc_per_node="${stage1_gpu_count}" "${STAGE1_ENTRY}" \
            --config-name="${config_name}.yaml" \
            ${params} \
            unio4.bppo_steps="${STAGE1_BPPO_STEPS}"
    else
        export CUDA_VISIBLE_DEVICES="${primary_gpu}"
        python "${STAGE1_ENTRY}" --config-name="${config_name}.yaml" \
            ${params} \
            unio4.bppo_steps="${STAGE1_BPPO_STEPS}"
    fi
}

run_sweep_job() {
    local gpu=$1
    local lr=$2
    local rollout_length=$3
    local clip_std_max=$4
    local timestamp
    local sweep_run_dir
    local params

    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
    sweep_run_dir="${sweep_root_dir}/${timestamp}-lr_${lr}_rollout_${rollout_length}_clip_${clip_std_max}"

    mkdir -p "${sweep_run_dir}"
    params=$(get_common_params "${sweep_run_dir}" "${lr}" "${rollout_length}" "${clip_std_max}" "cuda:0")

    echo "[GPU ${gpu}] Starting sweep job: lr=${lr}, rollout=${rollout_length}, clip_std_max=${clip_std_max}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export MUJOCO_EGL_DEVICE_ID=${gpu}
        exec python train_ddp.py --config-name="${config_name}.yaml" \
            ${params} \
            +unio4.stage1_resume_dir="${stage1_run_dir}" \
            +unio4.global_best_dir="${root_run_dir}/best"
    )
}

if [ "${RUN_STAGE1}" = true ] || [ "${RUN_STAGE1}" = "True" ]; then
    if stage1_complete "${stage1_run_dir}"; then
        echo "=== Stage 1 artifacts already exist, skipping stage 1 ==="
        echo "stage1_run_dir=${stage1_run_dir}"
    else
        run_stage1
    fi
fi

if ! stage1_complete "${stage1_run_dir}"; then
    echo "Stage 1 artifacts are incomplete under ${stage1_run_dir}" >&2
    exit 1
fi

if [ "${RUN_SWEEP}" = true ] || [ "${RUN_SWEEP}" = "True" ]; then
    echo "=== Stage 2: BPPO sweep ==="
    IFS=',' read -ra GPU_ARRAY <<< "${gpu_list}"
    num_available_gpus=${#GPU_ARRAY[@]}

    declare -a param_combinations=()
    for lr in ${LR_VALUES}; do
        for rollout_length in ${ROLLOUT_VALUES}; do
            for clip_std_max in ${CLIP_STD_MAX_VALUES}; do
                param_combinations+=("${lr}:${rollout_length}:${clip_std_max}")
            done
        done
    done

    echo "total parameter combinations: ${#param_combinations[@]}"
    echo "available GPUs: ${num_available_gpus} (${gpu_list})"

    declare -a batch_pids=()
    batch_slot=0
    sweep_failed=0

    wait_for_batch() {
        local pid
        local status
        for pid in "${batch_pids[@]}"; do
            set +e
            wait "${pid}"
            status=$?
            set -e
            if [ "${status}" -ne 0 ]; then
                echo "Sweep job pid=${pid} failed with exit code ${status}" >&2
                sweep_failed=1
            fi
        done
        batch_pids=()
    }

    for combo in "${param_combinations[@]}"; do
        IFS=':' read -r lr rollout_length clip_std_max <<< "${combo}"
        gpu=${GPU_ARRAY[$batch_slot]}

        run_sweep_job "${gpu}" "${lr}" "${rollout_length}" "${clip_std_max}" &
        batch_pids+=($!)
        batch_slot=$((batch_slot + 1))

        if [ "${batch_slot}" -eq "${num_available_gpus}" ]; then
            wait_for_batch
            batch_slot=0
        fi
    done

    if [ "${#batch_pids[@]}" -gt 0 ]; then
        wait_for_batch
    fi

    if [ "${sweep_failed}" -ne 0 ]; then
        echo "One or more BPPO sweep jobs failed" >&2
        exit 1
    fi

    echo "=== 3D two-stage sweep completed ==="
fi
