#!/usr/bin/env bash
set -euo pipefail

# Examples:
# bash scripts/train_policy_image_unet_online_ddp_distill.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy_image_unet_online_ddp_distill.sh rl100 adroit_door_medium 0112 100 8
# DISTILL_GPU_COUNT=4 bash scripts/train_policy_image_unet_online_ddp_distill.sh rl100 adroit_door_medium 0112 100 8
# DISTILL_GPUS=0,1,2,3 ONLINE_GPU=0 bash scripts/train_policy_image_unet_online_ddp_distill.sh rl100 adroit_door_medium 0112 100 8

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_2d_epsilon'
addition_info=${3}
seed=${4}
ft_seed=${seed}
train_env_num=${5:-4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs_3d/${exp_name}_seed${seed}"

DISTILL_GPUS=${DISTILL_GPUS:-""}
DISTILL_GPU_COUNT=${DISTILL_GPU_COUNT:-4}
ONLINE_GPU=${ONLINE_GPU:-""}

if [ -z "$DISTILL_GPUS" ]; then
    DISTILL_GPUS=$(bash scripts/find_gpus.sh "${DISTILL_GPU_COUNT}")
fi

if [ -z "$ONLINE_GPU" ]; then
    ONLINE_GPU=$(echo "$DISTILL_GPUS" | cut -d',' -f1)
fi

export DISTILL_GPUS

distill_gpu_count=$(python - <<'PY'
import os
s = os.environ['DISTILL_GPUS']
print(len([x for x in s.split(',') if x.strip()]))
PY
)

echo -e "\033[33mdistill gpus: ${DISTILL_GPUS} (count=${distill_gpu_count})\033[0m"
echo -e "\033[33monline gpu: ${ONLINE_GPU}\033[0m"
echo -e "\033[33mtrain_env_num: ${train_env_num}\033[0m"

if [ "$DEBUG" = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd RL-100

export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

act=${act:-'mish'}
model=${model:-'skipnet'}
encoder_type=${encoder_type:-'resnet'}

K_epochs=${K_epochs:-5}
lr_a=${lr_a:-4.6e-6}
clip=${clip:-0.8}

stage1_run_dir="data/outputs_2d/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"
run_dir="${stage1_run_dir}"
offline_distilled_path="${stage1_run_dir}/best/last/distilled_model.pt"

echo -e "\033[33mstage1_run_dir: ${stage1_run_dir}\033[0m"
echo -e "\033[33moffline_distilled_path: ${offline_distilled_path}\033[0m"

COMMON_ARGS=( \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=$DEBUG \
                training.seed=${ft_seed} \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt} \
                unio4.bppo_lr=3e-6 \
                unio4.rollout_length=30 \
                training.resume=True \
                policy._target_=rl_100.policy.rl100_2d.RL1002D \
                policy.ddim_noise_scheduler.num_train_timesteps=100 \
                use_action_embed=False \
                horizon=3 \
                n_action_steps=1 \
                n_obs_steps=3 \
                ft_all_actions=False \
                num_inference_steps=10 \
                unio4.bppo_steps=5000 \
                online=True \
                policy.encoder_output_dim=64 \
                policy.down_dims="[256,512,1024]" \
                policy.scheduler_type='ddim' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=2048 \
                task.env_runner.env_num=4 \
                task.env_runner.eval_episodes=30 \
                ++task.env_runner.with_pointcloud=False \
                ++ppo.use_vec_env_online=True \
                ++ppo.train_env_num=${train_env_num} \
                ++ppo.eval_env_num=$((${train_env_num} / 2)) \
                use_agent_pos=True \
                policy.use_visual=True \
                feature_type='2D' \
                ppo.fix_encoder=False \
                ppo.share_encoder=False \
                policy.model=$model \
                policy.act=$act \
                policy.mlp_policy_depth=3 \
                policy.use_aug=False \
                policy.vec_env_rng_align=True \
                ppo.max_train_steps=1000000 \
                encoder_type='resnet' \
                encoders.resnet.share_rgb_model=False \
                encoders.resnet.rgb_model.weights='r3m' \
                task.dataset.pre_image_norm=True \
                task.critic_dataset.pre_image_norm=True \
                task.scale_dataset.pre_image_norm=True \
                distill_phase='online' \
                distill_loss_type='action_same_noise' \
                distill2mean=True \
                update_phase='step' \
                ppo.save_online_cp=False \
                ppo.online_cp_save_freq=50 \
                training.num_epochs=200 \
                load_bc=False \
                clip_std_max=${clip} \
                ppo.load_online_cp=False \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                use_vib=True \
                use_recon=True \
                dynamics_type='diffusion' \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with='q' \
                policy.joint_opt_encoder=False \
                ppo.per_step_recon=False \
                ppo.recon=False \
                ppo.value_recon=False \
                ppo.force_stochastic_online=True \
                encoders.resnet.kl_beta=0.05 \
                encoders.resnet.recon_loss_weight=0.1 \
)

if [ ! -f "${offline_distilled_path}" ]; then
    echo -e "\033[33moffline distilled model not found, running multi-GPU distill first\033[0m"
    export CUDA_VISIBLE_DEVICES=${DISTILL_GPUS}
    unset CUDA_LAUNCH_BLOCKING
    export MUJOCO_EGL_DEVICE_ID=0
    export EGL_DEVICE_ID=0
    DISTILL_CMD=(
        torchrun
        --standalone
        --nproc_per_node=${distill_gpu_count}
        train_ddp.py
        --config-name=${config_name}.yaml
        training.device=cuda:0
    )
    DISTILL_CMD+=("${COMMON_ARGS[@]}")
    "${DISTILL_CMD[@]}"
else
    echo -e "\033[32mfound offline distilled model, skip multi-GPU distill\033[0m"
fi

echo -e "\033[33mstart single-GPU online training on GPU ${ONLINE_GPU}\033[0m"
export CUDA_VISIBLE_DEVICES=${ONLINE_GPU}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
ONLINE_CMD=(
    python
    train.py
    --config-name=${config_name}.yaml
    training.device=cuda:0
)
ONLINE_CMD+=("${COMMON_ARGS[@]}")
"${ONLINE_CMD[@]}"
