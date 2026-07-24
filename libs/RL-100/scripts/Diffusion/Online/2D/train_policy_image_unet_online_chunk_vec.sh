#!/usr/bin/env bash

# Examples:
# bash scripts/train_policy_image_unet_online_vec.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy_image_unet_online_vec.sh rl100 adroit_door_medium 0112 100 8

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_2d_epsilon'
addition_info=${3}
seed=${4}
train_env_num=${5:-4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs_3d/${exp_name}_seed${seed}"

gpu_id=$(bash scripts/find_gpu.sh)
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
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
export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export EGL_DEVICE_ID=${gpu_id}

act='mish'
model='dp3'
encoder_type='resnet'
run_dir="data/outputs_2d_chunk/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"

N_OBS_STEPS=${N_OBS_STEPS:-3}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
HORIZON=${HORIZON:-$((N_ACTION_STEPS + N_OBS_STEPS - 1))}
CRITIC_STRIDE=${CRITIC_STRIDE:-${N_ACTION_STEPS}}
FINETUNE_STRIDE=${FINETUNE_STRIDE:-${N_ACTION_STEPS}}
USE_ACTION_EMBED=${USE_ACTION_EMBED:-False}
USE_CONV_ACTION_EMBED=${USE_CONV_ACTION_EMBED:-False}
Q_HIDDEN_DIM=${Q_HIDDEN_DIM:-1024}
V_HIDDEN_DIM=${V_HIDDEN_DIM:-512}
CRITIC_Q_LAYER_NORM=${CRITIC_Q_LAYER_NORM:-True}
CRITIC_ACTION_EMBED_LAYER_NORM=${CRITIC_ACTION_EMBED_LAYER_NORM:-False}
CRITIC_ACTION_SCALE_NORM=${CRITIC_ACTION_SCALE_NORM:-False}
DYNAMICS_ACTION_EMBED_LAYER_NORM=${DYNAMICS_ACTION_EMBED_LAYER_NORM:-False}
DYNAMICS_ACTION_SCALE_NORM=${DYNAMICS_ACTION_SCALE_NORM:-False}
DYNAMICS_HIDDEN_DIMS=${DYNAMICS_HIDDEN_DIMS:-"[1024,1024,512,512]"}
DYNAMICS_WEIGHT_DECAY=${DYNAMICS_WEIGHT_DECAY:-"[2.5e-5,5.0e-5,7.5e-5,7.5e-5,1.0e-4]"}
critic_artifact_dir="${run_dir}/critic_c${CRITIC_STRIDE}_f${FINETUNE_STRIDE}"

echo -e "\033[33mcritic_artifact_dir: ${critic_artifact_dir}\033[0m"

for lr_a in 2e-6; do
    for K_epochs in 5; do
        for clip in 0.1; do
            python train.py --config-name=${config_name}.yaml \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt} \
                unio4.bppo_lr=3e-6 \
                unio4.rollout_length=30 \
                training.resume=True \
                policy._target_=rl_100.policy.rl100_2d.RL1002D \
                policy.ddim_noise_scheduler.num_train_timesteps=100 \
                use_action_embed=${USE_ACTION_EMBED} \
                use_conv_action_embed=${USE_CONV_ACTION_EMBED} \
                horizon=${HORIZON} \
                n_action_steps=${N_ACTION_STEPS} \
                n_obs_steps=${N_OBS_STEPS} \
                ft_all_actions=False \
                num_inference_steps=10 \
                unio4.bppo_steps=5000 \
                +unio4.critic_artifact_dir="${critic_artifact_dir}" \
                online=True \
                policy.encoder_output_dim=64 \
                policy.down_dims="[256,512,1024]" \
                policy.scheduler_type='ddim' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=1024 \
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
                policy.img_shape=[3,224,224] \
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
                ppo.idql_batch_size=64 \
                load_bc=False \
                clip_std_max=${clip} \
                ppo.load_online_cp=False \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                use_vib=True \
                use_recon=True \
                dynamics_type='diffusion' \
                dynamics.dynamics_hidden_dims=${DYNAMICS_HIDDEN_DIMS} \
                dynamics.dynamics_weight_decay=${DYNAMICS_WEIGHT_DECAY} \
                dynamics.action_embed_layer_norm=${DYNAMICS_ACTION_EMBED_LAYER_NORM} \
                dynamics.action_scale_norm=${DYNAMICS_ACTION_SCALE_NORM} \
                critic.q_hidden_dim=${Q_HIDDEN_DIM} \
                critic.v_hidden_dim=${V_HIDDEN_DIM} \
                critic.q_layer_norm=${CRITIC_Q_LAYER_NORM} \
                critic.action_embed_layer_norm=${CRITIC_ACTION_EMBED_LAYER_NORM} \
                critic.action_scale_norm=${CRITIC_ACTION_SCALE_NORM} \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with='q' \
                policy.joint_opt_encoder=False \
                predict_r=True \
                chunk_as_single_action=True \
                dynamics.prediction_mode='full' \
                gamma=0.997 \
                ppo.per_step_recon=True \
                ppo.recon=True \
                ppo.value_recon=True \
                ppo.force_stochastic_online=True \
                encoders.resnet.kl_beta=0.005 \
                encoders.resnet.recon_loss_weight=0.1
        done
    done
done
