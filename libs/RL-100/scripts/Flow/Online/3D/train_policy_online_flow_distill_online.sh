# Mode 2: Online interleaved distillation (10-step teacher rollout + PPO + 1-step student distill simultaneously)
# Pipeline: 10-step flow teacher does rollout + PPO update, 1-step student trained via distill_update()
# Student eval via use_cm=True, teacher eval via default path
# Example:
# bash scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_3d_flow'
addition_info=${3}
seed=${4}
train_env_num=${5:-16}
use_policy_iql_encoder=${6:-True}
exp_name=${task_name}-${alg_name}-flow-${addition_info}
if [ "${use_policy_iql_encoder}" = "True" ]; then
    online_iql_recon=False
    fix_iql_encoder=True
else
    online_iql_recon=True
    fix_iql_encoder=False
fi
run_dir="data/outputs/${exp_name}_seed${seed}"


gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
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
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export EGL_DEVICE_ID=${gpu_id}
act='relu'
encoder_type='dp3vib'
model='skipnet'
run_dir="data/outputs_flow_two_stage/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"


for lr_a in 5.8e-6; do
    for K_epochs in 5; do
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
                policy._target_=rl_100.policy.rl100_3d.RL1003D \
                policy.scheduler_type='flow' \
                use_action_embed=False \
                horizon=3 \
                n_action_steps=1 \
                n_obs_steps=3 \
                ft_all_actions=False \
                num_inference_steps=10 \
                flow_inference_steps=10 \
                flow_distill_inference_steps=1 \
                flow_distill_teacher_steps=10 \
                flow_sde_type='cps' \
                flow_cps_logprob_mode='gaussian' \
                flow_noise_level=0.7 \
                flow_sde_window_size=0 \
                flow_logit_normal_sampling=False \
                flow_noise_on_final_step=False \
                policy.model=$model \
                policy.encoder_type=$encoder_type \
                policy.act=$act \
                unio4.bppo_steps=5000 \
                online=True \
                policy.encoder_output_dim=64 \
                policy.diffusion_step_embed_dim=256 \
                policy.down_dims="[256,512,1024]" \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=512 \
                ppo.batch_size=2048 \
                task.env_runner.env_num=4 \
                task.env_runner.eval_episodes=30 \
                ++ppo.use_vec_env_online=True \
                ++ppo.train_env_num=${train_env_num} \
                ++ppo.eval_env_num=${train_env_num} \
                ppo.share_encoder=False \
                unio4.eval_times=1 \
                ppo.fix_encoder=False \
                ppo.max_train_steps=1000000 \
                policy.img_shape=[3,84,84] \
                policy.use_agent_pos=True \
                distill_phase='online' \
                distill_loss_type='action_same_noise' \
                distill2mean=True \
                update_phase='step' \
                policy.mlp_policy_depth=3 \
                ppo.save_online_cp=False \
                ppo.online_cp_save_freq=50 \
                training.num_epochs=200 \
                load_bc=False \
                clip_std_max=0.2 \
                ppo.load_online_cp=False \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                policy.use_vib=True \
                policy.use_recon=True \
                dynamics_type='diffusion' \
                ppo.recon=True \
                ppo.value_recon=True \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with='q' \
                policy.joint_opt_encoder=False \
                ppo.per_step_recon=False \
                ppo.force_stochastic_online=True \
                policy.beta_kl=0.001
    done
done
