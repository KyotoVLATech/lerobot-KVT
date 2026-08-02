#!/usr/bin/env bash
set -euo pipefail
# stage-1(BC+critic+dynamics) を極小エポックで単一プロセス実行して配管を検証するスモークテスト。
# RL-100 の git ルート(libs/RL-100)から、RL-100 venv で実行する。
#   cd libs/RL-100 && PYTHONPATH=$(pwd)/RL-100 ../.venv-rl100/bin/... は run_sushi 側で設定
ZARR=${ZARR:-data/iloha_sushi_smoke.zarr}
RUN_DIR=${RUN_DIR:-data/outputs_smoke/sushi_stage1}
CRITIC_DIR=${CRITIC_DIR:-${RUN_DIR}/critic}
PY=${PY:-python}

cd "$(dirname "${BASH_SOURCE[0]}")/../../RL-100"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

$PY train_ddp.py --config-name=rl100_2d_epsilon.yaml \
  task=sushi \
  hydra.job.chdir=False \
  hydra.run.dir="${RUN_DIR}" \
  training.debug=False training.seed=100 training.device=cuda:0 \
  exp_name=sushi-smoke logging.mode=offline checkpoint.save_ckpt=True \
  policy._target_=rl_100.policy.rl100_2d.RL1002D \
  policy.ddim_noise_scheduler.num_train_timesteps=100 \
  training.resume=True \
  horizon=18 n_action_steps=16 n_obs_steps=3 num_inference_steps=10 \
  offline=True use_agent_pos=True policy.use_visual=True \
  policy.model=dp3 policy.act=mish policy.mlp_policy_depth=3 \
  feature_type='2D' policy.scheduler_type='ddim' \
  encoder_output_dim=64 policy.down_dims=[256,512,1024] \
  ++task.env_runner.with_pointcloud=False \
  policy.use_aug=True critic.omega=0.9 critic.gamma=0.997 \
  policy.img_shape=[3,224,224] \
  task.dataset.pre_image_norm=True \
  ++task.critic_dataset.pre_image_norm=True \
  ++task.finetune_dataset.pre_image_norm=True \
  task.dataset.zarr_path="${ZARR}" \
  task.critic_dataset.zarr_path="${ZARR}" \
  task.finetune_dataset.zarr_path="${ZARR}" \
  task.scale_dataset.zarr_path="${ZARR}" \
  task.dataset.val_ratio=0.1 \
  use_recon=True use_vib=True dynamics_type='diffusion' dynamics.prediction_mode='full' \
  training.num_epochs=2 training.num_critic_epochs=2 dynamics.dynamics_max_epochs=2 \
  training.rollout_every=10000 training.checkpoint_every=1 \
  dataloader.batch_size=16 val_dataloader.batch_size=16 \
  optimizer.lr=2e-4 critic.q_lr=2e-4 critic.v_lr=2e-4 dynamics.dynamics_lr=4.4e-4 \
  encoder_type='resnet' encoders.resnet.share_rgb_model=False \
  encoders.resnet.rgb_model.weights='r3m' \
  encoders.resnet.recon_loss_weight=0.05 kl_annealing=False offline_use_aug=True \
  encoders.resnet.kl_beta=5e-4 \
  chunk_as_single_action=True bppo_chunk_level_ratio=True \
  offline_chunk_ratio_mode=scalar offline_chunk_adv_mode=scalar_iql \
  critic.q_layer_norm=True critic.q_hidden_dim=1024 critic.v_hidden_dim=512 \
  dynamics.prediction_mode='full' dynamics.dynamics_hidden_dims=[1024,1024,512,512] \
  predict_r=True \
  task.critic_dataset.sequence_stride=16 task.finetune_dataset.sequence_stride=16 \
  unio4.bppo_steps=0 unio4.eval_times=1 task.env_runner.eval_episodes=1 \
  +unio4.critic_artifact_dir="${CRITIC_DIR}"
