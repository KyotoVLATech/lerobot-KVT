# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

cd third_party/VRL3/src

task=${1}

MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 1000 \
                        --root_dir "../../../RL-100/data/" \
                        --expert_ckpt_path "../src/vrl3data/logs/exp_local/2024.10.17/135940_task=hammer/snapshot40000.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop \
                        --data_property 'medium' \

# --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
                       # --expert_ckpt_path "../src/vrl3data/logs/exp_local/2024.10.14/134128_task=door/snapshot20000.pt" \