# bash scripts/gen_demonstration_metaworld.sh basketball



cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 50 \
            --root_dir "../../RL-100/data/" \
