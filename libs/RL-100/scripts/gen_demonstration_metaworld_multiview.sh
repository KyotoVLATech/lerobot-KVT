# bash scripts/gen_demonstration_metaworld.sh basketball



cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python generate_metaworld_multiview_data.py --env_name=${task_name} \
            --num_episodes 30 \
            --root_dir "../../RL-100/data/" \
