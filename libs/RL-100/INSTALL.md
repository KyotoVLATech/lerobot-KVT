# RL-100 Installation

This guide installs the environment used by RL-100 for simulation training, iterative offline RL, online RL, and flow/diffusion policy distillation.

The setup below was verified on the current server with:

```text
NVIDIA driver: 550.54.15
System CUDA shown by nvidia-smi: 12.4
Python: 3.8.20
PyTorch: 2.4.0+cu121
torch.version.cuda: 12.1
```

The driver supports CUDA 12.4, while the verified PyTorch wheel is the CUDA 12.1 build. This is expected: CUDA 12.1 PyTorch wheels run correctly on the CUDA 12.4 driver.

## 1. Create the Environment

### Recommended on this server

The existing `dp3` environment is known to run this repo. The verified `rl100` environment was created by cloning it, then reinstalling editable packages to this repo path:

```bash
conda create -n rl100 --clone dp3 -y
conda activate rl100
```

### From scratch

For a clean machine, start with Python 3.8 and the verified PyTorch version:

```bash
conda create -n rl100 python=3.8 -y
conda activate rl100

python -m pip install "setuptools==59.5.0" wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Then install the Python package set used by RL-100:

```bash
python -m pip install \
  zarr==2.12.0 wandb==0.20.1 ipdb==0.13.13 gpustat==1.1.1 \
  dm_control==1.0.23 omegaconf==2.3.0 hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.8.1 diffusers==0.33.1 huggingface_hub==0.33.1 numba==0.56.4 \
  moviepy==1.0.3 imageio==2.35.1 av==12.3.0 matplotlib==3.7.5 termcolor==2.4.0 \
  open3d==0.19.0 opencv-python==4.11.0.86 scipy==1.10.1 scikit-learn==1.3.2 \
  scikit-image==0.21.0 pandas==2.0.3 h5py==3.11.0 trimesh==4.6.12 \
  pybullet==3.2.7 plyfile==1.0.3 transforms3d==0.4.2 shapely==2.0.7 rtree==1.3.0 \
  tensorboardx==2.6.2.2 tqdm==4.67.1 colorlog==6.9.0 tabulate==0.9.0 \
  gdown==5.2.0 ftfy==6.2.3 regex==2024.11.6 yacs==0.1.8 \
  fvcore==0.1.5.post20221221 iopath==0.1.10 blessed==1.21.0 wcwidth==0.2.13 \
  timm==1.0.26 transformers==4.40.0
```

Real-robot and camera tools are optional for simulation-only runs:

```bash
python -m pip install \
  ur_rtde==1.6.1 atomics==1.0.3 viser==0.2.23 pyrealsense2 \
  dt_apriltags fpsample zerorpc yapf==0.43.0 littleutils==0.2.4 \
  sorcery==0.2.2 wrapt==1.17.2 diffusers==0.33.1 \
  "timm>=0.9.0" "torchvision>=0.15.0" "einops>=0.6.0"
```

The optional package set is:

```text
ur_rtde==1.6.1
atomics==1.0.3
viser==0.2.23
pyrealsense2
dt_apriltags
fpsample
zerorpc
yapf
# ViT encoder dependencies
timm>=0.9.0
torchvision>=0.15.0
einops>=0.6.0
yapf==0.43.0
littleutils==0.2.4
sorcery==0.2.2
wrapt==1.17.2
diffusers==0.33.1
```

Some real-robot packages, such as `pyrealsense2`, `ur_rtde`, and hardware wrappers, may require matching hardware/OS support. If you only run simulated Adroit/DexArt/MetaWorld training, install failures from unavailable hardware packages can be handled separately.

## 2. Set Runtime Paths

`RL-100/` is not a pip-installable Python package in this repo because it has no `setup.py` or `pyproject.toml`. Use `PYTHONPATH`:

```bash
export REPO_ROOT=/path/to/RL-100-repo
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
```

For MuJoCo/EGL rendering:

```bash
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The online scripts set `CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID`, and `EGL_DEVICE_ID` based on the selected GPU.

## 3. Install MuJoCo

Install MuJoCo 2.1 under `~/.mujoco`:

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```

On a clean Ubuntu machine, `mujoco-py` EGL builds usually also need:

```bash
sudo apt-get update
sudo apt-get install -y libglew-dev libgl1-mesa-dev libosmesa6-dev libglfw3 libglfw3-dev patchelf ninja-build
```

## 4. Install Editable Repo Dependencies

From the repo root:

```bash
conda activate rl100

python -m pip install -e third_party/dexart-release
python -m pip install -e third_party/gym-0.21.0
python -m pip install -e third_party/Metaworld
python -m pip install -e third_party/rrl-dependencies/mj_envs/.
python -m pip install -e third_party/rrl-dependencies/mjrl/.
python -m pip install -e third_party/mujoco-py-2.1.2.14
python -m pip install -e third_party/pytorch3d_simplified
python -m pip install -e visualizer
```

Important notes:

- Do **not** run `pip install -e RL-100`; `RL-100/` is imported through `PYTHONPATH`.
- The current `third_party/r3m` directory does not contain an installable package. The verified server environment currently imports `r3m` from an external editable install. For a clean release, either populate `third_party/r3m` with the R3M source or install an equivalent `r3m` package before running image/R3M encoder code paths.
- `gym` should resolve to this repo's `third_party/gym-0.21.0`.

Check editable locations:

```bash
python -m pip show dexart gym metaworld mj-envs mjrl mujoco-py pytorch3d visualizer r3m
```

Editable project locations should point to this repo, except `r3m` if it is intentionally installed from an external source.

## 5. Assets and Data

Before training, make sure these paths exist when the selected task needs them:

```text
third_party/dexart-release/assets       # DexArt assets
third_party/VRL3/ckpts                  # Adroit expert checkpoints, if generating demos
RL-100/data/*.zarr                      # offline datasets
```

For Adroit medium tasks such as `adroit_door_medium`, the dataset path is configured in:

```text
RL-100/rl_100/config/task/adroit_door_medium.yaml
```

## 6. Sanity Checks

Run these from the repo root:

```bash
conda activate rl100
export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import rl_100, zarr, hydra, einops, metaworld, mujoco_py, open3d as o3d; print('imports ok', o3d.__version__)"
python -c "import gym; print('gym version:', gym.__version__)"
```

Expected:

```text
torch 2.4.0+cu121, torch.version.cuda 12.1, cuda available True
imports ok 0.19.0
gym version: 0.21.0
```

## 7. Verified Flow Online Distillation Run

The flow online distillation launcher is:

```text
scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh
```

Use this standard example command in the `rl100` environment:

```bash
conda activate rl100
export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

./scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

This expands to `train.py --config-name=rl100_3d_flow.yaml` with:

```text
task=adroit_door_medium
training.seed=100
++ppo.train_env_num=8
++ppo.eval_env_num=8
policy.scheduler_type=flow
distill_phase=online
flow_distill_inference_steps=1
flow_distill_teacher_steps=10
```

PG / PG + IDQL-style extraction and one-step distillation are launcher-level sweep settings. For offline distillation, finish the offline sweep first and then set `distill_phase='after_offline'`; for online distillation, set `distill_phase='online'`.

This launcher path was validated in the `rl100` environment: it reached the `python train.py` training process and initialized MuJoCo/EGL successfully. Since it is a long online training job (`ppo.max_train_steps=1000000`, `training.num_epochs=200`), the validation run was stopped after confirming the training entry process was active.

## 8. Common Issues

### Mixed editable installs

If you have several copies of this repo on the same machine, stale editable installs can silently import code from the wrong path.

Use:

```bash
python -m pip show dexart gym metaworld mj-envs mjrl mujoco-py pytorch3d visualizer r3m
```

and reinstall editable packages from the current repo if needed.

### `ModuleNotFoundError: rl_100`

Set:

```bash
export PYTHONPATH=$(pwd)/RL-100:${PYTHONPATH}
```

### `r3m` import issues

The current repo does not include an installable `third_party/r3m` package. Install or populate R3M before using R3M-based image encoders.

### MuJoCo/OpenGL errors

Check:

```bash
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
