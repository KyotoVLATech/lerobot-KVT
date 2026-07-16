# Pseudo Aloha制御用ライブラリ
## Setup
```bash
git clone -b dev/sushi --recursive https://github.com/KyotoVLATech/lerobot-KVT.git
cd lerobot-KVT
uv sync --extra intelrealsense --extra dynamixel
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
## Usage
- USBデバイスの固定化 (推奨)
再起動のたびにデバイス名 (`/dev/ttyUSBX`) が入れ替わるのを防ぐため、以下のスクリプトを実行して Dynamixel アダプタを固定名にマッピングしてください。
```bash
sudo bash scripts/setup_udev.sh
# メニューから 1) Apply Rules を選択
```
実行後、`/dev/ttyUSB_LeftDynamixel` および `/dev/ttyUSB_RightDynamixel` が作成されます。

- デバイス名の確認
```bash
ls -l /dev/ttyUSB*
```
`iloha_server.py` の `initialize_robot` メソッド内（86行目付近）でポートを割り当てます。
```py
# 固定名 (udev) を使用する場合の例
left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
# RobStride は現状通り /dev/ttyUSBx を指定
left_robstride_port="/dev/ttyUSB2",
right_robstride_port="/dev/ttyUSB3",
```
- カメラデバイスの確認
```bash
uv run lerobot-find-cameras realsense
```
`outputs/captured_images/realsense_<serial_number>.png`に画像が出力される。
それをもとに`iloha_server.py`の32行目付近、カメラ設定を調整。
- 実行
電源を投入してから以下のコマンドを実行。
```bash
uv run iloha_server.py
```
- train
```bash
export DATASET_NAME=aloha-dataset-1
export POLICY=act
uv run lerobot-train \
  --dataset.repo_id=local/${DATASET_NAME} \
  --dataset.root=datasets/${DATASET_NAME} \
  --dataset.video_backend=pyav \
  --policy.type=$POLICY \
  --output_dir=outputs/train/${POLICY}-${DATASET_NAME} \
  --job_name=${POLICY}-${DATASET_NAME} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --batch_size=8 \
  --steps=200000
```

- eval
ACT
```bash
uv run iloha_eval.py \
    --policy_path outputs/train/act_iloha-dataset-good/checkpoints/100000/pretrained_model \
    --dataset_path datasets/iloha-dataset-good \
    --episode_time_s 30 \
    --num_episodes 1 \
    --save_data \
    --disable_robot_relative_safety
```
X-VLA
```bash
uv run --extra xvla iloha_eval.py \
    --policy_path outputs/train/xvla_iloha-dataset-all/checkpoints/100000/pretrained_model \
    --dataset_path datasets/iloha-dataset-all \
    --episode_time_s 20 \
    --num_episodes 1 \
    --task "Grab the edge of the towel and fold it twice. Quality: High"
```
pi0.5（動かない）
```bash
uv run --extra xvla iloha_eval.py \
    --policy_path outputs/train/pi05_iloha-dataset-fix/checkpoints/020000/pretrained_model \
    --dataset_path datasets/iloha-dataset-fix \
    --episode_time_s 30 \
    --num_episodes 1 \
    --save_data
```
## データセット関連
### データセットの修復
- 壊れているか確認
```bash
uv run fix_dataset.py iloha-1 --check-only
```
- 壊れているファイルを修復（--no-backupで.bakバックアップなし）
```bash
uv run fix_dataset.py iloha-1
```
### 特定エピソードの削除
```bash
uv run lerobot-edit-dataset --repo_id local/iloha-11 --root datasets/iloha-11 --new_root datasets/iloha-11 --operation.type delete_episodes --operation.episode_indices "[9]"
```
### タスク指示書き換え
- 一括変更
```bash
uv run change_task.py iloha-dataset-success \
  --mode all \
  --new-task "Grab the edge of the towel and fold it twice. Quality: High"
```
- エピソード単位
```bash
uv run change_task.py iloha-dataset-good \
  --mode episode \
  --episodes "0,2,5-8" \
  --new-task "Fold the towel twice from the edge."
```
- タスク単位
```bash
uv run change_task.py iloha-dataset-good \
  --mode task \
  --from-task-index 0 \
  --new-task "Fold the towel twice from the edge."
```
### データセットのマージ
```bash
uv run merge_dataset_v30.py
```

## モデル学習用 Docker
Ubuntu 24.04 / Python 3.12 / uv の学習用 Dockerfile を `docker/Dockerfile.train` に追加しています。
環境構築時のモデルオプションは `sarm`, `pi`, `xvla` の3つです。
`sarm` と `pi` は同じ `pi-sarm` イメージとしてビルドされ、`xvla` は transformers 依存の衝突を避けやすいように別イメージとしてビルドされます。

### イメージのビルド
```bash
bash scripts/model_docker.sh --model sarm build
# pi も同じ pi-sarm イメージを使います
bash scripts/model_docker.sh --model pi build

# xvla は別イメージです
bash scripts/model_docker.sh --model xvla build
```

### Hugging Face / W&B ログイン
以下を一度実行すると、ログイン情報はホスト側の `.cache/model-docker/` 以下に保存されます。
同じリポジトリでコンテナを作り直してもキャッシュは再利用されます。
```bash
bash scripts/model_docker.sh --model sarm login
```

### コンテナを開く
```bash
bash scripts/model_docker.sh --model sarm shell
```

コンテナ内ではリポジトリが `/workspace/lerobot` にマウントされます。
`datasets/` と `outputs/` も同じ場所にマウントされるため、学習結果はホスト側にも残ります。

### 任意コマンドの実行
学習・評価コマンドは用途に応じて自由に指定してください。
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/model_docker.sh --model sarm run -- lerobot-info
CUDA_VISIBLE_DEVICES=0 bash scripts/model_docker.sh --model sarm run -- lerobot-train \
  --dataset.repo_id=local/iloha-dataset-good \
  --dataset.root=datasets/iloha-dataset-good \
  --dataset.video_backend=pyav \
  --policy.type=sarm \
  --policy.annotation_mode=single_stage \
  --policy.image_key=observation.images.cam_high \
  --output_dir=outputs/train/sarm_single \
  --batch_size=32 \
  --steps=5000 \
  --wandb.enable=true \
  --wandb.project=sarm \
  --job_name=sarm-iloha-dataset-good \
  --policy.device=cuda \
  --policy.push_to_hub=false
```

`DATASET_DIR`, `OUTPUT_DIR`, `CACHE_ROOT`, `IMAGE_NAME` を環境変数で指定すると、マウント先やイメージ名を変更できます。

CUDA_VISIBLE_DEVICES=0 bash scripts/model_docker.sh --model sarm run -- python src/lerobot/policies/sarm/compute_rabc_weights.py \
  --dataset-repo-id=local/iloha-dataset-fix \
  --dataset-root=datasets/iloha-dataset-fix \
  --reward-model-path outputs/train/sarm_single/checkpoints/005000 \
  --visualize-only \
  --num-visualizations 5 \
  --head-mode sparse \
  --output-dir ./sarm_viz

- pi0.5学習 on Docker
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/model_docker.sh --model sarm run -- lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=local/iloha-dataset-fix \
  --dataset.root=datasets/iloha-dataset-fix \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["joint_6", "joint_13"]'
  --use_rabc=true \
  --rabc_head_mode=sparse \
  --rabc_kappa=0.01 \
  --batch_size=32 \
  --steps=40000 \
  --job_name=pi05_rabc \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --dataset.video_backend=pyav \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16
```
- ACT学習 on Docker
```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/model_docker.sh --model sarm run -- lerobot-train --dataset.repo_id=local/iloha-dataset-good --dataset.root=datasets/iloha-dataset-good --policy.type=act --output_dir=outputs/train/act_iloha-dataset-good --job_name=act_iloha-dataset-good --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=100000
```
-X-VLA学習 on Normal
```bash
uv run lerobot-train \
  --policy.path="lerobot/xvla-folding" \
  --dataset.repo_id=local/iloha-dataset-good \
  --dataset.root=datasets/iloha-dataset-good \
  --output_dir=outputs/train/xvla_iloha_folding \
  --job_name=xvla_iloha_folding \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --dataset.video_backend=pyav \
  --batch_size=8 \
  --steps=20000 \
  --policy.device=cuda \
  --policy.action_mode=auto \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --rename_map='{"observation.images.cam_high":"observation.images.image","observation.images.cam_left_wrist":"observation.images.image2","observation.images.cam_right_wrist":"observation.images.image3"}'
```
