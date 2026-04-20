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
- データセットのマージ
```bash
uv run merge_dataset_v30.py
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
```bash
uv run iloha_eval.py \
  --policy_path outputs/train/act-kitcut-dataset/checkpoints/200000/pretrained_model \
  --dataset_path datasets/kitcut-dataset \
  --episode_time_s 45 \
  --num_episodes 1
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
  --job_name=sarm-${iloha-dataset-good} \
  --policy.device=cuda \
  --policy.push_to_hub=false
```

`DATASET_DIR`, `OUTPUT_DIR`, `CACHE_ROOT`, `IMAGE_NAME` を環境変数で指定すると、マウント先やイメージ名を変更できます。
