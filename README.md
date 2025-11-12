# Pseudo Aloha制御用ライブラリ
## Setup
```bash
git clone --recursive https://github.com/KyotoVLATech/lerobot-KVT.git
cd lerobot-KVT
uv sync
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
## Usage
- USBデバイス名の確認
```bash
ls /dev/ttyU*
```
`src/lerobot/my_aloha_server.py`の86行目付近を変更
```py
right_dynamixel_port="/dev/ttyUSB0",
right_robstride_port="/dev/ttyUSB2",
left_robstride_port="/dev/ttyUSB3",
left_dynamixel_port="/dev/ttyUSB1",
```
- カメラデバイスの確認
```bash
uv run lerobot-find-cameras realsense
```
`outputs/captured_images/realsense_<serial_number>.png`に画像が出力される。
それをもとに`src/lerobot/my_aloha_server.py`の32行目付近、カメラ設定を調整。
- 実行
電源を投入してから以下のコマンドを実行。
```bash
uv run src/lerobot/my_aloha_server.py
```