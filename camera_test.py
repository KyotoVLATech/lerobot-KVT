import time
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
CAMERA_CONFIGS = {
    "cam_high": {"serial_number_or_name": "029522250086", "width": 640, "height": 480, "fps": 30},
    "cam_left_wrist": {"serial_number_or_name": "341522301205", "width": 640, "height": 480, "fps": 30},
    "cam_right_wrist": {"serial_number_or_name": "146222252104", "width": 640, "height": 480, "fps": 30}
}
camera_configs = {}
for name, config_dict in CAMERA_CONFIGS.items():
    camera_configs[name] = RealSenseCameraConfig(**config_dict)
cameras = make_cameras_from_configs(camera_configs)

# カメラを接続する（warmupを無効にして接続を高速化）
for name, camera in cameras.items():
    print(f"{name} を接続中...")
    camera.connect(warmup=False)
    time.sleep(1)  # カメラの初期化を待つ

print("\n全カメラの接続完了。画像取得を開始します...\n")
time.sleep(2)  # すべてのカメラが安定するのを待つ

# 画像を読み取る（タイムアウトを長くする）
obs = {}
for name, camera in cameras.items():
    print(f"{name} から画像取得中...")
    obs[f"observation.images.{name}"] = camera.read(timeout_ms=2000)

# 使用後は切断する
for name, camera in cameras.items():
    camera.disconnect()

print("画像の取得に成功しました")
for name, image in obs.items():
    print(f"{name}: {image.shape}, dtype={image.dtype}")
