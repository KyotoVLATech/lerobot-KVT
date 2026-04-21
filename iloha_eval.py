#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np

from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.processor.rename_processor import rename_stats
from iloha_mapping import JOINT_NAMES, aloha_to_iloha, iloha_to_aloha


TASK = "Grab the edge of the towel and fold it twice."
SUPPORTED_POLICY_TYPES = {"act", "xvla", "pi05"}
CAMERA_MAX_FRAME_AGE_MS = 250
CAM_HIGH_CROP_SIZE = (480, 640)  # height, width
RELATIVE_WARMUP_SECONDS = 3.0
ABSOLUTE_MODE_DELTA_THRESHOLD = 0.2  # rad

# カメラ設定（iloha_server.pyと同じ）
CAMERA_CONFIGS = {
    "cam_high": {"serial_number_or_name": "146222252104", "width": 1280, "height": 720, "fps": 30},
    "cam_left_wrist": {"serial_number_or_name": "341522301205", "width": 640, "height": 480, "fps": 30},
    "cam_right_wrist": {"serial_number_or_name": "029522250086", "width": 640, "height": 480, "fps": 30}
}


async def reset_robot_to_home(robot: Iloha, init=True):
    """
    ロボットを初期位置に戻す（iloha_server.pyのhandle_reset_requestと同じロジック）
    """
    print("ロボットを初期位置に戻しています...")

    home_action = robot.old_action.copy()
    home_action[3:7] = 0.0
    home_action[10:14] = 0.0
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)

    home_action = np.zeros_like(home_action)
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)
    
    print("初期位置復帰完了")


def initialize_cameras() -> dict:
    """カメラを初期化して辞書で返す"""
    try:
        camera_configs = {}
        for name, config_dict in CAMERA_CONFIGS.items():
            camera_configs[name] = RealSenseCameraConfig(**config_dict)
        
        cameras = make_cameras_from_configs(camera_configs)
        
        for name, camera in cameras.items():
            print(f"{name} を接続中...")
            camera.connect(warmup=True)
            time.sleep(1.0)
        
        print(f"{len(cameras)}台のカメラを初期化しました")
        return cameras
    except Exception as e:
        print(f"カメラ初期化エラー: {e}")
        return {}


def get_next_dataset_number(root: Path, prefix: str = "aloha-eval-") -> int:
    """既存のデータセット番号を確認し、次の番号を返す"""
    if not root.exists():
        return 0
    
    existing_nums = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            try:
                num = int(path.name.split("-")[-1])
                existing_nums.append(num)
            except ValueError:
                continue
    
    return max(existing_nums) + 1 if existing_nums else 0


def load_model_config(policy_path: str) -> dict:
    """pretrained_model/config.jsonを読み込み、ポリシー種別やrelative設定を確認する"""
    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_preprocessor_rename_map(policy_path: str) -> dict[str, str]:
    """保存済みpreprocessorから観測rename_mapを読み込む"""
    preprocessor_path = Path(policy_path) / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        return {}
    with preprocessor_path.open("r", encoding="utf-8") as f:
        preprocessor_config = json.load(f)

    for step in preprocessor_config.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step.get("config", {}).get("rename_map", {}))
    return {}


def crop_cam_high_for_dataset(image: np.ndarray) -> np.ndarray:
    """cam_highの中央下部640x480をLeRobotDataset/Policy入力用に切り出す"""
    crop_h, crop_w = CAM_HIGH_CROP_SIZE
    height, width = image.shape[:2]
    if height < crop_h or width < crop_w:
        raise ValueError(
            f"cam_high image is too small for {crop_w}x{crop_h} crop: got {width}x{height}"
        )
    top = height - crop_h
    left = (width - crop_w) // 2
    return np.ascontiguousarray(image[top:top + crop_h, left:left + crop_w])


def read_camera_frame(camera) -> np.ndarray:
    """最新フレーム取得を優先し、未準備時だけ同期readにフォールバックする"""
    try:
        return camera.read_latest(max_age_ms=CAMERA_MAX_FRAME_AGE_MS)
    except Exception:
        try:
            return camera.async_read(timeout_ms=CAMERA_MAX_FRAME_AGE_MS)
        except Exception:
            return camera.read()


def capture_observation(robot: Iloha, state_names: tuple[str, ...]) -> dict:
    """iloha_server.pyと同じく、画像とrobot.old_action由来のALOHA状態を観測にする"""
    obs = {}
    for name, camera in robot.cameras.items():
        obs[name] = read_camera_frame(camera)
    if "cam_high" in obs:
        obs["cam_high"] = crop_cam_high_for_dataset(obs["cam_high"])

    aloha_state = iloha_to_aloha(robot.old_action)
    for i, joint_name in enumerate(state_names):
        obs[joint_name] = float(aloha_state[i])
    return obs


def action_tensor_to_aloha_array(action_tensor, action_names: tuple[str, ...]) -> np.ndarray:
    """Policy出力をaction names順のALOHA座標numpy配列にそろえる"""
    if isinstance(action_tensor, dict):
        if all(name in action_tensor for name in action_names):
            values = []
            for name in action_names:
                value = action_tensor[name]
                if hasattr(value, "detach"):
                    value = value.detach().float().cpu().numpy()
                values.append(float(np.asarray(value).squeeze()))
            return np.asarray(values, dtype=np.float32)

        if "action" in action_tensor:
            action_tensor = action_tensor["action"]
        else:
            raise KeyError(f"action dict does not contain expected keys: {list(action_tensor.keys())}")

    if hasattr(action_tensor, "detach"):
        action_array = action_tensor.detach().float().cpu().numpy()
    else:
        action_array = np.asarray(action_tensor)
    action_array = np.asarray(action_array, dtype=np.float32).squeeze()
    if action_array.ndim != 1:
        raise ValueError(f"Expected 1D action after squeeze, got shape {action_array.shape}")
    if action_array.shape[0] < len(action_names):
        raise ValueError(f"Action has {action_array.shape[0]} dims, expected at least {len(action_names)}")
    return action_array[:len(action_names)]


def enable_pi05_relative_actions_if_needed(preprocessor, postprocessor, action_names: tuple[str, ...]) -> None:
    """pi0.5のrelative action用processorを、モデルconfigに合わせて明示的に有効化する"""
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )

    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is None:
        print("警告: pi0.5 relative actionが有効ですが、preprocessorにrelative stepがありません")
        return

    relative_step.enabled = True
    relative_step.action_names = list(action_names)
    absolute_step = next((s for s in postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep)), None)
    if absolute_step is None:
        print("警告: pi0.5 relative actionが有効ですが、postprocessorにabsolute stepがありません")
        return
    absolute_step.enabled = True
    absolute_step.relative_step = relative_step


def check_policy_dependencies(policy_type: str) -> None:
    """Policyごとのoptional dependencyが無い場合、ハードウェア接続前に分かりやすく止める"""
    missing = []
    if policy_type == "xvla" and importlib.util.find_spec("transformers") is None:
        missing.append("transformers")
    if policy_type == "pi05" and importlib.util.find_spec("transformers") is None:
        missing.append("transformers")

    if missing:
        extra = "xvla" if policy_type == "xvla" else "pi"
        raise RuntimeError(
            f"{policy_type} policy requires optional dependency: {', '.join(missing)}.\n"
            f"Run with the matching extra, for example:\n"
            f"  uv run --extra {extra} iloha_eval.py --policy_path ... --dataset_path ..."
        )


async def evaluation_loop(
    robot: Iloha,
    policy,
    preprocessor,
    postprocessor,
    device,
    fps: int,
    episode_time_s: float,
    task: str,
    ds_features: dict,
    dataset: Optional[LeRobotDataset] = None,
    display_data: bool = False,
    use_robot_relative_safety: bool = True,
    relative_warmup_seconds: float = RELATIVE_WARMUP_SECONDS,
    absolute_mode_delta_threshold: float = ABSOLUTE_MODE_DELTA_THRESHOLD,
):
    """
    評価ループ: ポリシーからアクションを予測してロボットを制御
    """
    print(f"評価ループ開始（{episode_time_s}秒間）")
    
    frame_count = 0
    start_episode_t = time.perf_counter()
    action_names = tuple(ds_features["action"]["names"])
    state_names = tuple(ds_features["observation.state"]["names"])
    
    while True:
        start_loop_t = time.perf_counter()
        
        # 時間チェック
        elapsed = time.perf_counter() - start_episode_t
        if elapsed >= episode_time_s:
            print(f"エピソード時間（{episode_time_s}秒）に達しました")
            return frame_count
        
        # 1. サーバ実装に合わせて、最新画像とold_action由来のALOHA状態を取得
        obs_for_policy = capture_observation(robot, state_names)
        
        if frame_count == 0:
            print(f"観測データのキー: {list(obs_for_policy.keys())}")
        
        # 2. データセット形式のフレームを構築
        observation_frame = build_dataset_frame(
            ds_features, 
            obs_for_policy,
            prefix="observation"
        )
        
        if frame_count == 0:
            print(f"observation_frameのキー: {list(observation_frame.keys())}")
        
        # 3. ポリシーでアクションを予測
        try:
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.name,
            )
            
            # 4. Tensor/dictをnumpy配列に変換し、ALOHA座標からIloha送信座標へ変換
            action_array_aloha = action_tensor_to_aloha_array(action_tensor, action_names)
            predicted_action_values = {
                name: float(action_array_aloha[i]) for i, name in enumerate(action_names)
            }
            action_array_iloha = aloha_to_iloha(action_array_aloha)
            
            if frame_count == 0:
                print(f"予測されたアクション形状: {action_array_aloha.shape}")
                print(f"Alohaアクション値（最初の3要素）: {action_array_aloha[:3]}")
                print(f"Iloha送信値（最初の3要素）: {action_array_iloha[:3]}")
        
        except Exception as e:
            print(f"アクション予測エラー: {e}")
            import traceback
            traceback.print_exc()
            return frame_count
        
        # 6. ロボットにアクションを送信。初動と急変時はiloha_server.pyと同じ安全側の相対制限を使う
        previous_action = robot.old_action.copy()
        delta_from_previous = np.abs(action_array_iloha - previous_action)
        max_delta = float(np.max(delta_from_previous))
        use_relative = use_robot_relative_safety and (
            elapsed < relative_warmup_seconds or max_delta > absolute_mode_delta_threshold
        )
        await robot.async_send_action(
            action_array_iloha,
            use_relative=use_relative,
            use_filter=not use_relative,
        )
        actual_action_aloha = iloha_to_aloha(robot.old_action)
        actual_action_values = {
            name: float(actual_action_aloha[i]) for i, name in enumerate(action_names)
        }
        
        # 7. データセットに保存（オプション）
        if dataset is not None:
            action_frame = build_dataset_frame(
                ds_features, 
                actual_action_values,
                prefix="action"
            )
            frame = {**observation_frame, **action_frame, "task": task}
            dataset.add_frame(frame)
        
        # 8. 可視化（オプション）
        if display_data:
            log_rerun_data(observation=obs_for_policy, action=predicted_action_values)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"フレーム: {frame_count}, 経過時間: {elapsed:.1f}秒")
        
        # 9. FPS制御
        dt_s = time.perf_counter() - start_loop_t
        sleep_duration = 1.0 / fps - dt_s
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
    
    print(f"評価ループ終了（合計{frame_count}フレーム）")
    return frame_count


async def main(args):
    init_logging()

    # 1. ポリシーとプロセッサの読み込み
    # 依存関係エラーをロボット・カメラ接続前に検出する
    print("=" * 60)
    print(f"ポリシーを読み込み中: {args.policy_path}")

    print(f"データセット読み込み中: {args.dataset_path}")
    dataset_for_stats = LeRobotDataset(args.dataset_path, root=args.dataset_path)

    model_config = load_model_config(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = "cuda" if args.device == "cuda" else "cpu"

    policy_type = str(model_config.get("type", policy_cfg.type)).lower()
    if policy_type == "pi0.5":
        policy_type = "pi05"
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"未対応のPolicy typeです: {policy_type}. "
            f"対応: {', '.join(sorted(SUPPORTED_POLICY_TYPES))}"
        )
    check_policy_dependencies(policy_type)
    if policy_type == "pi05":
        relative_enabled = bool(
            model_config.get("use_relative_actions", getattr(policy_cfg, "use_relative_actions", False))
        )
        policy_cfg.use_relative_actions = relative_enabled
        print(f"pi0.5 relative action: {'enabled' if relative_enabled else 'disabled'}")
    print(f"Policy type: {policy_type}")

    rename_map = load_preprocessor_rename_map(args.policy_path)
    if rename_map:
        print(f"Rename map: {rename_map}")

    policy = make_policy(policy_cfg, ds_meta=dataset_for_stats.meta, rename_map=rename_map)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=rename_stats(dataset_for_stats.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    if policy_type == "pi05" and getattr(policy_cfg, "use_relative_actions", False):
        enable_pi05_relative_actions_if_needed(
            preprocessor,
            postprocessor,
            tuple(dataset_for_stats.features["action"]["names"]),
        )

    device = get_safe_torch_device(policy_cfg.device)
    print(f"ポリシー読み込み完了（デバイス: {device}）")
    
    # 2. ロボットの初期化と接続
    print("=" * 60)
    print("ロボットを初期化中...")
    config = IlohaConfig(
        left_robstride_port="/dev/ttyUSB3",
        left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
        right_robstride_port="/dev/ttyUSB2",
        right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
        max_relative_target_1=0.03,
        max_relative_target_2=0.01,
        max_relative_target_3=0.01,
        max_relative_target_4=0.03,
        max_relative_target_5=0.01,
        max_relative_target_6=0.03,
        current_limit_robstride={1: 4.0, 2: 16.0, 3: 4.0, 4: 4.0, 5: 16.0, 6: 4.0},
        current_limit_gripper_R=0.3,
        current_limit_gripper_L=0.3,
    )
    robot = Iloha(config, debug=False)
    await robot.connect()
    print("ロボット接続完了")
    
    # 3. 初期位置に戻す
    await reset_robot_to_home(robot)
    
    # 4. カメラの初期化
    print("=" * 60)
    print("カメラを初期化中...")
    cameras = initialize_cameras()
    if not cameras:
        print("エラー: カメラの初期化に失敗しました")
        await robot.disconnect()
        return
    robot.cameras = cameras
    
    # 5. データセットの作成（保存する場合）
    dataset = None
    video_encoding_manager = None
    
    if args.save_data:
        print("=" * 60)
        print("データセット作成中...")
        
        dataset_root = Path(args.output_root)
        dataset_num = get_next_dataset_number(dataset_root, prefix="aloha-eval-")
        dataset_name = f"aloha-eval-{dataset_num}"
        repo_id = f"local/{dataset_name}"
        dataset_path = dataset_root / dataset_name
        
        # データセット特徴量の定義
        dataset_features = {
            "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        }
        for key in CAMERA_CONFIGS.keys():
            dataset_features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ("height", "width", "channels")
            }
        
        dataset = LeRobotDataset.create(
            repo_id,
            args.fps,
            root=dataset_path,
            robot_type="aloha",
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=len(cameras),
            video_backend="pyav",
        )
        
        video_encoding_manager = VideoEncodingManager(dataset)
        video_encoding_manager.__enter__()
        
        print(f"データセット作成完了: {repo_id}")
    
    # 6. 可視化の初期化（オプション）
    if args.display_data:
        init_rerun(session_name="evaluation")
    
    # 7. エピソードループ
    print("=" * 60)
    print(f"{args.num_episodes}エピソードの評価を開始します")
    
    try:
        for episode_idx in range(args.num_episodes):
            print(f"\n--- エピソード {episode_idx + 1}/{args.num_episodes} ---")
            
            # ポリシーとプロセッサをリセット
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            
            # 評価ループを実行
            frame_count = await evaluation_loop(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                fps=args.fps,
                episode_time_s=args.episode_time_s,
                task=args.task,
                ds_features=dataset_for_stats.features,
                dataset=dataset,
                display_data=args.display_data,
                use_robot_relative_safety=not args.disable_robot_relative_safety,
                relative_warmup_seconds=args.relative_warmup_seconds,
                absolute_mode_delta_threshold=args.absolute_mode_delta_threshold,
            )
            
            # エピソードを保存
            if dataset is not None:
                if frame_count > 0:
                    dataset.save_episode()
                    print(f"エピソード {episode_idx + 1} を保存しました")
                else:
                    print(f"エピソード {episode_idx + 1} は0フレームのため保存をスキップしました")
            
            # 次のエピソードのためにロボットを初期位置に戻す
            if episode_idx < args.num_episodes - 1:
                print(f"\n次のエピソードのためにロボットをリセットします...")
                await reset_robot_to_home(robot, init=False)
                await asyncio.sleep(2.0)  # リセット後の待機時間
    
    except KeyboardInterrupt:
        print("\n中断されました")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 8. クリーンアップ
        print("=" * 60)
        print("クリーンアップ中...")
        
        # データセットのクリーンアップ
        if video_encoding_manager:
            video_encoding_manager.__exit__(None, None, None)
        if dataset:
            dataset.finalize()
            print("データセットを終了しました")
        
        # カメラの切断
        for name, camera in cameras.items():
            try:
                camera.disconnect()
                print(f"{name} を切断しました")
            except Exception as e:
                print(f"{name} 切断エラー: {e}")
        
        # ロボットを初期位置に戻して切断
        await reset_robot_to_home(robot, init=False)
        await robot.disconnect()
        print("ロボット切断完了")
        
        print("=" * 60)
        print("評価スクリプト終了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みPolicyでIlohaロボットを評価")
    
    # 必須引数
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="学習済みポリシーのパス（例: outputs/train/act-aloha-dataset-0/checkpoints/last/pretrained_model）"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="統計情報を取得するデータセットのパス（例: datasets/aloha-dataset-0）"
    )
    
    # オプション引数
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets/eval",
        help="保存先ルートディレクトリ（デフォルト: datasets/eval）"
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="観測データを保存する"
    )
    parser.add_argument(
        "--episode_time_s",
        type=float,
        default=60.0,
        help="1エピソードの実行時間（秒）（デフォルト: 60）"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="実行するエピソード数（デフォルト: 1）"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="制御周波数（Hz）（デフォルト: 30）"
    )
    parser.add_argument(
        "--display_data",
        action="store_true",
        help="rerunでリアルタイム可視化"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK,
        help=f"タスク指示文（デフォルト: {TASK}）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推論デバイス（デフォルト: cuda）"
    )
    parser.add_argument(
        "--disable_robot_relative_safety",
        action="store_true",
        help="初動・急変時のIloha相対制限安全制御を無効にする"
    )
    parser.add_argument(
        "--relative_warmup_seconds",
        type=float,
        default=RELATIVE_WARMUP_SECONDS,
        help=f"初回アクションから相対制限を強制する秒数（デフォルト: {RELATIVE_WARMUP_SECONDS}）"
    )
    parser.add_argument(
        "--absolute_mode_delta_threshold",
        type=float,
        default=ABSOLUTE_MODE_DELTA_THRESHOLD,
        help=f"相対制限を維持する最大差分しきい値rad（デフォルト: {ABSOLUTE_MODE_DELTA_THRESHOLD}）"
    )
    
    args = parser.parse_args()
    
    # 非同期でメイン関数を実行
    asyncio.run(main(args))
