# Iloha × RL-100 パイプライン (sushi データセット)

`datasets/iloha-dataset-sushi`（LeRobot v3.0, bimanual ALOHA互換, 14次元 state/action,
3RGBカメラ, 深度なし）を使って、RL-100 で **模倣学習(BC) → オフラインRL → 実機オンラインRL**
を回すための一式。深度は使用しない。

## 追加/変更ファイル

| ファイル | 役割 |
|---|---|
| `iloha_to_rl100_zarr.py` | LeRobot v3.0 → RL-100 2D zarr 変換（3RGB, 14次元, next_*, reward/done/return, episode_ends） |
| `libs/RL-100/RL-100/rl_100/dataset/iloha.py` | `Cloth` を継承した 14次元 state 用データセット（`use_velocity=True`＋`sequence_stride`対応） |
| `libs/RL-100/RL-100/rl_100/config/task/sushi.yaml` | RL-100 タスク設定（action[14], agent_pos[14], 3RGB[3,240,320], `fake_env=True`） |
| `libs/RL-100/scripts/iloha/train_sushi_offline.sh` | BC + オフラインRL のラッパ（既存の chunk two-stage ランチャを流用） |
| `iloha_online_rl.py` | 実機オンラインRL / ロールアウト収集（`iloha_eval.py` + RL-100 diffusion-PPO） |

## 前提の環境

- **変換 (`iloha_to_rl100_zarr.py`)**: lerobot の uv venv（`uv pip install zarr numcodecs` 済み）。
- **学習 (`train_sushi_offline.sh`) / オンライン (`iloha_online_rl.py`)**: RL-100 の学習環境
  （`libs/RL-100/INSTALL.md`、conda `rl100` 等）。`iloha_online_rl.py` は lerobot の実機
  ドライバも import するため、RL-100 環境に lerobot も入っている／PYTHONPATH が通っている必要あり。

---

## Stage 0 — zarr 変換

```bash
# lerobot venv で
uv run python iloha_to_rl100_zarr.py \
  --dataset datasets/iloha-dataset-sushi \
  --out libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr
```

- 出力は zarr **v2** フォーマット（RL-100 の zarr2 スタック互換）。
- 画像は 240×320（HWC uint8, RGB）。current と next の 3RGB を保存するため
  **圧縮後 ~60–90GB** 程度になる。ディスクを確認すること（`--image_h/--image_w` で縮小可）。
- 検証済み: 生成 zarr の各フレーム/状態は `LeRobotDataset` とピクセル一致（エピソード境界含む）。
- 動作確認だけしたいときは `--max_episodes 8` で部分変換できる。

---

## Stage 1+2 — 模倣学習(BC) + オフラインRL

```bash
# RL-100 環境で、RL-100 の git ルート(libs/RL-100)から
cd libs/RL-100
bash scripts/iloha/train_sushi_offline.sh run0 100 1     # tag seed num_gpus
```

- 内部で `scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh rl100 sushi run0 100 1` を呼ぶ。
  - **Stage 1** = BC 初期化 + IQL critic + dynamics 学習（`unio4.bppo_steps=0`）。
  - **Stage 2** = BPPO によるオフラインRL fine-tune（最小構成で1ジョブ）。
- action-chunk（`n_action_steps=16, n_obs_steps=3, horizon=18`）。
- `sushi.yaml` は `env_runner.fake_env=True`。オフライン学習中のオンライン評価はダミー指標を返すため、
  **オフラインスイープの「best」自動選択は実質意味を持たない**（実評価は実機で行う）。チェックポイントは
  `data/outputs_2d_chunk/sushi-rl100-run0_seed100/.../best`（および各 run の `checkpoints/`）に出る。
- スモークテスト: `TRAIN_NUM_EPOCHS=20 bash scripts/iloha/train_sushi_offline.sh run0 100 1`。
- 本格スイープが要るときは既存ランチャを直接叩き、`LR_VALUES`/`ROLLOUT_VALUES` 等を広げる。

---

## Stage 3 — 実機オンラインRL / ロールアウト収集

まず **収集のみ**で実機ループを検証（PPO更新なし・安全）:

```bash
# RL-100 環境（lerobot 実機ドライバも import 可能）で
export PYTHONPATH=$(pwd)/libs/RL-100/RL-100:$PYTHONPATH
python iloha_online_rl.py --collect_only \
  --offline_ckpt libs/RL-100/RL-100/data/outputs_2d_chunk/sushi-rl100-run0_seed100/.../best \
  --zarr libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr
```

問題なければ **オンラインRL 本番**:

```bash
python iloha_online_rl.py \
  --offline_ckpt <best_dir> \
  --zarr libs/RL-100/RL-100/data/iloha_sushi_240_320.zarr \
  --episode_time_s 30 --terminal_reward 1.0
```

- **報酬**: 疎な終端の人手ラベル。各エピソード終端でオペレータがキー入力
  （数字=成功→`terminal_reward`, 文字=失敗→0）。`IlohaGymEnv._label_terminal` を差し替えれば
  成功分類器（`--reward_fn` 相当）に置換可能。
- ロールアウト（`batch_size` チャンク遷移）ごとに `dp_align_update_no_share`（diffusion-PPO）で更新。
- value 関数は policy の `obs2latent` 特徴を入力にする MLP ヘッド（`ppo.share_encoder=True`）で
  モダリティ非依存。dynamics はオンラインGAE経路では未使用。
- チェックポイントは `outputs/iloha_online/online_*`、ロールアウトは `.../rollouts/*.npz`。

### 注意

- `iloha_online_rl.py` の `ONLINE_ARCH_OVERRIDES` は **オフライン学習側のアーキ設定と一致**させること。
  ズレると offline チェックポイントの重みが正しく load できない。オフラインのジオメトリ/エンコーダを
  変えたらこの配列も更新する。
- 座標系: zarr/policy は ALOHA 座標。実機送出時に `aloha_to_iloha`、観測取得時に `iloha_to_aloha`
  で変換している（`iloha_eval.py` と同じ）。
- 収集した `.npz`/ロールアウトを zarr に統合し直せば、RL-100 の反復オフライン・データフライホイールに乗る。
