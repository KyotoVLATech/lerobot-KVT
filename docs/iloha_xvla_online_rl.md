# Iloha / X-VLA の通常 EXPO-FT

`iloha_xvla_rl.py` は、学習済み X-VLA に対するオンライン RL 学習サーバーです。
`libs/expo-ft/expo_ft/agents/alg/expo_ft.py` の通常版 EXPO-FT を参照した PyTorch 実装で、
RTC の事前学習は必要ありません。ロボット側には既存の `iloha_rl.py` を使います。

## 起動

sushi モデルの `060000` チェックポイント、1 エピソード 20 秒・エピソード数無制限、
`Grab the edge of the towel and fold it twice. Quality: High` の指定で起動するには、
リポジトリ内の別々の端末で以下を実行してください。

```bash
# GPU 側
bash scripts/run_iloha_xvla_rl.sh server

# ロボット側（同じ PC ならそのまま。別 PC なら HOST=<GPU側IP> を付ける）
bash scripts/run_iloha_xvla_rl.sh client
```

通常の評価コマンドも `bash scripts/run_iloha_xvla_rl.sh eval` で実行できます。
オンライン RL は Ctrl+C で終了するまで続きます。`eval` は元の評価コマンドと同じ 5 エピソードです。
RL は保存済み前処理の正規化統計を使い、評価時の `--dataset_path` は `eval` モードに渡します。
RL の保存先は実行日時付きの `outputs/rl/xvla_sushi_expo_ft_YYYYMMDD_HHMMSS/` です。
`--dry-run` をモードの前に付けると、実機に接続せず起動コマンドを確認できます。
環境変数と追加引数の指定は `bash scripts/run_iloha_xvla_rl.sh --help` を参照してください。

GPU 側で、実機で評価済みの Iloha 用 X-VLA チェックポイントを指定します。
14 次元の絶対 ALOHA 関節行動を出す `auto` / `joint` モデルが対象です。
保存済みの前処理・後処理ファイルと正規化統計も必要です。

```bash
uv run --extra xvla iloha_xvla_rl.py \
  --policy_path outputs/train/xvla_iloha-dataset-all/checkpoints/060000/pretrained_model \
  --output_dir outputs/rl/xvla_expo_ft \
  --port 8106
```

ロボット側で以下を起動します。同一 PC なら `127.0.0.1`、別 PC なら GPU 側の IP を指定します。
`--task` は学習済みモデルの実機評価と同じ指示文にしてください。
`iloha_eval.py` の状態入力に合わせるため、例では `--state_source command` を指定しています。
実測状態で学習したモデルでは `measured` を選んでください。

```bash
uv run --extra xvla iloha_rl.py \
  --host 127.0.0.1 --port 8106 \
  --observation_format xvla \
  --state_source command \
  --reward_mode terminal-score \
  --episode_time_s 30 --fps 30
```

1. 環境を整え、ロボット側の端末で Enter を押して開始します。
2. 30 秒経過、または端末で Enter（`end` + Enter も可）を押すと終了します。
3. ロボットが待機状態に入った後、**0〜1 の実数で報酬を入力**します。例: `0.7`。
   範囲外、数値でない入力、NaN / infinity は受け付けません。
4. 入力した値を終端報酬として学習し、次のエピソードの環境リセットを待ちます。

エピソード中に成功・失敗を入力する必要はありません。
時間切れも自動で報酬ゼロにはせず、終了後に人間が採点します。
`terminal-score` はクライアントの既定値です。従来の pi0.5 の二値判定を使う場合は
`--reward_mode binary` を明示してください。

## 学習する内容

- X-VLA から既定で 2 個の行動チャンクを生成し、Gaussian 編集ポリシーから 2 個の編集候補を作ります。
  ターゲット Q の低い側の値で比較し、最も高い候補を実行します。
- 既定では 8 ステップごとに再計画します。RTC の接頭辞条件付けは使いません。
  推論中は直前の関節指令を保持する同期方式です。
- 編集ポリシー、2 個の Q 関数、エントロピー温度を off-policy で更新します。
  VLA の次行動は更新時に再生成し、ターゲット Q は Polyak 更新します。
- X-VLA は報酬が正のエピソードの実行行動を模倣して更新します。
  ミニバッチに複数エピソードを含める場合、模倣損失を採点値で重み付けします。
  `--imitate_failures` を指定するとゼロ報酬も含めた等重みの模倣に切り替わります。
- 学習対象の行動は、ロボットから返された**安全制限適用後の実際の指令値**です。
  チャンクの終端が途中に来た場合は、実行した部分だけを X-VLA の損失に使います。
  Q 関数用の固定長表現は最後の実行行動で埋めます。
  終端で bootstrap を無効にし、報酬と継続 discount は実行ステップ数で計算します。

公式コードは pi0.5 / JAX 用なので、その Learner を直接呼んではいません。
この移植は、4090 向けに VLM の凍結特徴を共有し、twin Q と現在の VLA による候補生成を使います。
公式の別画像エンコーダー・Q ensemble・ターゲット VLA の全パラメータ複製を持たないため、
論文の設定や結果をそのまま再現する実装ではありません。

## RTX 4090 向けの既定設定

Florence の画像・言語エンコーダーを凍結し、推論した特徴を CPU にキャッシュします。
画像は `iloha_eval.py` と同じ cam_high クロップ済み元画像から、保存済みの rename / tokenizer /
normalizer と X-VLA 内のリサイズ処理を使って入力します。pi0.5 用 224px リサイズは通しません。
学習時に画像・言語エンコーダーを再計算する必要はありません。

X-VLA の学習対象は既定で **soft prompt のみ**です。凍結部分は BF16、学習対象は FP32、
Transformer の各ブロックには gradient checkpointing を使用します。
VLA の更新は microbatch 1 で、RL 更新 4 回につき 1 回です。
CPU の replay は既定で 256 チャンクです。VLA 特徴にはメモリを使うため、容量を増やす際は RAM も確認してください。

`--train_scope transformer` で行動 Transformer 全体も学習できますが、Adam の状態と勾配が増えます。
4090 の動作検証値は soft prompt の既定構成に対するものです。
収集の最初の 32 チャンクでは編集を適用せず、既存 VLA の行動を使います。
replay への追加と学習はエピソード終了後なので、初回エピソードは全体がこの warmup の対象になります。
初期 VLA が一度も正の報酬を得られない場合、正のロールアウトを使う VLA 模倣更新は始まりません。

既定の編集幅は関節角 ±0.03 rad、グリッパー ±0.03 です。
既存クライアントの相対制限・初動制限・実行結果フィードバックを引き継ぎます。
学習や採点の前、および切断時には待機状態へ移行します。

## 保存・再開・実機評価

5 エピソードごとに `episode_000005/` のようなディレクトリへ保存します。
セッション終了時にも `latest/` を保存します。
`pretrained_model/` は通常の X-VLA と同じ重み・前処理・後処理で、`iloha_eval.py` で評価できます。
この評価では編集ポリシーは使わず、更新された X-VLA 本体の能力を測ります。
`learner.pt` には編集ポリシー、Q、最適化状態、CPU replay、乱数状態、FP32 の VLA 学習対象を保存します。

```bash
uv run --extra xvla iloha_xvla_rl.py \
  --policy_path outputs/rl/xvla_expo_ft/latest/pretrained_model \
  --resume outputs/rl/xvla_expo_ft/latest \
  --output_dir outputs/rl/xvla_expo_ft --port 8106

uv run --extra xvla iloha_eval.py \
  --policy_path outputs/rl/xvla_expo_ft/latest/pretrained_model \
  --dataset_path datasets/iloha-dataset-all
```

再開時は replan_steps / train_scope / hidden_dim / 編集幅を保存時と一致させてください。
`--episodes N` は再開分を含めた総エピソード数です。既定の 0 は無制限です。
ロールアウトの動画を残す場合はクライアントに `--save_data` を追加します。
学習に使う報酬・特徴・実行行動は Learner checkpoint に保存します。

## 実機なしでの検証

```bash
uv run --extra xvla scripts/iloha_xvla_rl_verify.py \
  --policy_path outputs/train/xvla_iloha-dataset-all/checkpoints/060000/pretrained_model \
  --device cuda

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_iloha_xvla_rl.py -q
```

検証スクリプトは合成 RGB 観測で推論、非終端・終端の Q 更新、編集ポリシー更新、
実際の soft prompt 更新、重み・前処理の保存再読込、Learner 再開を確認します。
ロボットには接続しません。出力の VRAM 値は PyTorch の最大割当量・予約量で、
他プロセスや CUDA ドライバーのメモリは含みません。
実機での学習効果・成功率・動的タスクでの制御性能は別途評価が必要です。

参照: [公式 EXPO-FT コード](https://github.com/pd-perry/expo-ft/)、
[EXPO-FT 論文](https://arxiv.org/abs/2605.25477)。
