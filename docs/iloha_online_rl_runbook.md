# iLoHa オンラインRL 実行手順

更新日: 2026-10-06。このPCでLearnerと実機クライアントを別ターミナルで起動する。

## 現在の確認状況

- GPU単体検証: 指定checkpointと実データ128件で、学習更新3回と更新後の推論が成功。
- 全デモ222エピソード・86,896フレームの読み込み、Learner初期化、実機接続、3台のカメラ、Policy動作まで確認。
- 実機動作について、操作者から「正常に動いているように見えた」との報告あり。
- 同時に、ログにはDynamixelの `There is no status packet!` / `Incorrect status packet!` と一部RobStrideタイムアウトが出た。
  ログだけで実際の異常動作を断定することはできない。原因・実測値への影響・安定した30 Hz制御は未確認。
- 通信ログを受けて検証を中断したため、最後の「0＝失敗」はLearnerに反映されていない。
  **実機エピソード終了 → 学習更新 → 更新済みPolicyの実機動作という一連の流れは、まだ完了確認していない。**
- 最後の実機クライアントは終了済み。検証Learnerは一時停止したため、再起動前に以下の既存プロセス確認を行う。

## 使用データと設定

| 項目 | 設定 |
| --- | --- |
| 初期Policy | `libs/expo-ft/checkpoints/iloha_towel_rtc_offline/pi_rtc_iloha_towel_high_maxdelay10/checkpoints/10000/params` |
| 成功デモ | `libs/expo-ft/data/iloha_towel/success`（全件、`num_data=0`） |
| batch size | 16 |
| critic UTD | 10（1回のLearner更新でcriticを10回更新） |
| 候補数 | `N=8`, `n_edit_samples=8`, `filter_N=8` |
| actor gradient accumulation | 4件 × 4分割、有効batch 16、optimizer更新は1回 |
| GPUメモリ | fraction 0.95、preallocate無効、platform allocator |
| host側 | UTD batch 160件をCPUに保持し16件ずつGPUへ転送。画像replayとtarget PolicyはSSD退避 |
| RAM保護 | LearnerにRAM 20 GiB / swap 1 GiB上限 |
| RTC | delay 5、replan 8 |
| Policyへ渡すstate | `measured`（意図された定義は変更しない） |

ベースのソフトウェアオフセットは**右180度・左270度**。
`right_settings.py` / `left_settings.py` に設定済みで、全腕の初期化が正常であることを操作者が確認済み。
他の機体にそのまま適用しない。

## 1. 起動前の確認

ロボットの電源はOFFのまま、リポジトリのルートへ移動する。

```bash
cd /home/kupac/SourceCode/lerobot-KVT
systemctl --user status iloha-rl-learner.scope
ss -ltn 'sport = :8104'
df -h .
```

既存Learnerがあれば二重起動しない。一時停止した前回の検証Learnerを終了する場合は、
**実機クライアントを終了しロボットを電源OFFにした状態で**、次を実行する。

```bash
systemctl --user stop --no-block iloha-rl-learner.scope
systemctl --user kill --signal=SIGCONT iloha-rl-learner.scope
systemctl --user status iloha-rl-learner.scope
```

停止要求を先に出し、一時停止中のプロセスを起こして終了させるための手順。
ユニットが既に終了していれば `Unit ... not found` になることがある。
`status`で終了を確認してから新しいLearnerを起動する。機体の動作を再開する目的では使わない。

全デモの展開画像だけで約36.55 GiBを使う。target重み・キャッシュ・ログ用の余裕も必要。
中断時には `libs/expo-ft/replay_cache/replay-*` が残る場合がある。
使用中のキャッシュや元データは削除しない。容量500,000件を画像で埋めると約210 GiBとなるため、
長期学習には現在のSSD空き容量だけでは不足する可能性がある。

## 2. Learnerを先に起動する（ターミナルA）

まずは以下の**実機検証用**コマンドを使う。全デモは維持し、最初のエピソードから
1回のLearner更新を試す。checkpoint保存・replay保存・W&B外部送信は無効。
`run_name`は必要に応じて新しい名前に変える。

```bash
cd /home/kupac/SourceCode/lerobot-KVT
WANDB_MODE=offline \
HF_HOME=/tmp/iloha-rl-hf \
OPENPI_DATA_HOME=/tmp/iloha-rl-openpi \
JAX_COMPILATION_CACHE_DIR=/tmp/iloha-rl-jax-cache \
bash libs/expo-ft/scripts/iloha_towel/run_server.sh \
  --output_dir=./checkpoints/iloha_towel_verify \
  --run_name=real_robot_manual_verify \
  --nocheckpoint_buffer \
  --nocheckpoint_model \
  --learning_starts_episodes=0 \
  --num_updates=1 \
  --notqdm
```

このPCではLearnerに `libs/expo-ft/.venv`、実機にルートの `.venv` を使用する。
起動スクリプトがLearner環境を選び、RAM上限付きのsystemd scope内で実行する。
`/tmp`のモデル資産・コンパイルキャッシュは今回の検証で用意したもの。
削除後は再取得・再コンパイルが必要になる場合がある。

**ロボットの電源はまだ入れない。** 全デモ読み込みには今回約4分かかった。
次のログを待つ。

```text
Creating environment...
EnvClient listening for rollout client on 0.0.0.0:8104
```

別ターミナルで確認する場合:

```bash
ss -ltn 'sport = :8104'
systemctl --user show iloha-rl-learner.scope \
  --property=MemoryCurrent,MemoryPeak,MemoryMax,MemorySwapMax
```

`LISTEN`を確認してから実機を起動する。この時点では実機観測での初回推論はまだ行っていないため、
エピソード開始後に追加のコンパイル待ちが入る場合がある。

## 3. 電源ON・実機クライアント起動（ターミナルB）

タオルを配置し、周囲を空け、非常停止・電源OFFをすぐに行える状態にする。
**クライアント起動直後に全腕の初期化・初期姿勢への移動が始まる。**

```bash
cd /home/kupac/SourceCode/lerobot-KVT
.venv/bin/python -u iloha_rl.py --host 127.0.0.1 --port 8104
```

全腕初期化、3台のカメラ接続、Learner接続を確認する。
別マシンにLearnerを置く場合だけ、`127.0.0.1`をそのマシンのIPへ変更する。
相対制限安全制御は有効のまま使い、`--disable_robot_relative_safety`は指定しない。

## 4. エピソード開始と成否入力

入力するのは**実機クライアントのターミナルB**。
通常の手動実行ではチャットへの入力はプログラムへ転送されない。

| 入力 | 動作 |
| --- | --- |
| 準備待ちでEnter | 初期位置復帰後、そのエピソードのPolicy動作を開始 |
| エピソード中に `0` + Enter | タスク失敗で終了 |
| エピソード中に `1` + Enter | タスク成功で終了 |
| 最初のstepから30秒経過 | 時間切れ失敗で終了 |

終了後は初期位置への復帰・モータ停止を行い、次のエピソード開始入力を待つ。
**Enterによる次エピソード開始は、学習更新の完了ログを確認してから行う。**
終了後のリセットと学習は並行して進むため、準備待ち表示だけでは学習完了を意味しない。

## 5. 学習更新を確認する

ターミナルAで次のログを確認する。

```text
Learner update completed: actor_step=1 critic_step=10 actor_loss=... critic_loss=...
```

今回の検証設定では、エピソード終了ごとにLearner更新を1回試す。
actorは1回、criticはUTD 10回更新する。損失が `nan` / `inf` でないことを確認する。
収集ループが16ステップに達する前に終了した場合は、最初の更新は実行されない。
全デモを使った実機更新の時間はまだ未測定。先の128件GPU検証では1回約26〜36秒だった。

更新完了後、ターミナルBでタオルを再配置してEnterを押し、更新済みPolicyで次のエピソードを行う。
成否入力 → 更新ログ → 次エピソードまで通って初めて、実機オンラインRLの連携確認とする。

## 6. 通常学習へ切り替える場合

検証完了後、Learnerを終了して通常設定で再起動する。
ターミナルAのコマンドから検証用の末尾引数を外す。

```bash
cd /home/kupac/SourceCode/lerobot-KVT
WANDB_MODE=offline \
HF_HOME=/tmp/iloha-rl-hf \
OPENPI_DATA_HOME=/tmp/iloha-rl-openpi \
JAX_COMPILATION_CACHE_DIR=/tmp/iloha-rl-jax-cache \
bash libs/expo-ft/scripts/iloha_towel/run_server.sh
```

通常は10エピソード完了後に更新を有効化する。
`num_updates=0`では、収集した30遷移につきLearner更新1回をエピソード境界で行い、端数を繰り越す。
これはcritic UTD 10とは別の頻度設定。学習開始前の遷移を後からまとめて更新する処理はしない。
batch 16・候補数8・UTD 10・4×4 accumulation・全デモは同じ。

通常の出力先は `libs/expo-ft/checkpoints/iloha_towel/ours_iloha_towel_high_delay5`。
モデル保存・収集遷移保存が有効で、モデル保存間隔は20,000収集ステップ。
同じ出力先のcheckpointがあれば `--resume`で再開する。指定の初期Policyを必ず使う新規実験には
別の `--run_name`を指定し、既存checkpointを上書きしない。
RAM上限内での大きなcheckpoint保存・復元と長期間の運転は別途確認が必要。

## 7. 終了・異常時

通常終了は、エピソードを判定して待機状態になった後、学習更新の完了を確認し、
ターミナルBでCtrl+C、次にターミナルAでCtrl+C。
**Ctrl+Cは非常停止ではない。クライアントの終了処理にも原点復帰の位置指令が含まれる。**
終了処理の間も周囲を空け、停止確認に失敗した場合は電源OFFする。

意図しない動作、通信が途切れて停止確認できない場合は、まず非常停止／電源OFFを行う。
通信エラー表示があっても見た目の姿勢は正常な場合があるため、ログと動作は分けて記録する。
一方、実測stateの読取り失敗時はコードが指令値で補完するため、正常な見た目だけで実測取得の健全性は判断しない。
通信ログを無視して安定した制御・学習が確認済みと扱わない。

この手順書の作成ではプロセスの再開・実機動作は行っていない。
