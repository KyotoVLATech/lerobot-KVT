# Iloha Trajectory Studio

リポジトリのルートで起動します。実機には接続しません。データセットは読み取り専用で、このツールがファイルを書き換えることはありません。

```sh
uv run iloha_trajectory_web.py
```

ブラウザで <http://127.0.0.1:8765> を開いてください。NumPy、PyTorch、Unity、Node、CDNは起動時に不要です。Python 3.12以降とPyArrowを使用し、`uv run` がスクリプト専用の依存環境を用意します。PyArrowが導入済みなら `python iloha_trajectory_web.py` でも起動できます。

```sh
uv run iloha_trajectory_web.py --datasets-root /path/to/datasets --port 8765
```

## 操作

- 初期状態で `iloha-best` → `iloha-common` のエピソード0を読み込みます。別のデータセットやエピソードも追加できます。クリップの↑で順番を変更できます。
- 開始・終了時刻または2つのトリムハンドルで不要な区間を除きます。時刻は**元データの秒数**です。
- 各クリップに `iloha_catch_replay.py` と同じ5つの速度設定があります。ベース速度、追加倍率上限、グリッパー前後の余白、追加倍率までの範囲、グリッパー判定閾値を独立に指定します。初期値はCLI実装の `1 / 2 / 0.5 / 1 / 0.0001` に合わせています。
- クロスディゾルブは速度変更後の末尾・先頭を重ね、smoothstepの重みで関節角を混合します。線形補間は両端の間に移動区間を追加します。「滑らかな補間」は両端の速度を使う三次Hermite補間で、速度を接続しますが位置のオーバーシュートは許容します。「そのまま接続」は境界を補間しません。
- 接続時間は**速度変更後の秒数**です。処理順は「各元クリップの速度スケジュールを計算 → 指定区間を取り出して時間を変更 → 境界接続 → 理想軌道の3D表示」です。グリッパーからの距離はトリム前の元エピソードで判定します。
- 灰色の理想手先軌道を常に全区間表示します。両腕の現在の理想姿勢は関節を点、リンクを線で描き、「現在の姿勢」で表示を切り替えられます。手首ロールとグリッパー開度は先端の横線に反映します。
- 再生バーで任意の時刻に移動すると、現在の理想姿勢が連動します。「境界へ」で接続区間の手前に移動できます。
- ドラッグで3D回転、Shift+ドラッグで平行移動、ホイールで拡大縮小できます。正面・上面・側面の視点もあります。
- プレビュー速度は画面の再生速度だけを変更します。物理的な再生速度の変更は各クリップの速度設定で行います。
- 「設定を保存／読込」でクリップ選択・編集・速度・ベース間隔を保存できます。「軌道を書き出す」で実機用の `trajectory_settings.json` を保存できます。実機再生スクリプトへの自動投入は行いません。

## 実機で再生する設定の書き出し

右上の「軌道を書き出す」で `trajectory_settings.json` をダウンロードします。**画面で再生している軌道がそのまま実機の動きになります。** 使用区間・各クリップの速度設定・境界の接続方式・再構成に使うFPSを記録し、フレーム列そのものは持ちません。

```sh
uv run iloha_catch_game2.py --settings trajectory_settings.json --color blue
```

`iloha_catch_game2.py` は設定に書かれた元データセットを読み、`iloha_trajectory_replay.py`（`core.mjs` のプレビュー計算をPythonへ移植したもの）でブラウザと同じフレーム列を組み立ててから、一定周期で実機へ送信します。速度変更はフレーム間隔として既に織り込まれているため、再生側で速度を指定する引数はありません。再生と原点復帰が終わると、これまで通り片腕の遠隔操作へ移ります。`--dry_run` で実機に接続せず内容だけ確認できます。設定が参照するデータセットは `--datasets_root`（既定: `datasets`）から探します。

```json
{
  "schema_version": 3,
  "speed_applied": true,
  "clips": [{"dataset": "iloha-best", "episode": 0, "start": 0, "end": 166.3,
             "replay": {"base_speed": 1, "max_speedup": 2, "gripper_margin": 0.5,
                        "speedup_distance": 1, "gripper_threshold": 0.0001}}],
  "edit": {"mode": "crossfade", "blend": 1.2, "fps": 60, "time_basis": "speed_adjusted_seconds"},
  "trajectory": {"fps": 60, "frames": 8507, "duration": 141.77, "segments": [], "boundaries": []}
}
```

`trajectory.frames` は実機側の検算に使います。元データセットが書き出し時と変わっていてフレーム数が一致しない場合は、ロボットを動かさずにエラーにします。動作確認用デモを含む設定は実機で再生できないため、書き出し時に拒否します。

以前の「理想軌道をLeRobotデータセットへ書き出す」機能（`iloha_trajectory_dataset.py` と `POST /api/export-dataset`）は削除しました。速度設定を適用せずに書き出す仕様のため、画面で再生した軌道とは別物のデータセットができていました（例: 画面141.8秒に対して書き出し198.4秒）。設定ファイルから同じ軌道を再現できるようになったため、データセットの複製自体が不要です。サーバーは読み取り専用になり、書き込みAPIはありません。

## 3D表示と設定の互換性

`ref_ik.cs` のリンク長 `[0.1, 0.305834, 0.2033, 0.0967, 0.07015, 0.03]` mと角度変換を逆算した順運動学を使用します。解析座標はZ上、+YがUnityの+X（右）です。左ベースY=-0.295 m、右Y=+0.295 m、初期間隔590 mm。ベース間隔は画面から変更できます。

実軌道シミュレーションとその制限設定は削除しました。表示するのは指令データの理想軌道と現在の理想姿勢のみで、実機の追従誤差や電流・加速度制限の影響を予測するものではありません。

プロジェクトJSONはversion 2で、編集・再生速度・表示用ベース間隔を保存します。旧version 1のプロジェクトも読み込めますが、廃止したシミュレーション設定は無視します。既存のデータセットや保存済みJSONは変更しません。

## 検証

```sh
node --test tests/test_iloha_trajectory_core.mjs
uv run --no-project --with pyarrow python -m unittest discover -s tests -p test_iloha_trajectory_web.py
# ブラウザ実装とPython実装が同じフレーム列を作ることをNodeと突き合わせて確認
uv run --no-project --with numpy python -m unittest discover -s tests -p test_iloha_trajectory_replay.py
# 実データを一時領域にコピーし、検証用サーバーとインストール済みChromeを使用
uv run --no-project --with pyarrow --with numpy --with playwright python tests/check_iloha_trajectory_browser.py
```

書き出した設定から実機側が再構成したフレーム列が、ブラウザのプレビューとビット単位で一致することを確認しています。

`meta/info.json` だけでは再生できません。`data/**/*.parquet` の `action`, `episode_index`, `frame_index` 列を読み込みます。データ欠損時に自動で人工データへ置換することはありません。動作確認用デモは明示的なボタン操作でのみ読み込みます。
