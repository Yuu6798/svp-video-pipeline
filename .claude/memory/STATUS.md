# STATUS.md — svp-video-pipeline プロジェクト状況

## Phase

M シリーズ実装が進行中（直近は planner レスポンス解析や image audit のリファクタ PR #27–#31 がマージ）。2026-07-07 に ugh-prompt-engine から開発規律インフラ（discipline テスト・session memory・AGENTS.md 分業規約）を移植し、運用を開始した。同日、確定済み Design Memo に基づき技術負債改善（docs/ 新設 + README 分離、タイムスタンプ UTC 統一、エラーハンドリング是正、pyproject/CI 強化、CLAUDE.md 文言同期の P1 全解消）を実施した。同日 Session 3 で PLANNER-1（generator/planner.py の 4 シナリオ×同形メソッド群を generator/planner_rules.py の宣言的ルールテーブル + 単一適用エンジンに集約、test_planner.py 無変更で全 pass・合計 200 行削減）を実施した。同日 Session 4 で TEST-2（test_planner.py・test_cli.py を計 10 ファイルへテスト名グルーピングで分割し、`tests/fixtures/helpers.py`・`tests/fixtures/fakes.py` に SAMPLES_DIR/`_load`/`_write_png`/Fake* 系の重複定義を集約、全 345 件 pass・ruff clean を維持）を実施した。同日 Session 5 で ENV-1（`.env.example` の死にエントリ `DEFAULT_IMAGE_MODEL` 削除）と LOG-1（`pipeline.py:_build_log_data` で log.json の float を書き込み時に 4 桁丸め、新規テスト 1 件追加で全 346 件 pass）を実施した。同日 Session 6 で PROBE-0（A/A ノイズ床計測ハーネス `svp-video probe-noise` + `probe/noise.py`・`probe/corpus.py` 新設、`noise_floor.json` レポート、`docs/measurement.md` 新設、fake-backend テスト 25 件追加で全 371 件 pass）を実施した。副産物として `main` コマンドが `probe-noise` 追加で Typer が単一コマンド→Group 構成に切り替わり暗黙起動が壊れる問題を `_DefaultCommandGroup`（`TyperGroup` 継承、`parse_args` で未知の先頭トークンに `main` を前置）で解消し既存 CLI 起動方法を完全維持した。

## Next-Issue Queue

| ID | Title | Priority | Notes |
|---|---|---|---|
| TYPE-1 | 型チェッカー（mypy/pyright）導入検討 | P3 | 現状 ruff の型ヒントチェック（型ヒント必須の規約）はあるが静的型チェッカー未導入。導入コスト/CI 時間への影響を評価してから判断 |
| DEP-1 | 依存の floor-only 制約と lockfile 不在の解消方針決定 | P3 | fast-moving SDK（anthropic/openai/google-genai）の非再現ビルドリスク |
| PROBE-1 | probe-noise の実測実行（ノイズ床 v1 データ取得） | P2 | マシン依存（API 課金）。User/Codex トラック。runbook は docs/measurement.md |
| GRIP-1 | SVP 1 フィールド摂動による grip 計測ハーネス | P2 | PROBE-1 のノイズ床が分母。device_profiles 化まで |
| SENSOR-1 | CLIP/顔 identity 埋め込みセンサーの隔離層導入 | P3 | 画素メトリクスの意味盲を補完。optional 依存 |

## Recently Merged

| PR | Title | Date | Phase |
|---|---|---|---|
| #31 | refactor: split derived identity lock checks (planner response parsing) | 2026-05-06 | M シリーズ リファクタ |
| #30 | refactor: group manual image audit findings (image audit entrypoint) | 2026-05-06 | M シリーズ リファクタ |
| #29 | refactor: split reference policy rendering | 2026-05-06 | M シリーズ リファクタ |
| #28 | refactor: split openai image generation | 2026-05-06 | M シリーズ リファクタ |
| #27 | refactor: split audit workflow orchestration | 2026-05-06 | M シリーズ リファクタ |
