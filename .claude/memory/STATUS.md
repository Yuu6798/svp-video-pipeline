# STATUS.md — svp-video-pipeline プロジェクト状況

## Phase

開発規律インフラ（discipline テスト・session memory・AGENTS.md 分業規約）の移植と能動対応すべき技術負債の解消を 2026-07-07 の PR #32–#37 で完了し、計測トラックの立ち上げ段階にある。PROBE-0（A/A ノイズ床計測ハーネス `svp-video probe-noise` + `docs/measurement.md`）はマージ済みで、次は PROBE-1（実測・課金ありのため User/Codex トラック）でノイズ床 v1 データを取得し、そのデータを分母として GRIP-1（SVP 1 フィールド摂動による grip 計測）の設計に進む。

## Next-Issue Queue

| ID | Title | Priority | Notes |
|---|---|---|---|
| PROBE-1 | probe-noise の実測実行（ノイズ床 v1 データ取得） | P2 | マシン依存（API 課金）。User/Codex トラック。runbook は docs/measurement.md |
| GRIP-1 | SVP 1 フィールド摂動による grip 計測ハーネス | P2 | PROBE-1 のノイズ床が分母。device_profiles 化まで |
| SENSOR-1 | CLIP/顔 identity 埋め込みセンサーの隔離層導入 | P3 | 画素メトリクスの意味盲を補完。optional 依存 |
| TYPE-1 | 型チェッカー（mypy/pyright）導入検討 | P3 | トリガー待ち（型起因バグの発生時に src/ 限定・緩め設定で試験導入） |
| DEP-1 | 依存の floor-only 制約と lockfile 不在の解消方針決定 | P3 | トリガー待ち（依存起因の CI 破損時に着手）。fast-moving SDK の非再現ビルドリスク |

## Recently Merged

| PR | Title | Date | Phase |
|---|---|---|---|
| #37 | feat: A/A ノイズ床計測ハーネス probe-noise (PROBE-0) | 2026-07-07 | 計測トラック土台 |
| #36 | fix: .env.example 死にエントリ削除 + log.json float 丸め (ENV-1, LOG-1) | 2026-07-07 | 技術負債改善 |
| #35 | test: 肥大テスト分割と共有 fixture 集約 (TEST-2) | 2026-07-07 | 技術負債改善 |
| #34 | refactor: planner のシナリオ制約をルールテーブル化 (PLANNER-1) | 2026-07-07 | 技術負債改善 |
| #33 | refactor: 技術負債改善 — P1 全解消と規約・実態の同期 | 2026-07-07 | 技術負債改善 |
