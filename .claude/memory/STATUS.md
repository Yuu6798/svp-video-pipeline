# STATUS.md — svp-video-pipeline プロジェクト状況

## Phase

M シリーズ実装が進行中（直近は planner レスポンス解析や image audit のリファクタ PR #27–#31 がマージ）。2026-07-07 に ugh-prompt-engine から開発規律インフラ（discipline テスト・session memory・AGENTS.md 分業規約）を移植し、運用を開始した。

## Next-Issue Queue

| ID | Title | Priority | Notes |
|---|---|---|---|
| README-1 | svp_pipeline/README.md 343 行 → docs/<topic>.md 分離 | P2 | hard limit 350 行目前（README 管理ポリシー）。単一 section が 30 行を超える箇所を洗い出し、`docs/<topic>.md` へ抽出してリンク化する |
| TYPE-1 | 型チェッカー（mypy/pyright）導入検討 | P3 | 現状 ruff の型ヒントチェック（型ヒント必須の規約）はあるが静的型チェッカー未導入。導入コスト/CI 時間への影響を評価してから判断 |
| CI-1 | CI markdown-lint ジョブの実質化 | P3 | `.github/workflows/ci.yml` の `markdown-lint` ジョブは `README.md` の非空チェックのみで実質的な lint をしていない。markdownlint 等の導入を検討 |
| TEST-1 | テスト規約（モック不使用推奨）と実態の乖離解消 | P3 | CLAUDE.md の Testing 節「モック不使用を推奨」は `tests/fixtures/mock_*.py`（mock_openai / mock_gemini / mock_fal / mock_drive / mock_responses）の実態と乖離。外部 API 呼び出しの多いこのリポジトリでは mock 前提が実情に合うため、CLAUDE.md 文言を実態に合わせて修正する |

## Recently Merged

| PR | Title | Date | Phase |
|---|---|---|---|
| #31 | refactor: split derived identity lock checks (planner response parsing) | 2026-05-06 | M シリーズ リファクタ |
| #30 | refactor: group manual image audit findings (image audit entrypoint) | 2026-05-06 | M シリーズ リファクタ |
| #29 | refactor: split reference policy rendering | 2026-05-06 | M シリーズ リファクタ |
| #28 | refactor: split openai image generation | 2026-05-06 | M シリーズ リファクタ |
| #27 | refactor: split audit workflow orchestration | 2026-05-06 | M シリーズ リファクタ |
