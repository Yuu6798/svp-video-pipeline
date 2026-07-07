# Session Memory Index

各セッションの 1 行要約。詳細は同ディレクトリの `YYYY-MM-DD.md` を参照。

- 2026-07-07: ugh-prompt-engine から開発規律インフラを移植（discipline テスト・memory 骨格・wrap-up/new-brief skills・AGENTS.md・SessionStart hook）。[詳細](2026-07-07.md)
- 2026-07-07 (Session 2): Design Memo に基づき技術負債改善を実施（docs/ 新設 + README 分離、utils/timestamps.py で UTC 統一、image_openai/pipeline のハード依存 import 是正、composite.py の重複委譲解消、pyproject FA/pytest-cov/addopts、CI readme-check 改称+concurrency、CLAUDE.md 文言同期）。全 334 テスト green・ruff clean。[詳細](2026-07-07.md)
- 2026-07-07 (Session 3): PLANNER-1 実装。planner.py の 4 シナリオ×同形 22 メソッド（815 行）を planner_rules.py の宣言的ルールテーブル + 単一 apply_scenario エンジンに集約。test_planner.py 無変更で全 83 件 pass、合計 200 行削減、新設 test_planner_rules.py 11 件追加。全 345 テスト green・ruff clean。[詳細](2026-07-07.md)
- 2026-07-07 (Session 4): TEST-2 実装。test_planner.py(1357行)を4分割・test_cli.py(1239行)を6分割し、tests/fixtures/helpers.py・fakes.py にSAMPLES_DIR/_load/_write_png/Fake*系の重複定義を集約。build_image_responseをbackend別名にリネーム。テスト名集合345件は前後で完全一致・ruff clean。[詳細](2026-07-07.md)
