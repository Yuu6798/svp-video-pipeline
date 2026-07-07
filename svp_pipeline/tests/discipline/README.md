# tests/discipline

セッション運用の規律ルール（CLAUDE.md § Session Memory / wrap-up skill /
AGENTS.md §8）を CI 失敗に変換する実行可能チェック群。
`ugh-prompt-engine` の `tests/discipline/` から移植・本リポジトリの
ネスト構成（`svp_pipeline/` 配下）に適応済み。

- `test_status_md_phase_single_paragraph.py`: `.claude/memory/STATUS.md` の
  `## Phase` は単一の正準段落を維持する。
- `test_status_md_next_queue_no_completed.py`: 完了/マージ済み item を
  `## Next-Issue Queue` に残置してはならない（`## Recently Merged` へ移動）。
- `test_index_md_entry_compactness.py`: `.claude/memory/_index.md` の各
  エントリは 500 文字以内（詳細は dated session log へ）。
- `test_claude_md_line_cap.py`: ルート `CLAUDE.md` は 400 行以内
  （always-loaded policy のため。reference detail は docs/ / skill に
  ポインタ化）。
- `test_readme_line_cap.py`: ルート `README.md` と `svp_pipeline/README.md`
  はともに 350 行以内（README 管理ポリシーの hard limit）。

各テストは実ファイル検査に加えて `fixtures/` の違反サンプルに対する
self-test を持ち、パーサ自体の劣化（違反を検出できなくなる drift）を防ぐ。

実行（wrap-up skill step 8 の pre-push gate と同一コマンド。リポジトリ
ルートから svp_pipeline/ に移動して実行する）:

```bash
cd svp_pipeline && python -m pytest tests/discipline/ -q
```

注: 本リポジトリの dev extras に pytest-cov は含まれないため
`--no-cov` は付けない（unrecognized argument でエラーになる）。
