# CLAUDE.md — svp-video-pipeline

このファイルは Claude Code / Claude Agent SDK がこのリポジトリで作業する際の
普遍的な運用ポリシーをまとめる。リポジトリ固有の設計詳細は
`docs/<topic>.md` と各 `README.md` に分離する。

## Advisor Strategy（モデル運用方針）

**2026-07-07 改訂**（ugh-prompt-engine の 2026-07-05 体制を移植。高価な
メインエージェントを設計判断に最大集中させる措置）:

- **メインエージェント**: Fable 5 — **設計・設計判断のみ**（Design Memo 起草、
  レビュー指摘の採否判定と対応方針の設計、結果解釈・数値判読、メモリ管理）。
  **実装・実行・検証など実際に手を動かす作業はサブエージェントに委譲し、
  メイン自身は行わない**。Fable 非稼働セッションでは Opus が代行
- **実装・探索サブエージェント**: Sonnet 固定（実装、探索・読み取り中心の調査タスク）
- **実行・検証・非設計分析サブエージェント**: Opus または Sonnet（実測検証・
  E2E/スモーク実行、設計判断を伴わないレビュー指摘の分析・トリアージ）

運用細則（委譲固定費の逆ザヤ防止。Web ツール含む全ツールに適用）:

- **マイクロ操作例外**: 単発の状態確認・git 操作・メモリ読み書き・質問など **1–2
  コールかつ結果ペイロードが軽い操作のみ** メイン直接可。実装・実行・検証・探索・
  複数ソース Web 調査に加え、レビュースレッド取得/返信投稿など結果が重い操作は
  コール数によらず委譲する
- **生データ様式**: 検証・計測の委譲時は生データをファイル保存させメインが直接
  Read して判読する（要約経由の判断劣化防止。判読=設計判断でメインの職務）

Agent ツールで spawn する際は必ず `model` を明示すること。

```python
# 正しい例（実装・探索は Sonnet 固定）
Agent({"model": "sonnet", "subagent_type": "Explore", "prompt": "..."})

# NG — model 省略するとメインと同モデルで動き、コスト効率が下がる
Agent({"subagent_type": "Explore", "prompt": "..."})
```

## Workflow（Codex × Claude × User 分業）

Claude Code が Task Brief を読んで **Design Memo** を起草 → User が Codex に
渡して `codex/<topic>` ブランチで実装・PR 作成 → Claude Code が PR をレビュー
（指摘 or Approve）→ 対応往復 → User が最終マージ判断、のループを回す。
Claude Code は通常このリポジトリで実装コードを書かない（PR レビュー、
Design Memo、メモリ管理が担当領域）。メッセージ・フォーマット規約
（Design Memo / Completion Summary / エスカレーション条件等）の
source of truth は [`AGENTS.md`](AGENTS.md)。

## Session Memory（永続記憶ワークフロー）

セッション間の記憶喪失を防ぐため、`.claude/memory/` にセッションサマリーを蓄積する。

### 起動時ルール

1. セッション開始時に `.claude/memory/_index.md` を読み、過去の決定事項・コンテキストを把握する
2. 直近 3 件のサマリーファイルは必要に応じて詳細を参照する
3. 過去の設計判断に関する質問には、サマリーを確認してから回答する

### 終了時ルール（自動トリガー）

ユーザーがセッション終了を示す発言をしたら、**確認なしで即座に `/wrap-up` を実行する**。

**トリガーフレーズ**（文脈付きの終了意図を検出。汎用トークン単体では発火しない）:
- 「今日はここまで」「今日は終わり」「今日はおわり」
- 「セッション終了」「セッション閉じて」
- 「また明日」「また今度」「お疲れ様」「お疲れさま」
- 「done for today」「that's all」
- 手動: `/wrap-up`

`.claude/skills/wrap-up/SKILL.md` が終了手順全体の **source of truth**
（8 ステップ: reflection 保存 → `_index.md` 追記 → archive → STATUS.md
sweep → discipline ゲート）。本ファイルと skill が乖離した場合は
**skill が勝つ** — このポインタを直し、skill を古い CLAUDE.md に合わせて
編集してはならない。

discipline ゲート: `.claude/memory/` の直 main push の前に必ず
`cd svp_pipeline && python -m pytest tests/discipline/ -q` を全パスさせる
（例外は post-hoc 検出のみのため、違反は main を直接赤くする）。

## 設計ドキュメント索引

| ドキュメント | 内容 |
|---|---|
| [`docs/cli.md`](docs/cli.md) | `svp-video` の全ワークフロー例と CLI Options 全項目 |
| [`docs/architecture.md`](docs/architecture.md) | パイプライン段階構成と各補助フラグの詳細 |
| [`docs/backend_comparison.md`](docs/backend_comparison.md) | Gemini/OpenAI/split-composite の定性比較スナップショット |
| [`docs/measurement.md`](docs/measurement.md) | `probe-noise` A/A ノイズ床計測: 方法論、`noise_floor.json` スキーマ、実測 runbook、GRIP-1 への接続 |

## ドキュメント管理ポリシー

**CLAUDE.md はリポジトリ横断の普遍的内容のみ記述する (目標: 400 行以内)。**

新機能・新仕様を追加する際のドキュメント作成ルール:

1. **機能・仕様の詳細は `docs/<topic>.md` を新規作成して記述する**
   - 設計思想、計算式、パラメータ、検証結果、使用例など
   - CLAUDE.md に詳細を追加してはならない
2. **CLAUDE.md への追記は最小限に留める**
   - ファイル配置の一覧に 1 行
   - 設計ドキュメント索引表に 1 行（新 doc へのリンク）
   - それ以外の詳細は追加しない
3. **既存の task-specific 内容を見つけたら対応する `docs/` に移管する**
   - CLAUDE.md が肥大化していないか定期的に精査する

**判断基準**:
- **普遍的 (CLAUDE.md に残す)**: 開発環境、コーディング規約、git workflow、
  ファイル配置の一覧、ドキュメント索引 — どの作業者・どの機能でも参照する内容
- **task-specific (`docs/` に分離)**: 1 コンポーネントの実装詳細、1 指標の校正結果、
  1 機能の API スキーマ、1 実験の検証データ — 特定タスクの深掘り情報

## README 管理ポリシー

**README.md は入口情報に限定し、再膨張を防ぐ (目標: 300 行以内、hard limit: 350 行)。**

README の運用ルール:

1. **単一 section が 30 行を超えたら `docs/<topic>.md` へ抽出する**
   - README にはリンク + 2-3 行の要約のみ残す
2. **新規 docs を作成したら索引を 2 箇所更新する**
   - README の「設計ドキュメント」表に 1 行追加
   - CLAUDE.md の設計ドキュメント索引表に 1 行追加
3. **README と docs の責務を混ぜない**
   - README: 5 分で全体像を掴む入口情報、コンセプト図、クイックスタート、
     主要指標の一行定義、設計 docs への索引
   - docs: 仕様詳細、検証データ、1 コンポーネントの仕様詳細、
     トラブルシューティング事例、実装 recipe

## Coding Conventions

### Style

- ruff 準拠（line-length はプロジェクト設定に従う）
- 型ヒント必須: PEP 604/585 スタイル（`X | None`, `list[str]`, `dict[str, int]`）
  を使用（ruff UP と整合）
- `from __future__ import annotations` を全モジュール先頭に記述
- docstring / コメントは日本語 OK
- float 表示は小数点 3–4 桁に丸める

### Patterns

- **Frozen dataclass / pydantic model**: 値オブジェクトは不変で定義する
- **フォールバックチェーン**: import 時に try/except でフラグ設定、実行時に分岐
- **値のクランプ**: 正規化が必要な float 値は `max(lo, min(hi, value))` で範囲内に
  収める（画像/配列処理では `np.clip` / PIL 等の同等手段でよい）
- **タイムスタンプ**: UTC, ISO 8601 形式で保存

### Error Handling

- 明示的な例外送出は避け、フォールバックチェーンで吸収する
- オプショナル依存の import は `try/except ModuleNotFoundError` でモジュール名を
  確認してからフラグ設定（transitive 依存エラーは fail-fast）
- リソース（DB 接続・ファイル・ネットワーク）はコンテキストマネージャで管理する

### Testing

- テストファイル: `tests/test_*.py`
- `tmp_path` でファイルシステムを分離
- ドメインオブジェクトはヘルパーファクトリで生成。外部 API 境界
  （OpenAI/Gemini/fal/Drive）は `tests/fixtures/mock_*.py` の fake レスポンスで
  密閉する（実 API 呼び出し禁止）
- `pytest.approx()` で float 比較
- 規律自己検証テスト: `svp_pipeline/tests/discipline/`（CLAUDE.md 400 行 cap、
  README 350 行 cap、memory 構造検証）

## Git Workflow

### Branches

- `main` — 安定版。直接 push しない（例外: `.claude/memory/` の運用ログは直接 commit 可）
- `claude/*` — Claude Code が実装する作業ブランチ
- `codex/*` — Codex が実装する作業ブランチ

### Commit Messages

- Conventional Commits 形式: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- 日本語メッセージ可

### Pull Request

**変更は必ず Pull Request で実施する**。`main` への直接 push は禁止
（唯一の例外は `.claude/memory/` 運用ログ）。GitHub MCP の
`mcp__github__create_pull_request` で本文付き作成するか、リンク発行で
作成する（`gh pr create` は使わない）。

```bash
# 1. ブランチを push
git push -u origin <branch-name>

# 2. PR リンクを提示
# https://github.com/Yuu6798/svp-video-pipeline/compare/main...<branch-name>?expand=1
```

#### PR 本文の必須記述

PR を作成するときは、**本文を必ず作成する**（リンクのみ提示で本文を空にしない）。

本文に最低限含める要素:

```markdown
## Summary
<2–4 行で「何を / なぜ」変更したかを記述>

## Changes
- <主要な変更点を箇条書き、ファイル単位 or 機能単位>

## Verification
- [ ] `cd svp_pipeline && ruff check src tests` pass
- [ ] `cd svp_pipeline && pytest tests/ -q --tb=short` pass
- [ ] <該当する場合> 手動検証手順とその結果

## Related
- Phase: <STATUS.md ## Phase の該当箇所等>
- Brief / Issue: <該当する場合のリンク>

## Notes for Reviewer
<逸脱事項、未解決課題、次のループへの引き継ぎ等。なければ "None">
```

ドキュメント単独 PR の場合は `Verification` を「該当なし（docs のみ）」で省略可。
Codex が PR を作成する場合は [`AGENTS.md`](AGENTS.md) §2 の Completion Summary
フォーマットを本フォーマットの代わりに使ってよい（情報量は等価）。

## CI 基本方針

- Push / PR で lint（`ruff check src tests`）+ test（`pytest tests/ -q --tb=short`、
  `svp_pipeline/` 配下で実行）が通ることを必須とする
- CI 通過 = lint clean + 全テスト pass
- CI 固有のワークフロー詳細は `.github/workflows/*.yml` と `docs/` に記述する
