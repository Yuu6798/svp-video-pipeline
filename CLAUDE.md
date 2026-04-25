# CLAUDE.md — svp-video-pipeline

このファイルは Claude Code / Claude Agent SDK がこのリポジトリで作業する際の
普遍的な運用ポリシーをまとめる。リポジトリ固有の設計詳細は
`docs/<topic>.md` と各 `README.md` に分離する。

## Advisor Strategy（モデル運用方針）

- **メインエージェント**: Opus（設計判断・レビュー・品質ゲート）
- **サブエージェント**: Sonnet 固定（探索・実装・定型タスク）

Agent ツールで spawn する際は必ず `model: "sonnet"` を指定すること。

```python
# 正しい例
Agent({ model: "sonnet", subagent_type: "Explore", prompt: "..." })

# NG — model 省略すると Opus で動き、コスト効率が下がる
Agent({ subagent_type: "Explore", prompt: "..." })
```

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

**実行内容:**
- 会話の振り返りサマリーを `.claude/memory/YYYY-MM-DD.md` に保存
- `_index.md` に 1 行サマリーを追記
- CLAUDE.md への更新候補があればユーザーに提案

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
- 型ヒント必須: `Optional`, `List`, `Dict` を使用
- `from __future__ import annotations` を全モジュール先頭に記述
- docstring / コメントは日本語 OK
- float 表示は小数点 3–4 桁に丸める

### Patterns

- **Frozen dataclass / pydantic model**: 値オブジェクトは不変で定義する
- **フォールバックチェーン**: import 時に try/except でフラグ設定、実行時に分岐
- **値のクランプ**: 正規化が必要な float 値は `max(lo, min(hi, value))` で範囲内に収める
- **タイムスタンプ**: UTC, ISO 8601 形式で保存

### Error Handling

- 明示的な例外送出は避け、フォールバックチェーンで吸収する
- オプショナル依存の import は `try/except ModuleNotFoundError` でモジュール名を
  確認してからフラグ設定（transitive 依存エラーは fail-fast）
- リソース（DB 接続・ファイル・ネットワーク）はコンテキストマネージャで管理する

### Testing

- テストファイル: `tests/test_*.py`
- `tmp_path` でファイルシステムを分離
- ヘルパーファクトリでオブジェクト生成（モック不使用を推奨）
- `pytest.approx()` で float 比較

## Git Workflow

### Branches

- `main` — 安定版。直接 push しない（例外: `.claude/memory/` の運用ログは直接 commit 可）
- `claude/*` — Claude Code が実装する作業ブランチ
- `codex/*` — Codex が実装する作業ブランチ

### Commit Messages

- Conventional Commits 形式: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- 日本語メッセージ可

### Pull Request

**変更は必ず Pull Request で実施する**。`main` への直接 push は禁止。
PR はリンク発行で作成する（`gh pr create` は使わない）。

```bash
# 1. ブランチを push
git push -u origin <branch-name>

# 2. PR リンクを提示
# https://github.com/Yuu6798/svp-video-pipeline/compare/main...<branch-name>?expand=1
```

## CI 基本方針

- Push / PR で lint（`ruff check .`）+ test（`pytest -q --tb=short`）が通ることを必須とする
- CI 通過 = lint clean + 全テスト pass
- CI 固有のワークフロー詳細は `.github/workflows/*.yml` と `docs/` に記述する
