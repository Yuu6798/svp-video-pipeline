---
name: new-brief
description: Draft a Design Memo (Claude→Codex handoff) for the svp-video-pipeline repo, running the AGENTS.md §7 brief-drafting checklist as a pre-flight gate before emitting the AGENTS.md §1 Design Memo format. Use when the user asks to write/draft a new Design Memo, Task Brief, or implementation handoff, or to change an existing one.
---

# new-brief — Design Memo drafter with §7 pre-flight gate

Drafts a Design Memo in the `AGENTS.md §1` format, but only after running the
`AGENTS.md §7` brief-drafting checklist (蒸留した項目、レビュー往復を減らす
ためのチェックリスト). This skill is the **executor**; the policy sources of
truth are `AGENTS.md` §1 / §7 / §8. If they diverge from this file, they
win — fix this skill rather than acting on a stale copy.

Goal of the gate: front-load the checks that historically cause multi-round
review churn, so the memo lands in fewer rounds (review-round count is the
leading quality indicator — `AGENTS.md §8`).

## 0. Pre-flight reading

Before drafting, read:

- `AGENTS.md §1` (Design Memo format) + `§4` (escalation rules) + `§5`
  (branch rules) + `§7` (起草チェックリスト) + `§8` (経験外部化規律).
- `.claude/memory/STATUS.md` § Phase + § Next-Issue Queue for current
  priority, and `.claude/memory/_index.md` (直近 5 entries) + the 直近 3
  dated `YYYY-MM-DD.md` session logs. Skipping the memory log re-introduces
  the "過去 session の決定を知らずに再発明する" anti-pattern.
- The relevant `docs/` design doc for the area at hand (schema, generator
  backends, semantic layer) if one exists.

If a required doc is stale or missing, surface that in the draft rather than
inventing context (documented recurring failure mode).

## 1. Pre-flight checklist (AGENTS.md §7 — run before writing spec)

1. **STATUS.md の Next-Issue Queue を確認** — 既に同等のタスクがないか。
2. **現在の Phase / 直近マージ PR との関係を特定** — タスクが
   `STATUS.md ## Phase` のどの流れに属するか明確にする。
3. **前提となる設計判断を列挙** — 未決定事項があれば memo 内で選択肢を提示
   する (必要なら AskUserQuestion で先に user に確定させる)。
4. **Acceptance Criteria を検証可能な形で書く** — 「〜を改善する」ではなく
   「〜が X を返す」「`pytest tests/test_x.py` が pass する」。
5. **Scope IN/OUT を明示** — 変更してよいファイルと変更禁止のファイルを列挙。
6. **依存追加の有無を確認** — 新規依存が必要なら Allowed Dependencies に
   明記 (記載なし = escalation 対象)。
7. **タスク粒度が 0.5–2 日か確認** — 大きすぎる場合はフェーズ分割。
8. **レビュー回数の予測** — 0 回が理想。3 回以上かかりそうなら memo の仕様
   が不足している。
9. **SVP スキーマ / バックエンドインターフェース変更の入場試験を確認** —
   SVP スキーマ（`schema/svp.py` 五層）またはバックエンドインターフェース
   （`generator/image_base.py` 等）を変更する Brief は、
   `tests/fixtures/` の mock バックエンド（`mock_openai.py` /
   `mock_gemini.py` / `mock_fal.py` / `mock_drive.py` / `mock_responses.py`）
   と `tests/samples/` の SVP JSON（`shibuya_dusk.json` 等）への影響を
   memo 内に明記する。
10. **locked file と未検出フィールドを初手で縛る** — Scope OUT の「変更禁止
    ファイル」（特に共有スキーマ `schema/svp.py` 等）は edge case 対応でも
    破ってよくないと明記する。あわせて、実装対象外の値の扱い（素直に欠落
    させる / デフォルト値を置く / schema は触らない、のいずれか）を memo
    段階で確定する。未確定だと実装者が schema を安易に拡張して吸収し、
    レビューの連鎖指摘を誘発するリスクがある。

### 1a. Schema grounding  ⚠️ highest-yield

Every module path / CLI command / config key / model field you name in the
memo MUST be verified to exist in the implementation by grep, not from
memory. Canonical surfaces to grep:

- `svp_pipeline/src/svp_pipeline/` (module paths)
- `svp_pipeline/src/svp_pipeline/cli.py` (CLI subcommands + option names)
- `svp_pipeline/src/svp_pipeline/schema/svp.py` (SVP 五層スキーマのフィールド名)
- `svp_pipeline/src/svp_pipeline/generator/image_base.py` (バックエンド
  インターフェース — compile-pass / runtime-fail の定番混同源)
- `svp_pipeline/tests/fixtures/` / `svp_pipeline/tests/samples/` (mock
  バックエンド・SVP JSON サンプルの実態)

## 2. Emit the memo (AGENTS.md §1 format)

Use the Design Memo template verbatim from `AGENTS.md §1`:
`Phase / Goal / Acceptance Criteria / Implementation Approach / Risks /
Test Strategy / Scope / Schema Admission（該当時） / Allowed Dependencies /
Required Outputs / Done When`. Schema Admission is mandatory whenever the
brief adds/changes an SVP schema (`schema/svp.py`) or backend interface
(`generator/image_base.py` 等) field (checklist item 9).

Make every Acceptance Criterion **verifiable** (a command, a test, or a
grep-able assertion). Target task size ≈ 0.5–2 days. Branch name is
`codex/<topic>`; Done When requires `cd svp_pipeline && ruff check src
tests` + `pytest tests/ -q --tb=short` green and a Completion Summary PR
body (`AGENTS.md §2`).

**Output the entire memo inside a single fenced code block** so the user can
copy-paste it verbatim into Codex (use an outer ```` ```` ```` fence when the
memo itself contains ``` code fences). The memo body is the deliverable; keep
prose outside the block to a minimum.

## 3. Closeout

Hand the memo to the user (it is paste-ready for Codex). Note any §1a grep
that surfaced a schema mismatch, any unresolved design decision the user must
settle, and any 5+ round dispute that should be externalized into docs/tests
per `AGENTS.md §8`.
