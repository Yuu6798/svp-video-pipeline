"""共通 UTC タイムスタンプヘルパー。

`log.json` の timestamp / run_dir 命名 / Drive アーカイブの `uploaded_at` を
UTC・ISO 8601 で統一するための単一の source of truth。CLAUDE.md の
「タイムスタンプ: UTC, ISO 8601 形式で保存」規約を実コードで enforce する。
"""

from __future__ import annotations

import datetime as dt


def utc_now_iso() -> str:
    """UTC・秒精度・ISO 8601（`Z` サフィックス）の現在時刻文字列を返す。"""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now_compact() -> str:
    """UTC の `%Y%m%d-%H%M%S` 形式（run_dir 命名用）の現在時刻文字列を返す。"""
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
