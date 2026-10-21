"""uv の exclude-newer-package waiver が期限切れのまま残っていないことを検証する。

`[tool.uv] exclude-newer = "7 days"` は全依存に公開 7 日の待機窓を課す。個別の
`[tool.uv.exclude-newer-package]` waiver は絶対日時なので、この窓を超えて残ると
当該パッケージを黙って固定し続ける。期限切れ（waiver 時刻から 14 日超過）を
CI 失敗として可視化し、削除して `uv lock` を実行させる。
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime, timedelta

from ._helpers import REPO_ROOT

PYPROJECT = REPO_ROOT / "svp_pipeline" / "pyproject.toml"
WAIVER_MAX_AGE = timedelta(days=14)


def _expired(waivers: dict[str, object], now: datetime) -> list[str]:
    expired = []
    for pkg, value in waivers.items():
        stamp = datetime.fromisoformat(str(value))
        assert stamp.tzinfo is not None, f"{pkg} の waiver 時刻 {value!r} に UTC offset がない"
        if now - stamp.astimezone(UTC) > WAIVER_MAX_AGE:
            expired.append(pkg)
    return expired


def test_uv_waivers_not_expired():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    waivers = config.get("tool", {}).get("uv", {}).get("exclude-newer-package", {})
    expired = _expired(waivers, datetime.now(UTC))
    assert not expired, (
        f"uv の exclude-newer-package waiver が 14 日を超過: {', '.join(expired)}。\n"
        "該当エントリを pyproject.toml から削除して `uv lock` を実行すること"
        "（公開 7 日以上の修正版は 7 日ルールで解決される）。"
    )


def test_expired_waiver_is_detected():
    now = datetime(2026, 11, 1, tzinfo=UTC)
    waivers = {"old": "2026-10-17T00:00:00Z", "fresh": "2026-10-25T00:00:00Z"}
    assert _expired(waivers, now) == ["old"]
