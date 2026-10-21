"""依存 waiver の期限ゲート。

`[tool.uv.exclude-newer-package]` の waiver は、公開 7 日未満の修正版を lock に入れる一時措置。
残ると当該パッケージが 7 日ルールの外で黙って固定され続けるため、削除期限を
`[tool.dependency-waivers]` に併記させ、期限超過を CI 失敗にする。
"""

from __future__ import annotations

import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
FIX = (
    "pyproject の [tool.uv.exclude-newer-package] と [tool.dependency-waivers] "
    "から削除して `uv lock`"
)


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def waiver_problems(waivers: dict, deadlines: dict, today: date) -> list[str]:
    problems = [
        f"{pkg}: 削除期限が [tool.dependency-waivers] にない"
        for pkg in sorted(waivers.keys() - deadlines.keys())
    ]
    problems += [
        f"{pkg}: waiver が既にないのに期限だけ残っている（stale）"
        for pkg in sorted(deadlines.keys() - waivers.keys())
    ]
    problems += [
        f"{pkg}: 期限 {_as_date(deadlines[pkg])} を超過"
        for pkg in sorted(waivers.keys() & deadlines.keys())
        if _as_date(deadlines[pkg]) < today
    ]
    return problems


def test_dependency_waivers_have_deadlines_and_are_not_expired() -> None:
    tool = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    waivers = tool.get("uv", {}).get("exclude-newer-package", {})
    deadlines = tool.get("dependency-waivers", {})
    problems = waiver_problems(waivers, deadlines, datetime.now(UTC).date())
    assert not problems, "依存 waiver の違反:\n" + "\n".join(problems) + f"\n対処: {FIX}"


def test_waiver_problems_reports_missing_stale_and_expired() -> None:
    cutoff = "2026-10-17T00:00:00Z"
    waivers = {"nodeadline": cutoff, "late": cutoff, "ok": cutoff, "ok2": cutoff}
    deadlines = {
        "late": date(2026, 10, 31),
        "ghost": "2026-12-31",
        "ok": "2026-11-01",
        "ok2": datetime(2026, 11, 1),
    }
    problems = waiver_problems(waivers, deadlines, date(2026, 11, 1))
    assert len(problems) == 3
    assert "nodeadline" in problems[0] and "ghost" in problems[1] and "late" in problems[2]
