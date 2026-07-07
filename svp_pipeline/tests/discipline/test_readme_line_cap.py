"""README.md 群が行数キャップ以内に収まっていることを検証する。

README は「5 分で全体像を掴む入口情報」の役割に限定する（CLAUDE.md §README
管理ポリシー）。目標 300 行・hard limit 350 行のうち、本テストは
**hard limit 350 行**を CI で enforce する。ルート README.md と
svp_pipeline/README.md の双方が対象（ネスト構成のため 2 ファイル）。
超過した場合は該当 section を `docs/<topic>.md` へ抽出し、README には
リンク + 2–3 行の要約だけを残すこと。
"""

from __future__ import annotations

import pytest

from ._helpers import REPO_ROOT

README_LINE_CAP = 350

README_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "svp_pipeline" / "README.md",
]


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _assert_line_cap(text: str, *, source: str, cap: int) -> None:
    count = _line_count(text)
    assert count <= cap, (
        f"{source} が {count} 行で {cap} 行 hard limit を超過。\n"
        "README は入口情報のみ: 30 行を超える section は docs/<topic>.md へ抽出し、"
        "README にはリンク + 2–3 行の要約だけを残すこと。"
    )


@pytest.mark.parametrize("readme_path", README_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_readme_within_line_cap(readme_path):
    assert readme_path.exists(), f"{readme_path} が存在しない"
    _assert_line_cap(
        readme_path.read_text(encoding="utf-8"),
        source=str(readme_path.relative_to(REPO_ROOT)),
        cap=README_LINE_CAP,
    )


def test_line_cap_detects_overflow():
    overflow = "\n".join(f"line {i}" for i in range(README_LINE_CAP + 1))
    assert _line_count(overflow) == README_LINE_CAP + 1
    with pytest.raises(AssertionError, match="hard limit を超過"):
        _assert_line_cap(overflow, source="fixture", cap=README_LINE_CAP)
