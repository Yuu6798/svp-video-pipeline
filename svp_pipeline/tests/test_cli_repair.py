"""Tests for svp-video CLI (--repair-from-rpe behavior)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import svp_pipeline.cli as cli_mod
from svp_pipeline.cli import app
from tests.fixtures.fakes import FakePipeline
from tests.fixtures.helpers import load_sample as _load

runner = CliRunner()

# Rich/Typer の help 出力には端末判定によって ANSI 制御コードが挿入される
# ことがあり、`--duration` のようなトークンが raw output 上で連続文字列として
# 存在しなくなる。部分文字列マッチを行う前にこの正規表現で剥がして、
# 環境（ローカル/CI/TERM=dumb 等）依存性を排除する。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _set_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-test")


def _clear_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda: None)
    for key in (
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "FAL_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline.instances = []
    FakePipeline.error = None
    monkeypatch.setattr(cli_mod, "Pipeline", FakePipeline)


def test_repair_from_rpe_writes_proposal_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps(
            {
                "missing": ["single character remains centered"],
                "violations": ["sword-like background reflection"],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Semantic repair proposal written" in result.output
    repair_dirs = list(tmp_path.glob("repair-*"))
    assert len(repair_dirs) == 1
    assert (repair_dirs[0] / "semantic_diff.json").exists()
    assert (repair_dirs[0] / "repair_proposal.json").exists()
    assert (repair_dirs[0] / "target_svp.proposed.json").exists()

def test_repair_from_rpe_ignores_bad_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_PLANNER_MODEL", "not-a-planner")
    monkeypatch.setenv("DEFAULT_IMAGE_BACKEND", "not-an-image-backend")
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text(
        json.dumps({"violations": ["sword-like background reflection"]}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Semantic repair proposal written" in result.output

def test_repair_from_rpe_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    observed_rpe = tmp_path / "observed_rpe.json"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    observed_rpe.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--repair-from-rpe",
            str(observed_rpe),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output

def test_repair_from_rpe_requires_svp_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    observed_rpe = tmp_path / "observed_rpe.json"
    observed_rpe.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["--repair-from-rpe", str(observed_rpe), "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "requires --from-svp or --reuse-run" in result.output
