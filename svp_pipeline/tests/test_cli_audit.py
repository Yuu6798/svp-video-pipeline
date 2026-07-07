"""Tests for svp-video CLI (--audit-*/--compare-image behavior)."""

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
from tests.fixtures.helpers import write_png as _write_png
from tests.fixtures.mock_gemini import TINY_PNG_BYTES

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


def test_audit_image_writes_observed_rpe_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-observed",
            "silver ponytail is visible",
            "--audit-missing",
            "red eyes are weak",
            "--audit-violation",
            "sword-like background reflection",
            "--audit-object-state",
            "scabbard: separate dark sheath at upper-left",
            "--audit-contact-graph",
            "left_hand -> scabbard",
            "--audit-viewer-contact-graph",
            "viewer_left_hand -> katana_handle",
            "--audit-anatomical-contact-graph",
            "character_right_hand -> katana_handle",
            "--audit-pose-intent",
            "unsheathing / draw-pose",
            "--audit-failure-mode",
            "left hand grips the wrong object",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Image RPE audit written" in result.output
    audit_dirs = list(tmp_path.glob("audit-*"))
    assert len(audit_dirs) == 1
    observed = json.loads((audit_dirs[0] / "observed_rpe.json").read_text(encoding="utf-8"))
    assert observed["missing"] == ["red eyes are weak"]
    assert observed["violations"] == ["sword-like background reflection"]
    assert observed["state"]["object_graph"] == ["scabbard: separate dark sheath at upper-left"]
    assert observed["state"]["contact_graph"] == ["left_hand -> scabbard"]
    assert observed["state"]["viewer_contact_graph"] == ["viewer_left_hand -> katana_handle"]
    assert observed["state"]["anatomical_contact_graph"] == [
        "character_right_hand -> katana_handle"
    ]
    assert observed["state"]["pose_intent"] == "unsheathing / draw-pose"
    assert observed["state"]["failure_modes"] == ["left hand grips the wrong object"]
    assert "expected_identity" in observed["state"]

def test_audit_image_ignores_bad_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("DEFAULT_PLANNER_MODEL", "not-a-planner")
    monkeypatch.setenv("DEFAULT_IMAGE_BACKEND", "not-an-image-backend")
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "sword-like background reflection",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Image RPE audit written" in result.output

def test_audit_image_with_repair_writes_proposed_svp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "sword-like background reflection",
            "--audit-repair",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Proposed SVP" in result.output
    assert list(tmp_path.glob("audit-*/repair-*/target_svp.proposed.json"))

def test_audit_image_with_repair_extracts_failure_preset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-violation",
            "katana reflection appears as a second sword in the wet background",
            "--audit-repair",
            "--extract-failure-preset",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Failure preset candidate" in result.output
    candidates = list(tmp_path.glob("audit-*/repair-*/failure_preset.candidate.json"))
    assert len(candidates) == 1
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "SVP.failure_preset.v1"
    assert payload["failure_taxonomy"]

def test_extract_failure_preset_requires_audit_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--extract-failure-preset",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--extract-failure-preset requires --audit-repair" in result.output

def test_audit_image_with_compare_writes_visual_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(source)
    _write_png(candidate)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(source),
            "--compare-image",
            str(candidate),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Visual comparison" in result.output
    comparison = list(tmp_path.glob("audit-*/visual_comparison.json"))
    assert len(comparison) == 1
    payload = json.loads(comparison[0].read_text(encoding="utf-8"))
    assert payload["dimensions_equal"] is True
    assert payload["pixel_rms"] == 0.0

def test_audit_regenerate_requires_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-regenerate",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--audit-regenerate requires --audit-repair" in result.output

def test_audit_repair_regenerate_writes_pipeline_output_and_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(source, color=(10, 10, 10))
    _write_png(target, color=(20, 20, 20))

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(source),
            "--compare-image",
            str(target),
            "--audit-missing",
            "glowing lavender eyes from the target image",
            "--audit-repair",
            "--audit-regenerate",
            "--no-video",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Regenerated image" in result.output
    assert "Regenerated visual comparison" in result.output
    audit_dirs = list(tmp_path.glob("audit-*"))
    assert len(audit_dirs) == 1
    proposed_svp = next(audit_dirs[0].glob("repair-*/target_svp.proposed.json"))
    assert FakePipeline.instances[-1].run_calls[0]["from_svp_path"] == proposed_svp
    assert list(audit_dirs[0].glob("regenerated/20260425-000000/image.png"))
    comparison = audit_dirs[0] / "regenerated_comparison" / "visual_comparison.json"
    assert comparison.exists()
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert payload["source_image"]["path"].endswith("source_image.png")
    assert payload["candidate_image"]["path"].endswith("candidate_image.png")

def test_openai_audit_requires_openai_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    _write_png(image)

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--audit-backend",
            "openai",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is required" in result.output

def test_audit_image_rejects_unreadable_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_required_keys(monkeypatch)
    svp_path = tmp_path / "source.svp.json"
    image = tmp_path / "image.png"
    svp_path.write_text(_load("shibuya_dusk.json").model_dump_json(), encoding="utf-8")
    image.write_text("not an image", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "--from-svp",
            str(svp_path),
            "--audit-image",
            str(image),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "not a readable image" in result.output

def test_audit_flags_require_audit_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)

    result = runner.invoke(
        app,
        ["prompt", "--audit-contact-graph", "left_hand -> scabbard", "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Audit options require --audit-image" in result.output

def test_compare_image_requires_audit_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_required_keys(monkeypatch)
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(TINY_PNG_BYTES)

    result = runner.invoke(
        app,
        ["prompt", "--compare-image", str(candidate), "--output", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Audit options require --audit-image" in result.output
