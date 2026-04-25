"""Tests for split character/background compositing helpers."""

from __future__ import annotations

from pathlib import Path

from svp_pipeline.generator.composite import _render_background_prompt
from svp_pipeline.schema import SVPVideo

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_background_prompt_does_not_force_cyberpunk_scene() -> None:
    svp = _load("still_life_macro.json")

    prompt = _render_background_prompt(svp)

    assert "早朝の静かな室内" in prompt
    assert "cyberpunk" not in prompt.lower()
    assert "magenta and cyan neon bokeh" not in prompt
    assert "Wet street reflections" not in prompt


def test_background_prompt_keeps_style_agnostic_quality_rules() -> None:
    svp = _load("still_life_macro.json")

    prompt = _render_background_prompt(svp)

    assert "No people" in prompt or "people" in prompt
    assert "No gritty texture noise" in prompt
    assert "No text, logo, or watermark" in prompt
