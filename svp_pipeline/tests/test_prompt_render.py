"""Tests for prompt rendering from SVP to image brief."""

from __future__ import annotations

import copy
from pathlib import Path

from svp_pipeline.schema import SVPVideo
from svp_pipeline.utils.prompt_render import render_image_prompt

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load(name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_render_includes_por_core() -> None:
    for name in ("shibuya_dusk.json", "still_life_macro.json", "action_ninja.json"):
        svp = _load(name)
        prompt = render_image_prompt(svp)
        for item in svp.por_core:
            assert item in prompt


def test_render_includes_grv_anchor() -> None:
    for name in ("shibuya_dusk.json", "still_life_macro.json", "action_ninja.json"):
        svp = _load(name)
        prompt = render_image_prompt(svp)
        for item in svp.grv_anchor:
            assert item in prompt


def test_render_excludes_motion_layer() -> None:
    svp = _load("shibuya_dusk.json")
    mutated = copy.deepcopy(svp.model_dump())
    mutated["motion_layer"]["subject_motion"] = [
        {
            "subject": "MOTION_UNIQUE_SUBJECT",
            "action": "MOTION_UNIQUE_ACTION",
            "intensity": "subtle",
        }
    ]
    mutated["motion_layer"]["temporal_anchors"] = [
        {"time_range": "0-1s", "description": "MOTION_UNIQUE_TEMPORAL_ANCHOR"}
    ]
    svp_with_unique_motion = SVPVideo.model_validate(mutated)

    prompt = render_image_prompt(svp_with_unique_motion)

    assert "subject_motion" not in prompt
    assert "temporal_anchors" not in prompt
    assert "MOTION_UNIQUE_SUBJECT" not in prompt
    assert "MOTION_UNIQUE_ACTION" not in prompt
    assert "MOTION_UNIQUE_TEMPORAL_ANCHOR" not in prompt
    assert svp.motion_layer.camera_movement.type not in prompt


def test_render_aggregates_forbidden() -> None:
    svp = _load("action_ninja.json")
    prompt = render_image_prompt(svp)

    expected = []
    expected.extend(svp.composition_layer.constraints.forbidden)
    expected.extend(svp.face_layer.constraints.forbidden)
    expected.extend(svp.style_layer.constraints.forbidden)
    expected.extend(svp.pose_layer.constraints.forbidden)
    expected.extend(svp.c3.constraints.forbidden)
    expected_unique = list(dict.fromkeys(expected))

    assert "## Avoid" in prompt
    assert "Avoid:" in prompt
    for item in expected_unique:
        assert item in prompt


def test_render_handles_no_subject() -> None:
    svp = _load("still_life_macro.json")
    prompt = render_image_prompt(svp)

    assert "No human subject (still life / landscape)." in prompt


def test_render_output_is_deterministic() -> None:
    svp = _load("shibuya_dusk.json")
    first = render_image_prompt(svp)
    second = render_image_prompt(svp)
    assert first == second


def test_render_for_all_three_samples() -> None:
    for name in ("shibuya_dusk.json", "still_life_macro.json", "action_ninja.json"):
        svp = _load(name)
        prompt = render_image_prompt(svp)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
