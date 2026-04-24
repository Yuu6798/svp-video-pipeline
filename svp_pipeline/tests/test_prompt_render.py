"""Tests for prompt rendering from SVP to image brief."""

from __future__ import annotations

import copy
from pathlib import Path

from svp_pipeline.schema import SVPVideo
from svp_pipeline.utils.prompt_render import render_image_prompt, render_motion_prompt

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


def test_render_includes_required_constraints() -> None:
    svp = _load("shibuya_dusk.json")
    payload = copy.deepcopy(svp.model_dump())

    payload["composition_layer"]["constraints"]["required"] = ["REQ_COMPOSITION_UNIQUE"]
    payload["face_layer"]["constraints"]["required"] = ["REQ_FACE_UNIQUE"]
    payload["style_layer"]["constraints"]["required"] = ["REQ_STYLE_UNIQUE"]
    payload["pose_layer"]["constraints"]["required"] = ["REQ_POSE_UNIQUE"]
    payload["c3"]["constraints"]["required"] = ["REQ_GLOBAL_UNIQUE"]
    payload["motion_layer"]["constraints"]["required"] = ["REQ_MOTION_UNIQUE_SHOULD_NOT_RENDER"]

    mutated = SVPVideo.model_validate(payload)
    prompt = render_image_prompt(mutated)

    assert "## Required Constraints" in prompt
    assert "REQ_COMPOSITION_UNIQUE" in prompt
    assert "REQ_FACE_UNIQUE" in prompt
    assert "REQ_STYLE_UNIQUE" in prompt
    assert "REQ_POSE_UNIQUE" in prompt
    assert "REQ_GLOBAL_UNIQUE" in prompt
    assert "REQ_MOTION_UNIQUE_SHOULD_NOT_RENDER" not in prompt


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


def test_render_motion_prompt_includes_image_reference() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)
    assert "@Image1" in prompt


def test_render_motion_prompt_includes_por_core() -> None:
    svp = _load("action_ninja.json")
    prompt = render_motion_prompt(svp)
    for item in svp.por_core:
        assert item in prompt


def test_render_motion_prompt_includes_grv_anchor() -> None:
    svp = _load("still_life_macro.json")
    prompt = render_motion_prompt(svp)
    for item in svp.grv_anchor:
        assert item in prompt


def test_render_motion_prompt_includes_duration() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)
    assert str(svp.motion_layer.duration_seconds) in prompt


def test_render_motion_prompt_includes_camera_movement() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)
    assert svp.motion_layer.camera_movement.type in prompt
    assert svp.motion_layer.camera_movement.speed in prompt


def test_render_motion_prompt_includes_temporal_anchors() -> None:
    svp = _load("action_ninja.json")
    prompt = render_motion_prompt(svp)
    for anchor in svp.motion_layer.temporal_anchors:
        assert anchor.time_range in prompt
        assert anchor.description in prompt


def test_render_motion_prompt_aggregates_forbidden() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)
    for item in svp.motion_layer.constraints.forbidden:
        assert item in prompt
    for item in svp.c3.constraints.motion_forbidden:
        assert item in prompt


def test_render_motion_prompt_includes_required_constraints() -> None:
    svp = _load("shibuya_dusk.json")
    payload = copy.deepcopy(svp.model_dump())
    payload["motion_layer"]["constraints"]["required"] = ["REQ_MOTION_UNIQUE"]
    payload["c3"]["constraints"]["required"] = ["REQ_GLOBAL_UNIQUE"]
    mutated = SVPVideo.model_validate(payload)

    prompt = render_motion_prompt(mutated)
    assert "## Required Constraints" in prompt
    assert "[motion] REQ_MOTION_UNIQUE" in prompt
    assert "[global] REQ_GLOBAL_UNIQUE" in prompt


def test_render_motion_prompt_keeps_continuity_guardrails_positive() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)

    assert "## Continuity Guardrails" in prompt
    assert "Keep PoR core elements visible throughout the timeline." in prompt
    assert "Keep primary grv anchors inside the frame throughout the timeline." in prompt
    assert "Avoid: Keep PoR core elements visible throughout the timeline." not in prompt
    assert "Avoid: Keep primary grv anchors inside the frame throughout the timeline." not in prompt


def test_render_motion_prompt_excludes_image_only_layers() -> None:
    svp = _load("shibuya_dusk.json")
    prompt = render_motion_prompt(svp)
    for feature in svp.face_layer.distinctive_features:
        assert feature not in prompt
    assert svp.style_family in prompt


def test_render_motion_prompt_for_all_samples() -> None:
    for name in ("shibuya_dusk.json", "still_life_macro.json", "action_ninja.json"):
        svp = _load(name)
        prompt = render_motion_prompt(svp)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_render_motion_prompt_output_is_deterministic() -> None:
    svp = _load("shibuya_dusk.json")
    first = render_motion_prompt(svp)
    second = render_motion_prompt(svp)
    assert first == second
