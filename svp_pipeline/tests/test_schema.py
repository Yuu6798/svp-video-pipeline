"""Tests for the SVP.v4x-five-layer.video Pydantic schema."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from svp_pipeline.schema import SVPVideo

# --------------------------------------------------------------------------- #
# Sample-based validation
# --------------------------------------------------------------------------- #


def _load(samples_dir: Path, name: str) -> SVPVideo:
    return SVPVideo.model_validate_json((samples_dir / name).read_text(encoding="utf-8"))


def test_sample_shibuya_dusk_validates(samples_dir: Path) -> None:
    svp = _load(samples_dir, "shibuya_dusk.json")
    assert svp.motion_layer.duration_seconds == 5
    assert svp.composition_layer.aspect_ratio == "16:9"
    assert "PoR_core要素のフレームアウト" in svp.motion_layer.constraints.forbidden


def test_sample_still_life_macro_validates(samples_dir: Path) -> None:
    svp = _load(samples_dir, "still_life_macro.json")
    assert svp.motion_layer.duration_seconds == 4
    assert svp.style_layer.entropy == "low"
    assert svp.face_layer.distinctive_features == []


def test_sample_action_ninja_validates(samples_dir: Path) -> None:
    svp = _load(samples_dir, "action_ninja.json")
    assert svp.motion_layer.duration_seconds == 6
    assert svp.composition_layer.aspect_ratio == "21:9"
    assert svp.motion_layer.camera_movement.type == "handheld"
    # All samples must explicitly forbid PoR / grv_anchor from leaving the frame.
    forbidden = svp.motion_layer.constraints.forbidden
    assert "PoR_core要素のフレームアウト" in forbidden
    assert "grv_anchor主要要素の画面外移動" in forbidden


# --------------------------------------------------------------------------- #
# Minimal valid SVP builder (used by boundary / failure tests)
# --------------------------------------------------------------------------- #


def _minimal_svp_dict() -> dict:
    """Return the smallest dict that is a valid SVPVideo."""
    return {
        "schema_version": "SVP.v4x-five-layer.video",
        "por_identity": "最小構成の検証用プロンプト。",
        "por_core": ["A", "B", "C"],
        "grv_anchor": ["A", "B"],
        "de_profile": {"target_mean": 0.05, "tolerance": 0.03},
        "composition_layer": {
            "camera_angle": "eye_level",
            "framing": "medium_shot",
            "aspect_ratio": "16:9",
        },
        "face_layer": {
            "expression": "neutral",
            "eye_direction": "forward",
        },
        "style_layer": {
            "line_density": "medium",
            "specular_reflect": "medium",
            "glow_radius": "narrow",
            "entropy": "low",
        },
        "pose_layer": {
            "body_pose": "standing",
            "hand_state": "free",
        },
        "motion_layer": {
            "duration_seconds": 5,
            "camera_movement": {"type": "static"},
        },
        "style_family": "TEST_STYLE_v1",
        "color_axis": ["neutral"],
        "texture_axis": ["plain"],
        "c3": {
            "context": "test context",
            "constraints": {
                "required": [],
                "forbidden": [],
                "motion_forbidden": [],
            },
            "consistency": [],
        },
        "axes": {
            "composition": "center",
            "light_air": "soft",
            "expression": "neutral",
            "stroke": "medium",
            "motion": "static",
            "material": "generic",
            "narrative": "none",
            "emotion_symbol": "none",
        },
    }


def test_minimal_valid_svp() -> None:
    svp = SVPVideo.model_validate(_minimal_svp_dict())
    assert svp.schema_version == "SVP.v4x-five-layer.video"
    assert svp.motion_layer.duration_seconds == 5


# --------------------------------------------------------------------------- #
# Failure cases
# --------------------------------------------------------------------------- #


def test_por_core_too_few() -> None:
    data = _minimal_svp_dict()
    data["por_core"] = ["only", "two"]
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_por_core_too_many() -> None:
    data = _minimal_svp_dict()
    data["por_core"] = ["a", "b", "c", "d", "e", "f", "g"]
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_grv_anchor_too_few() -> None:
    data = _minimal_svp_dict()
    data["grv_anchor"] = ["only_one"]
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_grv_anchor_too_many() -> None:
    data = _minimal_svp_dict()
    data["grv_anchor"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_duration_below_minimum() -> None:
    data = _minimal_svp_dict()
    data["motion_layer"]["duration_seconds"] = 3
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_duration_above_maximum() -> None:
    data = _minimal_svp_dict()
    data["motion_layer"]["duration_seconds"] = 16
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_invalid_aspect_ratio() -> None:
    data = _minimal_svp_dict()
    data["composition_layer"]["aspect_ratio"] = "5:4"
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_invalid_camera_angle() -> None:
    data = _minimal_svp_dict()
    data["composition_layer"]["camera_angle"] = "weird_angle"
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_invalid_camera_movement_type() -> None:
    data = _minimal_svp_dict()
    data["motion_layer"]["camera_movement"]["type"] = "spiral"
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_schema_version_immutable() -> None:
    data = _minimal_svp_dict()
    data["schema_version"] = "SVP.v4x-other"
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_extra_field_forbidden() -> None:
    data = _minimal_svp_dict()
    data["unexpected_key"] = "nope"
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


def test_missing_required_layer() -> None:
    data = _minimal_svp_dict()
    del data["motion_layer"]
    with pytest.raises(ValidationError):
        SVPVideo.model_validate(data)


# --------------------------------------------------------------------------- #
# Boundary cases
# --------------------------------------------------------------------------- #


def test_duration_lower_boundary() -> None:
    data = _minimal_svp_dict()
    data["motion_layer"]["duration_seconds"] = 4
    svp = SVPVideo.model_validate(data)
    assert svp.motion_layer.duration_seconds == 4


def test_duration_upper_boundary() -> None:
    data = _minimal_svp_dict()
    data["motion_layer"]["duration_seconds"] = 15
    svp = SVPVideo.model_validate(data)
    assert svp.motion_layer.duration_seconds == 15


def test_por_core_lower_boundary() -> None:
    data = _minimal_svp_dict()
    data["por_core"] = ["a", "b", "c"]
    svp = SVPVideo.model_validate(data)
    assert len(svp.por_core) == 3


def test_por_core_upper_boundary() -> None:
    data = _minimal_svp_dict()
    data["por_core"] = ["a", "b", "c", "d", "e", "f"]
    svp = SVPVideo.model_validate(data)
    assert len(svp.por_core) == 6


# --------------------------------------------------------------------------- #
# Meta: schema export for downstream (M2) tool_use binding
# --------------------------------------------------------------------------- #


def test_model_json_schema_is_exportable() -> None:
    schema = SVPVideo.model_json_schema()
    assert schema["title"] == "SVPVideo"
    # Spot-check that nested layer definitions are inlined or referenced.
    assert "properties" in schema
    assert "motion_layer" in schema["properties"]


def test_minimal_dict_builder_is_independent() -> None:
    """Sanity: the helper must return a fresh dict each call so mutation in one test cannot leak."""
    a = _minimal_svp_dict()
    b = _minimal_svp_dict()
    a["por_core"] = ["mutated"]
    assert b["por_core"] == ["A", "B", "C"]
    # And copy.deepcopy of the baseline is equivalent.
    assert copy.deepcopy(b) == _minimal_svp_dict()
