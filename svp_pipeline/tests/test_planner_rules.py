"""Unit tests for the planner rule engine (PLANNER-1).

These tests exercise ``apply_scenario`` / ``LayerRule`` / ``ScenarioContext``
directly with synthetic rule tables, independent of the planner's real
scenario data (that behavioral-equivalence coverage lives in
``test_planner.py``, which is intentionally left untouched by this change).
"""

from __future__ import annotations

import pytest

from svp_pipeline.generator.planner_rules import LayerRule, ScenarioContext, apply_scenario
from svp_pipeline.schema import SVPVideo


def _minimal_svp_dict() -> dict:
    """Smallest dict that validates as an SVPVideo, for engine-level tests."""
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
            "constraints": {"required": [], "forbidden": [], "motion_forbidden": []},
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


def _minimal_svp() -> SVPVideo:
    return SVPVideo.model_validate(_minimal_svp_dict())


def test_list_append_adds_items_to_nested_layer_field() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="face_layer",
            list_appends={"constraints.required": ("trait a", "trait b")},
        ),
    )

    updated = apply_scenario(svp, rules, ScenarioContext())

    assert updated.face_layer.constraints.required == ["trait a", "trait b"]
    # Original object is untouched (pydantic model_copy, not mutation).
    assert svp.face_layer.constraints.required == []


def test_list_append_dedupes_case_insensitively_and_preserves_order() -> None:
    svp = _minimal_svp()
    svp = svp.model_copy(
        update={
            "face_layer": svp.face_layer.model_copy(
                update={
                    "constraints": svp.face_layer.constraints.model_copy(
                        update={"required": ["Existing Trait"]}
                    )
                }
            )
        }
    )
    rules = (
        LayerRule(
            layer="face_layer",
            # "existing trait" differs only in case; "new trait" is genuinely new.
            list_appends={"constraints.required": ("existing trait", "new trait")},
        ),
    )

    updated = apply_scenario(svp, rules, ScenarioContext())

    assert updated.face_layer.constraints.required == ["Existing Trait", "new trait"]


def test_list_append_supports_ctx_driven_callable() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="c3",
            list_appends={"evaluation_criteria.hit_list": lambda ctx: list(ctx.locks)},
        ),
    )
    ctx = ScenarioContext(locks=("lock-a", "lock-b"))

    updated = apply_scenario(svp, rules, ctx)

    assert updated.c3.evaluation_criteria.hit_list == ["lock-a", "lock-b"]


def test_scalar_set_overwrites_non_list_field() -> None:
    svp = _minimal_svp()
    assert svp.pose_layer.hand_state == "free"
    rules = (LayerRule(layer="pose_layer", sets={"hand_state": "gripping a prop"}),)

    updated = apply_scenario(svp, rules, ScenarioContext())

    assert updated.pose_layer.hand_state == "gripping a prop"


def test_scalar_set_supports_ctx_driven_callable() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="role_visual_cue",
            sets={"role": lambda ctx: ctx.role_visual_cue_role or "character"},
        ),
    )

    updated = apply_scenario(svp, rules, ScenarioContext(role_visual_cue_role=None))

    assert updated.role_visual_cue.role == "character"


def test_when_condition_skips_rule_when_false() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="pose_layer",
            when=lambda ctx: ctx.character_weapon_contact,
            sets={"hand_state": "should not apply"},
        ),
    )

    updated = apply_scenario(svp, rules, ScenarioContext(character_weapon_contact=False))

    assert updated.pose_layer.hand_state == "free"


def test_when_condition_applies_rule_when_true() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="pose_layer",
            when=lambda ctx: ctx.character_weapon_contact,
            sets={"hand_state": "gripping weapon"},
        ),
    )

    updated = apply_scenario(svp, rules, ScenarioContext(character_weapon_contact=True))

    assert updated.pose_layer.hand_state == "gripping weapon"


def test_rules_apply_in_sequence_across_multiple_layers() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(layer="pose_layer", sets={"hand_state": "step one"}),
        LayerRule(
            layer="face_layer",
            list_appends={"constraints.required": ("step two",)},
        ),
        LayerRule(layer="pose_layer", sets={"body_pose": "step three"}),
    )

    updated = apply_scenario(svp, rules, ScenarioContext())

    assert updated.pose_layer.hand_state == "step one"
    assert updated.face_layer.constraints.required == ["step two"]
    assert updated.pose_layer.body_pose == "step three"


def test_invalid_layer_fails_fast() -> None:
    svp = _minimal_svp()
    rules = (LayerRule(layer="not_a_real_layer", sets={"x": "y"}),)

    with pytest.raises(AttributeError):
        apply_scenario(svp, rules, ScenarioContext())


def test_invalid_nested_subpath_fails_fast() -> None:
    svp = _minimal_svp()
    rules = (
        LayerRule(
            layer="face_layer",
            list_appends={"constraints.not_a_real_field": ("x",)},
        ),
    )

    with pytest.raises(AttributeError):
        apply_scenario(svp, rules, ScenarioContext())


def test_new_scenario_can_be_added_as_pure_rule_data() -> None:
    """A brand-new scenario is just a new tuple of LayerRule.

    No change to apply_scenario / LayerRule / ScenarioContext is needed to add
    scenario behavior -- this is the property PLANNER-1 relies on to keep
    planner.py's 4 real scenarios declarative.
    """
    svp = _minimal_svp()
    hypothetical_new_scenario = (
        LayerRule(
            layer="style_layer",
            list_appends={"constraints.forbidden": ("glitter overlay", "lens flare")},
        ),
        LayerRule(
            layer="c3",
            when=lambda ctx: "spooky" in ctx.risk_flags,
            list_appends={
                "evaluation_criteria.critical_fail_conditions": (
                    "mood reads as comedic rather than spooky",
                ),
            },
        ),
    )

    updated = apply_scenario(
        svp,
        hypothetical_new_scenario,
        ScenarioContext(risk_flags=frozenset({"spooky"})),
    )

    assert "glitter overlay" in updated.style_layer.constraints.forbidden
    assert "lens flare" in updated.style_layer.constraints.forbidden
    assert (
        "mood reads as comedic rather than spooky"
        in updated.c3.evaluation_criteria.critical_fail_conditions
    )
