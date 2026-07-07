"""Tests for planner (M2) — background-noise / wet-surface heuristics."""

from __future__ import annotations

from svp_pipeline.generator import Planner
from tests.fixtures.fakes import DummyClient
from tests.fixtures.mock_responses import VALID_SHIBUYA_RESPONSE, VALID_STILL_LIFE_RESPONSE


def test_planner_adds_background_noise_controls_for_high_risk_prompt() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "cyberpunk rainy neon city, single young adult woman with silver-gray "
        "high ponytail, vivid red eyes, transparent umbrella, katana at her waist, "
        "wet reflections"
    )

    assert "background acts as smooth lighting support, not the subject" in (
        svp.composition_layer.constraints.required
    )
    assert "background: sparse neon blocks instead of dense signage" in (
        svp.composition_layer.depth_layers
    )
    assert "midground: broad smooth wet reflection bands" in (
        svp.composition_layer.depth_layers
    )
    assert "dense signage" in svp.composition_layer.constraints.forbidden
    assert "speckled light noise" in svp.style_layer.constraints.forbidden
    assert "background simplicity has higher priority than background detail" in (
        svp.c3.constraints.required
    )
    assert "main weapon is a single physical object" in (
        svp.pose_layer.constraints.required
    )
    assert "no weapon-like reflections in the background" in (
        svp.reference_usage_policy.object_instance_rules
    )
    assert "broad smooth wet reflection bands" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_background_risk_respects_negated_weapon_terms() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "cyberpunk rainy neon city, single young adult woman with silver-gray "
        "high ponytail, vivid red eyes, transparent umbrella, no katana, no sword, "
        "no blade, no weapon"
    )

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "weapon-like reflections in the background" not in (
        svp.composition_layer.constraints.forbidden
    )

def test_planner_background_risk_respects_negated_reflection_terms() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana sheathed at her waist, simple dark "
        "indoor background, no rain, no umbrella, no neon signs, no wet reflections, "
        "no katana reflection, no sharp linear floor reflections"
    )

    assert "midground: broad smooth wet reflection bands" not in (
        svp.composition_layer.depth_layers
    )
    assert "broad smooth wet reflection bands" not in (
        svp.reference_usage_policy.background_quality_rules
    )
    assert "fragmented noisy wet reflections" not in (
        svp.composition_layer.constraints.forbidden
    )

def test_planner_detects_wet_surface_predicate_word_order() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman in a neon city where the pavement is wet"
    )

    assert "midground: broad smooth wet reflection bands" in (
        svp.composition_layer.depth_layers
    )
    assert "broad smooth wet reflection bands" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_detects_plural_wet_surface_phrasing() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman in a neon city where wet streets and pavements "
        "are wet"
    )

    assert "midground: broad smooth wet reflection bands" in (
        svp.composition_layer.depth_layers
    )
    assert "broad smooth wet reflection bands" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_detects_past_tense_wet_surface_predicate() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman in a neon city where the street was wet and "
        "the pavements were wet"
    )

    assert "midground: broad smooth wet reflection bands" in (
        svp.composition_layer.depth_layers
    )
    assert "broad smooth wet reflection bands" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_detects_sidewalk_wet_surface_predicate() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("single young adult woman in a neon city where the sidewalk is wet")

    assert "midground: broad smooth wet reflection bands" in (
        svp.composition_layer.depth_layers
    )
    assert "broad smooth wet reflection bands" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_background_depth_does_not_force_single_character_for_group() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("two characters in a rainy neon city with wet reflections")

    assert "foreground: single character in sharp detail" not in (
        svp.composition_layer.depth_layers
    )
    assert "foreground: requested subject(s) in sharp detail" in (
        svp.composition_layer.depth_layers
    )

def test_planner_does_not_inject_neon_style_for_weapon_only_risk() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("single young adult woman with a katana in a quiet forest")

    assert "neon atmosphere is carried by large clean light blocks" not in (
        svp.style_layer.constraints.required
    )
    assert "main weapon is a single physical object" in (
        svp.pose_layer.constraints.required
    )

def test_planner_does_not_add_character_contact_for_still_life_weapon() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on a wooden table")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon stays attached to the specified hand, waist, or contact point" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )

def test_planner_background_quality_rules_follow_risk_flags() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("single young adult woman with a katana in a quiet forest")

    assert "broad smooth wet reflection bands" not in (
        svp.reference_usage_policy.background_quality_rules
    )
    assert "sparse soft neon blocks" not in (
        svp.reference_usage_policy.background_quality_rules
    )
    assert "background detail stays subordinate to character detail" in (
        svp.reference_usage_policy.background_quality_rules
    )

def test_planner_respects_explicit_detailed_city_background() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman in a detailed cityscape with readable neon signage"
    )

    assert "dense signage" not in svp.composition_layer.constraints.forbidden
    assert "tiny readable text" not in svp.composition_layer.constraints.forbidden
    assert "background: sparse neon blocks instead of dense signage" not in (
        svp.composition_layer.depth_layers
    )
    assert "background detail remains organized and subordinate to the PoR" in (
        svp.composition_layer.constraints.required
    )
