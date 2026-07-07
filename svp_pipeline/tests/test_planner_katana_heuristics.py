"""Tests for planner (M2) — katana/umbrella/waist-sheath heuristics."""

from __future__ import annotations

import pytest

from svp_pipeline.generator import Planner
from tests.fixtures.fakes import DummyClient
from tests.fixtures.mock_responses import VALID_SHIBUYA_RESPONSE, VALID_STILL_LIFE_RESPONSE


def test_planner_treats_hand_holding_katana_as_character_contact() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "hands holding a katana in a simple dark indoor background"
    )

    assert "main weapon is a single physical object" in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" in (
        svp.pose_layer.contact_points
    )
    assert "no weapon-like reflections in the background" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_treats_katana_in_hand_as_character_contact() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("katana in hand in a simple dark indoor background")

    assert "main weapon is a single physical object" in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" in (
        svp.pose_layer.contact_points
    )
    assert "no weapon-like reflections in the background" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_resolves_umbrella_katana_contact_consistency() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "cyberpunk rainy neon city, single young adult woman with silver-gray "
        "high ponytail, vivid red eyes, transparent umbrella, katana at her waist"
    )

    assert svp.pose_layer.hand_state == (
        "one hand holds the umbrella handle; the other hand stays relaxed "
        "near the waist; no hand holds the katana"
    )
    assert "one hand <-> umbrella handle" in svp.pose_layer.contact_points
    assert "katana sheath <-> waist belt" in svp.pose_layer.contact_points
    assert "one hand holds the umbrella handle" in svp.pose_layer.constraints.required
    assert "katana is sheathed and attached to the waist belt" in (
        svp.pose_layer.constraints.required
    )
    assert "hands do not hold the katana while holding the umbrella" in (
        svp.pose_layer.constraints.required
    )
    assert "floating katana" in svp.pose_layer.constraints.forbidden
    assert "unsheathed blade" in svp.pose_layer.constraints.forbidden
    assert "drawn katana" in svp.pose_layer.constraints.forbidden
    expected_contact_rule = (
        "Object-contact proposition: one hand holds umbrella; "
        "katana remains sheathed at waist"
    )
    assert expected_contact_rule in svp.c3.constraints.required
    assert "both hands hold umbrella while katana appears unsheathed or floating" in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )
    assert "umbrella count = exactly one" in (
        svp.reference_usage_policy.object_instance_rules
    )
    assert "katana must be sheathed and attached to waist, not floating" in (
        svp.reference_usage_policy.object_instance_rules
    )
    assert "katana may not be held if umbrella occupies a hand" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_allows_explicit_drawn_katana_with_umbrella() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman under a transparent umbrella holding a drawn katana"
    )

    assert svp.pose_layer.hand_state != (
        "one hand holds the umbrella handle; the other hand stays relaxed "
        "near the waist; no hand holds the katana"
    )
    assert "drawn katana" not in svp.pose_layer.constraints.forbidden
    assert "katana may not be held if umbrella occupies a hand" not in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_allows_drawing_katana_from_waist() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman drawing a katana from the sheath at her waist "
        "in a simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_drawing_katana_from_determined_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman drawing a katana from her sheath, katana at her "
        "waist, simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_drawing_possessive_katana_from_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman drawing her katana from her sheath, katana at "
        "her waist, simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

@pytest.mark.parametrize(
    "prompt",
    [
        "single young adult woman, she draws her katana at her waist in a simple "
        "dark indoor background",
        "single young adult woman drawing her katana at her waist in a simple "
        "dark indoor background",
        "single young adult woman, she drew her katana at her waist in a simple "
        "dark indoor background",
        "single young adult woman, she pulled her katana at her waist in a simple "
        "dark indoor background",
        "single young adult woman, she had drawn her katana at her waist in a "
        "simple dark indoor background",
    ],
)
def test_planner_allows_actor_draw_katana_without_sheath_preposition(prompt: str) -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(prompt)

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_possessive_held_katana_at_waist() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman holding her katana at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_unsheathing_possessive_katana_at_waist() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman unsheathing her katana at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_unsheathe_possessive_katana_at_waist() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman ready to unsheathe her katana at her waist in a "
        "simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_allows_unsheathed_possessive_katana_at_waist() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman unsheathed her katana at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_draw_imperative_keeps_katana_policy() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "draw a single young adult woman with a katana at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_does_not_force_umbrella_rules_without_umbrella() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "cyberpunk rainy neon city, single young adult woman with silver-gray "
        "high ponytail, vivid red eyes, katana at her waist"
    )

    assert svp.pose_layer.hand_state != (
        "one hand holds the umbrella handle; the other hand stays relaxed "
        "near the waist; no hand holds the katana"
    )
    assert "one hand <-> umbrella handle" not in svp.pose_layer.contact_points
    assert "umbrella count = exactly one" not in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_limits_katana_reflections_to_physical_waist_object() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana sheathed at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana does not cast a distinct reflection, shadow, trail, or silhouette" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "blade-like line outside the waist sheath" in (
        svp.style_layer.constraints.forbidden
    )
    assert "floor or background contains a blade-like reflection" in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )
    assert (
        "no blade-like line may appear on floor, wall, glass, umbrella, rain, or background"
        in svp.reference_usage_policy.object_instance_rules
    )

def test_planner_does_not_force_waist_katana_policy_for_back_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a sheathed katana strapped across her back "
        "in a simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana does not cast a distinct reflection, shadow, trail, or silhouette" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" not in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_detects_side_qualified_hip_katana() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a sheathed katana at her left hip in a "
        "simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_detects_determinerless_waist_katana() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana at waist in a simple dark indoor "
        "background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )

@pytest.mark.parametrize(
    "prompt",
    [
        "single young adult woman with a left hip katana tattoo in a simple dark "
        "indoor background",
        "single young adult woman with a left hip katana sheath tattoo in a simple "
        "dark indoor background",
        "single young adult woman with a left hip katana scabbard tattoo in a "
        "simple dark indoor background",
        "single young adult woman with a left hip sheathed katana tattoo in a "
        "simple dark indoor background",
    ],
)
def test_planner_does_not_treat_hip_katana_tattoo_as_waist_sheath(prompt: str) -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(prompt)

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_waist_high_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on waist-high table")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden

def test_planner_does_not_treat_hip_high_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on hip-high table")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden

def test_planner_does_not_treat_waist_height_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on waist-height table")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden

def test_planner_does_not_treat_hip_height_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on hip-height table")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden

def test_planner_does_not_treat_waist_level_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana on waist-level table in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_hip_level_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana on hip level shelf in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_space_separated_waist_high_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana on waist high table in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_space_separated_hip_high_surface_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a katana on hip high table in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_bare_belt_as_character_contact() -> None:
    client = DummyClient(responses=[VALID_STILL_LIFE_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("still life product shot of a katana on a leather belt")

    assert "main weapon is a single physical object" not in (
        svp.pose_layer.constraints.required
    )
    assert "main weapon attached to character contact point" not in (
        svp.pose_layer.contact_points
    )
    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden

@pytest.mark.parametrize(
    "prompt",
    [
        "single young adult woman next to a katana on the belt display stand in a "
        "simple dark indoor background",
        "single young adult woman next to a katana on the hip display stand in a "
        "simple dark indoor background",
        "single young adult woman next to a katana on the waist rack in a simple "
        "dark indoor background",
    ],
)
def test_planner_does_not_treat_display_fixture_as_waist_sheath(prompt: str) -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(prompt)

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana appears anywhere except physical waist hilt/sheath" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )

def test_planner_does_not_treat_near_waist_as_waist_sheath() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman holding a sheathed katana near her waist in a "
        "simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" not in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_detects_strapped_waist_katana() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a sheathed katana strapped to her waist in "
        "a simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_detects_pronoun_only_waist_katana() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "katana sheathed at her waist in a simple dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" in (
        svp.reference_usage_policy.object_instance_rules
    )

def test_planner_does_not_apply_katana_policy_to_generic_waist_sword() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with a sheathed sword at her waist in a simple "
        "dark indoor background"
    )

    assert "katana visible area is limited to physical waist hilt and sheath only" not in (
        svp.pose_layer.constraints.required
    )
    assert "katana reflection on floor" not in svp.composition_layer.constraints.forbidden
    assert "katana casts no distinct reflection, shadow, trail, or silhouette" not in (
        svp.reference_usage_policy.object_instance_rules
    )
    assert "floor or background contains a blade-like reflection" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )
