"""Tests for planner (M2)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from svp_pipeline.exceptions import PlannerAPIError, PlannerSchemaError
from svp_pipeline.generator import Planner
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_responses import (
    MISSING_FORBIDDEN_RESPONSE,
    TOO_FEW_POR_CORE_RESPONSE,
    VALID_ACTION_RESPONSE,
    VALID_SHIBUYA_RESPONSE,
    VALID_STILL_LIFE_RESPONSE,
)

REQUIRED_MOTION_FORBIDDEN = (
    "PoR_core要素のフレームアウト",
    "grv_anchor主要要素の画面外移動",
)


class DummyMessages:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("no mock response left")
        return self._responses.pop(0)


class DummyClient:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = DummyMessages(responses=responses, error=error)


def test_plan_returns_valid_svp() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("夕暮れの渋谷スクランブル交差点")

    assert isinstance(svp, SVPVideo)
    assert svp.schema_version == "SVP.v4x-five-layer.video"


def test_plan_with_opus_model() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(model="claude-opus-4-7", client=client)

    planner.plan("opus model test")

    assert client.messages.calls[0]["model"] == "claude-opus-4-6"


def test_plan_with_supported_opus_alias() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(model="claude-opus-4-6", client=client)

    planner.plan("opus alias model test")

    assert client.messages.calls[0]["model"] == "claude-opus-4-6"


def test_plan_default_model_resolves_to_supported_opus_alias() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("default model resolution test")

    assert client.messages.calls[0]["model"] == "claude-opus-4-6"


def test_plan_with_haiku_model() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(model="claude-haiku-4-5", client=client)

    planner.plan("haiku model test")

    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_plan_uses_tool_choice_forced() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("tool choice test")

    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "generate_svp_video"}


def test_system_prompt_teaches_svp_methodology() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("system prompt test")

    system_prompt = client.messages.calls[0]["system"]
    assert "SVP is not merely a JSON schema" in system_prompt
    assert "semantic control protocol" in system_prompt
    assert "positive physical/visual states" in system_prompt
    assert "Object/Contact Proposition Audit" in system_prompt
    assert "Background Simplicity Policy" in system_prompt


def test_plan_injects_duration() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("duration test", duration=8)

    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "8 秒" in user_content
    assert svp.motion_layer.duration_seconds == 8


def test_plan_rejects_non_integer_duration() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    with pytest.raises(ValueError, match="integer"):
        planner.plan("duration type test", duration=8.5)

    assert len(client.messages.calls) == 0


def test_forbidden_injection_when_missing() -> None:
    client = DummyClient(responses=[MISSING_FORBIDDEN_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("forbidden injection test")
    forbidden = svp.motion_layer.constraints.forbidden

    for item in REQUIRED_MOTION_FORBIDDEN:
        assert item in forbidden


def test_forbidden_injection_idempotent() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("idempotent test")
    forbidden = svp.motion_layer.constraints.forbidden

    for item in REQUIRED_MOTION_FORBIDDEN:
        assert forbidden.count(item) == 1


def test_forbidden_injection_preserves_others() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("preserve existing forbidden")
    forbidden = svp.motion_layer.constraints.forbidden

    assert "過剰な手ブレ" in forbidden
    for item in REQUIRED_MOTION_FORBIDDEN:
        assert item in forbidden


def test_character_lock_preserves_literal_subject_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "cyberpunk rainy neon city, young adult woman with silver-gray high "
        "ponytail, vivid red eyes, black and indigo floral kimono coat, "
        "katana at her waist"
    )

    assert "young adult woman" in svp.identity_locks
    assert "female character" in svp.identity_locks
    assert "silver-gray high ponytail" in svp.identity_locks
    assert "vivid red eyes" in svp.identity_locks
    assert "black and indigo floral kimono coat" in svp.identity_locks
    assert "katana at her waist" in svp.identity_locks
    assert "young adult woman" in svp.face_layer.constraints.required
    assert "wrong hair color" in svp.face_layer.constraints.forbidden
    assert "extra characters" in svp.composition_layer.constraints.forbidden
    assert "male character if a female character was specified" in svp.c3.constraints.forbidden
    assert "female character if a male character was specified" not in svp.c3.constraints.forbidden
    assert "silver-gray high ponytail" in svp.c3.evaluation_criteria.hit_list
    assert "hair color or hairstyle changes" in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )
    assert "clean cinematic neon city background" not in (
        svp.reference_usage_policy.background_quality_rules
    )
    assert "smooth neon bokeh in distant signs" not in (
        svp.reference_usage_policy.background_quality_rules
    )
    assert "clean background matching the SVP scene context" in (
        svp.reference_usage_policy.background_quality_rules
    )


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


def test_character_lock_preserves_male_subject_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult man with silver-gray high ponytail, vivid red eyes, "
        "black and indigo floral kimono coat, katana at his waist"
    )

    assert "young adult man" in svp.identity_locks
    assert "male character" in svp.identity_locks
    assert "silver-gray high ponytail" in svp.identity_locks
    assert "vivid red eyes" in svp.identity_locks
    assert "katana at his waist" in svp.identity_locks
    assert "young adult man" in svp.face_layer.constraints.required
    assert "male character" in svp.c3.evaluation_criteria.hit_list
    assert "single primary character only" in svp.composition_layer.constraints.required
    assert "extra characters" in svp.composition_layer.constraints.forbidden
    assert "female character if a male character was specified" in svp.c3.constraints.forbidden
    assert "male character if a female character was specified" not in svp.c3.constraints.forbidden


def test_character_lock_does_not_force_single_subject_for_multi_subject_prompt() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "two characters: a young adult woman with silver-gray high ponytail "
        "and red eyes, and another young adult woman with black hair"
    )

    assert "red eyes" in svp.identity_locks
    assert "single primary character only" not in svp.composition_layer.constraints.required
    assert "extra characters" not in svp.composition_layer.constraints.forbidden
    assert "extra characters" not in svp.c3.constraints.forbidden
    assert "extra or duplicated characters appear" not in (
        svp.c3.evaluation_criteria.critical_fail_conditions
    )


def test_character_lock_ignores_negated_identity_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with silver-gray high ponytail, no red eyes, "
        "without katana"
    )

    assert "silver-gray high ponytail" in svp.identity_locks
    assert "red eyes" not in svp.identity_locks
    assert "katana" not in svp.identity_locks
    assert "red eyes" not in svp.face_layer.constraints.required
    assert "katana" not in svp.c3.evaluation_criteria.hit_list


def test_character_lock_ignores_trailing_english_negated_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with silver-gray high ponytail, "
        "red eyes are not allowed, katana is not desired"
    )

    assert "silver-gray high ponytail" in svp.identity_locks
    assert "red eyes" not in svp.identity_locks
    assert "katana" not in svp.identity_locks
    assert "red eyes" not in svp.face_layer.constraints.required
    assert "katana" not in svp.c3.evaluation_criteria.hit_list


def test_character_lock_does_not_extract_red_eyes_from_eyeshadow() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan(
        "single young adult woman with silver-gray high ponytail and red eyeshadow"
    )

    assert "silver-gray high ponytail" in svp.identity_locks
    assert "red eyes" not in svp.identity_locks
    assert "red eyeshadow" not in svp.identity_locks
    assert "red eyes" not in svp.face_layer.constraints.required
    assert "red eyes" not in svp.c3.evaluation_criteria.hit_list


def test_character_lock_ignores_negated_japanese_identity_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("単独の若い女性、銀髪ポニーテール、赤い瞳ではない、刀は不要")

    assert "female character" in svp.identity_locks
    assert "silver-gray high ponytail" in svp.identity_locks
    assert "赤い瞳" not in svp.identity_locks
    assert "red eyes" not in svp.identity_locks
    assert "刀" not in svp.identity_locks
    assert "katana" not in svp.identity_locks
    assert "赤い瞳" not in svp.face_layer.constraints.required
    assert "katana" not in svp.c3.evaluation_criteria.hit_list


def test_character_lock_detects_japanese_single_subject_intent() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("単独の若い女性、銀髪ポニーテール、赤い瞳")

    assert "female character" in svp.identity_locks
    assert "red eyes" in svp.identity_locks
    assert "single primary character only" in svp.composition_layer.constraints.required
    assert "extra characters" in svp.composition_layer.constraints.forbidden
    assert "extra characters" in svp.c3.constraints.forbidden


def test_character_lock_preserves_japanese_male_subject_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("単独の若い成人男性、銀髪ポニーテール、赤い瞳")

    assert "男性" in svp.identity_locks
    assert "male character" in svp.identity_locks
    assert "silver-gray high ponytail" in svp.identity_locks
    assert "red eyes" in svp.identity_locks
    assert "single primary character only" in svp.composition_layer.constraints.required
    assert "extra characters" in svp.composition_layer.constraints.forbidden


def test_character_lock_ignores_negated_male_subject_traits() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("single character, no male character, 女性")

    assert "female character" in svp.identity_locks
    assert "male character" not in svp.identity_locks
    assert "male character" not in svp.face_layer.constraints.required
    assert "male character" not in svp.c3.evaluation_criteria.hit_list


def test_character_lock_can_be_disabled() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client, character_lock=False)

    svp = planner.plan("young adult woman with silver-gray high ponytail and vivid red eyes")

    assert svp.identity_locks == []


def test_pydantic_validation_error_triggers_retry() -> None:
    client = DummyClient(responses=[TOO_FEW_POR_CORE_RESPONSE, VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("retry on validation error")

    assert isinstance(svp, SVPVideo)
    assert len(client.messages.calls) == 2
    retry_messages = client.messages.calls[1]["messages"]
    assert retry_messages[-1]["content"][0]["type"] == "tool_result"
    assert retry_messages[-1]["content"][0]["is_error"] is True


def test_retry_returns_tool_results_for_all_prior_tool_uses() -> None:
    invalid_input = copy.deepcopy(VALID_SHIBUYA_RESPONSE.content[0].input)
    invalid_input["por_core"] = ["one", "two"]
    extra_input = copy.deepcopy(VALID_STILL_LIFE_RESPONSE.content[0].input)
    first_response = Message(
        id="msg_multi_tool_use",
        content=[
            ToolUseBlock(
                id="toolu_invalid",
                name="generate_svp_video",
                type="tool_use",
                input=invalid_input,
            ),
            ToolUseBlock(
                id="toolu_extra",
                name="generate_svp_video",
                type="tool_use",
                input=extra_input,
            ),
        ],
        model="claude-opus-4-7",
        role="assistant",
        type="message",
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    client = DummyClient(responses=[first_response, VALID_ACTION_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("retry should include all tool_result ids")

    assert isinstance(svp, SVPVideo)
    assert len(client.messages.calls) == 2
    retry_messages = client.messages.calls[1]["messages"]
    tool_results = retry_messages[-1]["content"]
    assert len(tool_results) == 2
    assert {block["tool_use_id"] for block in tool_results} == {"toolu_invalid", "toolu_extra"}
    assert all(block["type"] == "tool_result" for block in tool_results)
    assert all(block["is_error"] is True for block in tool_results)


def test_two_consecutive_failures_raise() -> None:
    client = DummyClient(responses=[TOO_FEW_POR_CORE_RESPONSE, TOO_FEW_POR_CORE_RESPONSE])
    planner = Planner(client=client)

    with pytest.raises(PlannerSchemaError):
        planner.plan("retry should fail twice")

    assert len(client.messages.calls) == 2


def test_api_error_is_wrapped() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    api_error = anthropic.APIError("api down", request=request, body=None)
    client = DummyClient(error=api_error)
    planner = Planner(client=client)

    with pytest.raises(PlannerAPIError):
        planner.plan("api error wrap")


def test_system_prompt_loaded_from_file() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("system prompt load test")

    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "svp_pipeline"
        / "prompts"
        / "planner_system.md"
    )
    expected = prompt_path.read_text(encoding="utf-8").strip()
    assert client.messages.calls[0]["system"] == expected


def test_mock_responses_are_valid_svp() -> None:
    for response in (VALID_SHIBUYA_RESPONSE, VALID_STILL_LIFE_RESPONSE, VALID_ACTION_RESPONSE):
        tool_use = next(block for block in response.content if block.type == "tool_use")
        svp = SVPVideo.model_validate(tool_use.input)
        assert isinstance(svp, SVPVideo)
