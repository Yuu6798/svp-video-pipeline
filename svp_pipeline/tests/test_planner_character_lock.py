"""Tests for planner (M2) — character_lock identity-trait heuristics."""

from __future__ import annotations

from svp_pipeline.generator import Planner
from tests.fixtures.fakes import DummyClient
from tests.fixtures.mock_responses import VALID_SHIBUYA_RESPONSE


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
