"""Tests for failure preset extraction and explicit application."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import svp_pipeline.semantic.failure_presets as failure_presets_mod
from svp_pipeline.schema import SVPVideo
from svp_pipeline.semantic.failure_presets import (
    apply_failure_preset_to_svp,
    apply_failure_presets_to_svp,
    extract_failure_preset_candidate,
    load_failure_preset,
    write_failure_preset_candidate,
)
from svp_pipeline.semantic.models import ObservedRPE, SemanticDiffReport
from svp_pipeline.utils.prompt_render import append_reference_usage_policy, render_image_prompt

SAMPLES_DIR = Path(__file__).parent / "samples"


def _load_svp() -> SVPVideo:
    return SVPVideo.model_validate_json(
        (SAMPLES_DIR / "shibuya_dusk.json").read_text(encoding="utf-8")
    )


def test_extract_duplicate_identity_preset() -> None:
    observed = ObservedRPE(
        violations=[
            "background contains a human face billboard copied from the reference character"
        ],
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert preset.applicability.subject_count == "single_character"
    assert any(item.type == "duplicate_identity_in_background" for item in preset.failure_taxonomy)
    assert any("exactly one visible instance" in item for item in preset.object_instance_rules)
    assert any("human face billboard" in item for item in preset.background_quality_rules)


def test_extract_weapon_residue_preset() -> None:
    observed = ObservedRPE(
        violations=["katana reflection appears as a second sword in the wet background"],
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert "weapon" in preset.applicability.object_classes
    assert any(item.type == "weapon_residue_in_background" for item in preset.failure_taxonomy)
    assert any("weapon-like" in item for item in preset.object_instance_rules)
    assert any("blade-shaped reflections" in item for item in preset.background_quality_rules)


def test_extract_does_not_hardcode_neon_city_for_unknown_background() -> None:
    observed = ObservedRPE(
        violations=["katana reflection appears as a second sword in the wet background"],
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert preset.applicability.background_type is None


def test_extract_background_simplicity_does_not_match_city_substring() -> None:
    observed = ObservedRPE(
        violations=[
            "katana reflection appears as a second sword; background simplicity is desired"
        ],
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert preset.applicability.background_type is None


def test_extract_neon_city_matches_word_boundaries() -> None:
    observed = ObservedRPE(
        violations=["katana reflection appears in a neon city background"],
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert preset.applicability.background_type == "neon_city"


def test_extract_contact_graph_preset() -> None:
    observed = ObservedRPE(
        state={
            "anatomical_contact_graph": ["character_right_hand -> katana_handle"],
            "viewer_contact_graph": ["viewer_left_hand -> katana_handle"],
            "failure_modes": ["left hand grips the wrong object"],
        }
    )
    preset = extract_failure_preset_candidate(observed, SemanticDiffReport())

    assert any(item.type == "hand_object_contact_confusion" for item in preset.failure_taxonomy)
    assert any("anatomical left/right" in item for item in preset.object_instance_rules)
    assert any("swapped hand roles" in item for item in preset.negative_anchors)


def test_apply_failure_preset_to_reference_usage_policy() -> None:
    svp = _load_svp()
    preset = load_failure_preset("single-character-weapon-clean-bg")

    updated = apply_failure_preset_to_svp(svp, preset)
    policy = updated.reference_usage_policy

    assert "single-character-weapon-clean-bg" in policy.failure_preset_ids
    assert any("exactly one visible instance" in item for item in policy.object_instance_rules)
    assert any("background simplicity" in item for item in policy.background_quality_rules)
    assert any(item == "object_instance_integrity" for item in policy.render_priority)
    assert svp.reference_usage_policy.failure_preset_ids == []


def test_apply_failure_presets_to_svp_accepts_path(tmp_path: Path) -> None:
    svp = _load_svp()
    preset = load_failure_preset("single-character-weapon-clean-bg")
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")

    updated = apply_failure_presets_to_svp(svp, [preset_path])

    assert updated.reference_usage_policy.failure_preset_ids == [
        "single-character-weapon-clean-bg"
    ]


def test_render_prompt_includes_failure_prevention_section() -> None:
    svp = apply_failure_presets_to_svp(_load_svp(), ["single-character-weapon-clean-bg"])
    prompt = append_reference_usage_policy(render_image_prompt(svp), svp)

    assert "## Failure Prevention Presets" in prompt
    assert "single-character-weapon-clean-bg" in prompt
    assert "object_instance_integrity" in prompt
    assert "weapon-shaped floor reflections" in prompt


def test_load_builtin_failure_preset() -> None:
    preset = load_failure_preset("single-character-weapon-clean-bg")

    dumped = json.loads(preset.model_dump_json())
    assert dumped["schema_version"] == "SVP.failure_preset.v1"
    assert preset.id == "single-character-weapon-clean-bg"


def test_load_builtin_failure_preset_reads_traversable(monkeypatch: pytest.MonkeyPatch) -> None:
    content = load_failure_preset("single-character-weapon-clean-bg").model_dump_json()

    class FakePresetResource:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            assert encoding == "utf-8"
            return content

    class FakePackageResource:
        def joinpath(self, _name: str) -> FakePresetResource:
            return FakePresetResource()

    monkeypatch.setattr(
        failure_presets_mod.resources,
        "files",
        lambda _package: FakePackageResource(),
    )

    preset = load_failure_preset("single-character-weapon-clean-bg")

    assert preset.id == "single-character-weapon-clean-bg"


def test_write_failure_preset_candidate_round_trips(tmp_path: Path) -> None:
    preset = extract_failure_preset_candidate(
        ObservedRPE(violations=["background face copy and katana reflection"]),
        SemanticDiffReport(),
    )
    path = tmp_path / "failure_preset.candidate.json"

    write_failure_preset_candidate(path, preset)

    loaded = load_failure_preset(path)
    assert loaded == preset
