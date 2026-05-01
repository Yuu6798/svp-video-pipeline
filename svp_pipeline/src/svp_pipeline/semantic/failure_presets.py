"""Failure-preset extraction and explicit application.

Failure presets are reusable, human-reviewable repair knowledge extracted from
RPE/semantic-diff packets. They are not auto-promoted or auto-applied; callers
must opt in by passing a preset id or JSON path.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from svp_pipeline.schema import SVPVideo

from .models import ObservedRPE, SemanticDiffReport


class FailurePresetApplicability(BaseModel):
    """Conditions where a failure preset is intended to be useful."""

    subject_count: str | None = None
    object_classes: list[str] = Field(default_factory=list)
    background_type: str | None = None
    reference_image_mode: bool | None = None
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class FailureTaxonomyEntry(BaseModel):
    """One classified failure mode and its repair strategy."""

    type: str
    object: str | None = None
    region: str | None = None
    severity: Literal["info", "warn", "critical"] = "critical"
    repair_strategy: str

    model_config = ConfigDict(extra="forbid")


class FailurePresetCandidate(BaseModel):
    """Reusable failure-prevention instructions derived from semantic repair."""

    schema_version: Literal["SVP.failure_preset.v1"] = "SVP.failure_preset.v1"
    id: str = "candidate"
    source: dict[str, str] = Field(default_factory=dict)
    applicability: FailurePresetApplicability = Field(default_factory=FailurePresetApplicability)
    failure_taxonomy: list[FailureTaxonomyEntry] = Field(default_factory=list)
    object_instance_rules: list[str] = Field(default_factory=list)
    background_quality_rules: list[str] = Field(default_factory=list)
    do_not_copy_from_reference: list[str] = Field(default_factory=list)
    render_priority: list[str] = Field(default_factory=list)
    conflict_resolution: list[str] = Field(default_factory=list)
    positive_anchors: list[str] = Field(default_factory=list)
    negative_anchors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def extract_failure_preset_candidate(
    observed: ObservedRPE,
    diff: SemanticDiffReport,
    *,
    source: dict[str, str] | None = None,
    preset_id: str = "candidate",
) -> FailurePresetCandidate:
    """Extract a conservative failure-preset candidate from RPE + semantic diff."""

    texts = _collect_failure_texts(observed, diff)
    blob = "\n".join(texts).lower()
    preset = FailurePresetCandidate(id=preset_id, source=source or {})

    background_type = _infer_background_type(blob)

    if _has_duplicate_identity_failure(blob):
        _add_duplicate_identity_rules(preset, background_type=background_type)
    if _has_weapon_residue_failure(blob):
        _add_weapon_residue_rules(preset, background_type=background_type)
    if _has_contact_graph_failure(blob):
        _add_contact_graph_rules(preset)
    if _has_background_noise_failure(blob):
        _add_background_simplicity_rules(preset)

    if preset.failure_taxonomy:
        preset.applicability.notes = _append_unique(
            preset.applicability.notes,
            ["Extracted from observed RPE and semantic diff; requires human review."],
        )
        preset.render_priority = _append_unique(
            preset.render_priority,
            [
                "character_identity",
                "pose_contact_graph",
                "object_instance_integrity",
                "lighting_atmosphere",
                "wet_reflections",
                "background_simplicity",
                "background_detail",
            ],
        )
        preset.conflict_resolution = _append_unique(
            preset.conflict_resolution,
            [
                "If background detail conflicts with object integrity, prefer object integrity.",
                "If wet reflections create object residue, simplify or blur the reflection.",
            ],
        )

    return preset


def write_failure_preset_candidate(path: Path, preset: FailurePresetCandidate) -> None:
    """Write a failure preset candidate as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")


def load_failure_preset(identifier_or_path: str | Path) -> FailurePresetCandidate:
    """Load a failure preset by JSON path or built-in preset id."""

    path = _resolve_preset_path(identifier_or_path)
    try:
        return FailurePresetCandidate.model_validate_json(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid failure preset JSON: {path}") from exc


def apply_failure_presets_to_svp(
    svp: SVPVideo,
    preset_ids_or_paths: list[str | Path],
) -> SVPVideo:
    """Apply multiple explicit failure presets to an SVP."""

    updated = svp
    for preset_id_or_path in preset_ids_or_paths:
        updated = apply_failure_preset_to_svp(updated, load_failure_preset(preset_id_or_path))
    return updated


def apply_failure_preset_to_svp(svp: SVPVideo, preset: FailurePresetCandidate) -> SVPVideo:
    """Merge one preset into SVP.reference_usage_policy."""

    updated = svp.model_copy(deep=True)
    policy = updated.reference_usage_policy
    policy.failure_preset_ids = _append_unique(policy.failure_preset_ids, [preset.id])
    policy.object_instance_rules = _append_unique(
        policy.object_instance_rules,
        preset.object_instance_rules,
    )
    policy.background_quality_rules = _append_unique(
        policy.background_quality_rules,
        preset.background_quality_rules,
    )
    policy.do_not_copy_from_reference = _append_unique(
        policy.do_not_copy_from_reference,
        preset.do_not_copy_from_reference,
    )
    policy.render_priority = _append_unique(policy.render_priority, preset.render_priority)
    policy.conflict_resolution = _append_unique(
        policy.conflict_resolution,
        preset.conflict_resolution,
    )
    policy.positive_anchors = _append_unique(policy.positive_anchors, preset.positive_anchors)
    policy.negative_anchors = _append_unique(policy.negative_anchors, preset.negative_anchors)
    return updated


def _collect_failure_texts(observed: ObservedRPE, diff: SemanticDiffReport) -> list[str]:
    texts: list[str] = []
    texts.extend(observed.observed)
    texts.extend(observed.missing)
    texts.extend(observed.violations)
    texts.extend(observed.notes)
    texts.extend(_flatten_state_texts(observed.state))
    for issue in diff.issues:
        texts.extend([issue.expected, issue.observed, issue.repair_hint, issue.layer])
    texts.extend(diff.observed_notes)
    return [text for text in texts if text]


def _flatten_state_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for key, nested in value.items():
            texts.append(str(key))
            texts.extend(_flatten_state_texts(nested))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for item in value:
            texts.extend(_flatten_state_texts(item))
        return texts
    return [str(value)]


def _has_duplicate_identity_failure(blob: str) -> bool:
    identity_terms = (
        "duplicate character",
        "extra character",
        "background character",
        "background face",
        "human face billboard",
        "portrait hologram",
        "character-shaped signage",
        "hologram face",
        "billboard face",
        "人物の写り込み",
        "背景に顔",
        "背景に人物",
    )
    return any(term in blob for term in identity_terms)


def _has_weapon_residue_failure(blob: str) -> bool:
    weapon_terms = ("katana", "sword", "weapon", "blade", "刀", "剣")
    residue_terms = (
        "reflection",
        "residue",
        "background",
        "signage",
        "symbol",
        "motif",
        "silhouette",
        "trail",
        "映り込み",
        "残像",
        "背景",
        "反射",
    )
    return any(term in blob for term in weapon_terms) and any(
        term in blob for term in residue_terms
    )


def _has_contact_graph_failure(blob: str) -> bool:
    contact_terms = (
        "wrong hand",
        "wrong grip",
        "contact",
        "scabbard",
        "handle",
        "left hand",
        "right hand",
        "viewer_left",
        "viewer_right",
        "character_left",
        "character_right",
        "鞘",
        "柄",
        "左手",
        "右手",
        "接触",
    )
    return any(term in blob for term in contact_terms)


def _has_background_noise_failure(blob: str) -> bool:
    noise_terms = (
        "background noise",
        "scratch",
        "speckle",
        "micro-line",
        "texture noise",
        "compression artifact",
        "ざらつき",
        "背景ノイズ",
        "細密",
    )
    return any(term in blob for term in noise_terms)


def _infer_background_type(blob: str) -> str | None:
    if any(term in blob for term in ("neon", "city", "shibuya", "urban", "都市", "街")):
        return "neon_city"
    if any(term in blob for term in ("forest", "woods", "tree", "森", "林")):
        return "forest"
    if any(term in blob for term in ("interior", "room", "indoor", "室内", "部屋")):
        return "interior"
    return None


def _add_duplicate_identity_rules(
    preset: FailurePresetCandidate,
    *,
    background_type: str | None,
) -> None:
    preset.applicability.subject_count = "single_character"
    preset.applicability.background_type = preset.applicability.background_type or background_type
    preset.applicability.reference_image_mode = True
    preset.failure_taxonomy = _append_unique_models(
        preset.failure_taxonomy,
        [
            FailureTaxonomyEntry(
                type="duplicate_identity_in_background",
                object="character",
                region="background",
                repair_strategy=(
                    "keep exactly one foreground character and remove "
                    "identity-like background copies"
                ),
            )
        ],
    )
    preset.object_instance_rules = _append_unique(
        preset.object_instance_rules,
        ["character: exactly one visible instance, foreground only"],
    )
    preset.background_quality_rules = _append_unique(
        preset.background_quality_rules,
        [
            "background may contain abstract neon signage but no human face billboard",
            "no portrait hologram, character-shaped signage, or second character silhouette",
        ],
    )
    preset.do_not_copy_from_reference = _append_unique(
        preset.do_not_copy_from_reference,
        [
            (
                "do not copy the reference character into background signage, "
                "holograms, billboards, or reflections"
            )
        ],
    )
    preset.positive_anchors = _append_unique(
        preset.positive_anchors,
        ["one foreground character with the intended face, hair, outfit, and pose"],
    )
    preset.negative_anchors = _append_unique(
        preset.negative_anchors,
        ["background face copies", "portrait holograms", "extra character silhouettes"],
    )


def _add_weapon_residue_rules(
    preset: FailurePresetCandidate,
    *,
    background_type: str | None,
) -> None:
    preset.applicability.object_classes = _append_unique(
        preset.applicability.object_classes,
        ["weapon"],
    )
    preset.applicability.background_type = preset.applicability.background_type or background_type
    preset.failure_taxonomy = _append_unique_models(
        preset.failure_taxonomy,
        [
            FailureTaxonomyEntry(
                type="weapon_residue_in_background",
                object="weapon",
                region="background/reflection",
                repair_strategy=(
                    "keep the weapon as one foreground object and suppress "
                    "weapon-like background residue"
                ),
            )
        ],
    )
    preset.object_instance_rules = _append_unique(
        preset.object_instance_rules,
        [
            "katana or sword: exactly one foreground object only",
            (
                "no weapon-like silhouettes, signage, reflections, shadows, "
                "or repeated motifs in the background"
            ),
        ],
    )
    preset.background_quality_rules = _append_unique(
        preset.background_quality_rules,
        [
            "wet reflections must be smooth color reflections, not blade-shaped reflections",
            "simplify or blur background details that resemble weapons",
        ],
    )
    preset.do_not_copy_from_reference = _append_unique(
        preset.do_not_copy_from_reference,
        ["do not repeat katana or weapon motifs in background signage or wet reflections"],
    )
    preset.positive_anchors = _append_unique(
        preset.positive_anchors,
        ["single foreground weapon attached to the intended hand/contact point"],
    )
    preset.negative_anchors = _append_unique(
        preset.negative_anchors,
        ["weapon-shaped floor reflections", "background sword symbols", "extra blade trails"],
    )


def _add_contact_graph_rules(preset: FailurePresetCandidate) -> None:
    preset.failure_taxonomy = _append_unique_models(
        preset.failure_taxonomy,
        [
            FailureTaxonomyEntry(
                type="hand_object_contact_confusion",
                object="hands/weapon",
                region="foreground",
                repair_strategy=(
                    "prioritize explicit character-body-side contact graph "
                    "over visual guesswork"
                ),
            )
        ],
    )
    preset.object_instance_rules = _append_unique(
        preset.object_instance_rules,
        [
            "preserve anatomical left/right hand assignments exactly as written",
            "do not swap viewer-left/viewer-right with character-left/character-right",
            "hands must contact only their assigned object parts",
        ],
    )
    preset.positive_anchors = _append_unique(
        preset.positive_anchors,
        ["explicit anatomical hand-object contact graph"],
    )
    preset.negative_anchors = _append_unique(
        preset.negative_anchors,
        [
            "swapped hand roles",
            "floating weapon parts",
            "two-handed grip when one hand should hold the scabbard",
        ],
    )


def _add_background_simplicity_rules(preset: FailurePresetCandidate) -> None:
    preset.failure_taxonomy = _append_unique_models(
        preset.failure_taxonomy,
        [
            FailureTaxonomyEntry(
                type="background_overdetail_noise",
                object="background",
                region="background",
                severity="warn",
                repair_strategy="reduce distant micro-details before adding more signs or texture",
            )
        ],
    )
    preset.background_quality_rules = _append_unique(
        preset.background_quality_rules,
        [
            "background simplicity has higher priority than background detail",
            "prefer smooth organized composition over dense distant micro-details",
        ],
    )


def _resolve_preset_path(identifier_or_path: str | Path) -> Path:
    path = Path(identifier_or_path)
    if path.is_file():
        return path

    name = str(identifier_or_path)
    candidate_names = [name]
    if not name.endswith(".json"):
        candidate_names.append(f"{name}.json")

    for candidate_name in candidate_names:
        try:
            candidate = resources.files("svp_pipeline.presets.failure").joinpath(
                candidate_name
            )
        except ModuleNotFoundError:
            candidate = None
        if candidate is not None and candidate.is_file():
            return Path(str(candidate))

    raise FileNotFoundError(f"Failure preset not found: {identifier_or_path}")


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    result = list(existing)
    seen = set(result)
    for item in additions:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _append_unique_models(
    existing: list[FailureTaxonomyEntry],
    additions: list[FailureTaxonomyEntry],
) -> list[FailureTaxonomyEntry]:
    result = list(existing)
    seen = {item.model_dump_json() for item in result}
    for item in additions:
        key = item.model_dump_json()
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result
