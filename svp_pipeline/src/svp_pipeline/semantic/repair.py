"""Repair SVP proposals from semantic diff reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..schema import SVPVideo
from .models import ObservedRPE, RepairProposal, SemanticDiffReport
from .rpe import diff_rpe, extract_expected_rpe, load_observed_rpe, write_json


@dataclass
class RepairResult:
    """Files written by an RPE repair pass."""

    output_dir: Path
    semantic_diff_path: Path
    proposal_path: Path
    proposed_svp_path: Path
    proposed_svp: SVPVideo
    diff: SemanticDiffReport


def repair_svp_from_observed_rpe(
    *,
    base_svp_path: Path,
    observed_rpe_path: Path,
    output_root: Path,
) -> RepairResult:
    """Create a repaired SVP proposal from a base SVP and observed RPE."""
    base_svp_path = Path(base_svp_path)
    observed_rpe_path = Path(observed_rpe_path)
    output_dir = _make_repair_dir(Path(output_root))

    svp = SVPVideo.model_validate_json(base_svp_path.read_text(encoding="utf-8"))
    observed = load_observed_rpe(observed_rpe_path)
    expected = extract_expected_rpe(svp)
    diff = diff_rpe(expected=expected, observed=observed)
    proposed = apply_repair(svp=svp, observed=observed, diff=diff)

    semantic_diff_path = output_dir / "semantic_diff.json"
    proposed_svp_path = output_dir / "target_svp.proposed.json"
    proposal_path = output_dir / "repair_proposal.json"
    needs_regeneration = bool(diff.issues)

    write_json(semantic_diff_path, diff)
    proposed_svp_path.write_text(proposed.model_dump_json(indent=2), encoding="utf-8")
    proposal = RepairProposal(
        base_svp=str(base_svp_path),
        observed_rpe=str(observed_rpe_path),
        proposed_svp=str(proposed_svp_path),
        semantic_diff=str(semantic_diff_path),
        rationale=[issue.repair_hint for issue in diff.issues],
        regeneration_plan={
            "reuse_reference": True,
            "reuse_image": not needs_regeneration,
            "regenerate_image": needs_regeneration,
            "regenerate_video": needs_regeneration,
        },
    )
    write_json(proposal_path, proposal)

    return RepairResult(
        output_dir=output_dir,
        semantic_diff_path=semantic_diff_path,
        proposal_path=proposal_path,
        proposed_svp_path=proposed_svp_path,
        proposed_svp=proposed,
        diff=diff,
    )


def apply_repair(
    *,
    svp: SVPVideo,
    observed: ObservedRPE,
    diff: SemanticDiffReport,
) -> SVPVideo:
    """Apply conservative repairs to a copied SVP."""
    state_required = _state_required_repairs(observed.state)
    state_contact_points = _state_contact_point_repairs(observed.state)
    state_forbidden = _state_forbidden_repairs(observed.state)
    required_additions = _dedupe([*observed.missing, *state_required])
    forbidden_additions = _dedupe([*observed.violations, *state_forbidden])
    critical_additions = _dedupe(
        [f"Observed failure: {issue.observed}" for issue in diff.issues]
    )

    c3_constraints = svp.c3.constraints.model_copy(
        update={
            "required": _append_unique(list(svp.c3.constraints.required), required_additions),
            "forbidden": _append_unique(list(svp.c3.constraints.forbidden), forbidden_additions),
        }
    )
    evaluation = svp.c3.evaluation_criteria.model_copy(
        update={
            "hit_list": _append_unique(
                list(svp.c3.evaluation_criteria.hit_list),
                required_additions,
            ),
            "critical_fail_conditions": _append_unique(
                list(svp.c3.evaluation_criteria.critical_fail_conditions),
                critical_additions + forbidden_additions,
            ),
        }
    )
    c3 = svp.c3.model_copy(
        update={
            "constraints": c3_constraints,
            "evaluation_criteria": evaluation,
        }
    )

    composition_constraints = svp.composition_layer.constraints.model_copy(
        update={
            "required": _append_unique(
                list(svp.composition_layer.constraints.required),
                _composition_required_repairs(required_additions),
            ),
            "forbidden": _append_unique(
                list(svp.composition_layer.constraints.forbidden),
                _composition_forbidden_repairs(forbidden_additions),
            ),
        }
    )
    composition_layer = svp.composition_layer.model_copy(
        update={"constraints": composition_constraints}
    )

    pose_constraints = svp.pose_layer.constraints.model_copy(
        update={
            "required": _append_unique(
                list(svp.pose_layer.constraints.required),
                _pose_required_repairs(required_additions),
            ),
            "forbidden": _append_unique(
                list(svp.pose_layer.constraints.forbidden),
                _pose_forbidden_repairs(forbidden_additions),
            ),
        }
    )
    pose_layer = svp.pose_layer.model_copy(update={"constraints": pose_constraints})
    pose_layer = pose_layer.model_copy(
        update={
            "contact_points": _merge_contact_points(
                list(pose_layer.contact_points),
                state_contact_points,
            )
        }
    )

    reference_policy = svp.reference_usage_policy.model_copy(
        update={
            "do_not_copy_from_reference": _append_unique(
                list(svp.reference_usage_policy.do_not_copy_from_reference),
                _reference_forbidden_repairs(forbidden_additions),
            ),
            "object_instance_rules": _append_unique(
                list(svp.reference_usage_policy.object_instance_rules),
                _object_rule_repairs(forbidden_additions),
            ),
            "background_quality_rules": _append_unique(
                list(svp.reference_usage_policy.background_quality_rules),
                _background_rule_repairs(forbidden_additions),
            ),
        }
    )

    return svp.model_copy(
        update={
            "c3": c3,
            "composition_layer": composition_layer,
            "pose_layer": pose_layer,
            "reference_usage_policy": reference_policy,
        }
    )


def _make_repair_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"repair-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}-{index:02d}"
        index += 1
    candidate.mkdir()
    return candidate


def _composition_required_repairs(items: list[str]) -> list[str]:
    return [f"Repair required visual proposition: {item}" for item in items]


def _composition_forbidden_repairs(items: list[str]) -> list[str]:
    return [f"Repair forbidden visual failure: {item}" for item in items]


def _pose_required_repairs(items: list[str]) -> list[str]:
    return [item for item in items if _mentions_object_contact(item)]


def _pose_forbidden_repairs(items: list[str]) -> list[str]:
    return [item for item in items if _mentions_object_contact(item)]


def _reference_forbidden_repairs(items: list[str]) -> list[str]:
    return [
        item
        for item in items
        if any(token in item.lower() for token in ("reference", "background", "noise", "artifact"))
    ]


def _object_rule_repairs(items: list[str]) -> list[str]:
    return [
        f"Object failure must not recur: {item}"
        for item in items
        if _mentions_object_contact(item)
    ]


def _background_rule_repairs(items: list[str]) -> list[str]:
    return [
        f"Background cleanup rule: {item}"
        for item in items
        if any(token in item.lower() for token in ("background", "reflection", "noise", "artifact"))
    ]


def _mentions_object_contact(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in ("katana", "sword", "blade", "umbrella", "hand", "floating", "reflection")
    )


def _state_required_repairs(state: dict) -> list[str]:
    """Turn observed successful visual-state variables into required constraints."""
    required: list[str] = []
    for item in _state_list(state, "object_graph"):
        required.append(f"Object state must be preserved: {item}")
    for item in _state_list(state, "contact_graph"):
        required.append(f"Contact graph must be preserved: {item}")
    for item in _state_list(state, "viewer_contact_graph"):
        required.append(f"Viewer contact graph must be preserved: {item}")
    for item in _state_list(state, "anatomical_contact_graph"):
        required.append(f"Anatomical contact graph must be preserved: {item}")
    pose_intent = state.get("pose_intent")
    if isinstance(pose_intent, str) and pose_intent.strip():
        required.append(f"Pose intent must be preserved: {pose_intent.strip()}")
    return required


def _state_contact_point_repairs(state: dict) -> list[str]:
    contact_points: list[str] = []
    for item in _state_list(state, "contact_graph"):
        contact_points.append(f"RPE contact graph: {item}")
    for item in _state_list(state, "viewer_contact_graph"):
        contact_points.append(f"RPE viewer contact graph: {item}")
    for item in _state_list(state, "anatomical_contact_graph"):
        contact_points.append(f"RPE anatomical contact graph: {item}")
    pose_intent = state.get("pose_intent")
    if isinstance(pose_intent, str) and pose_intent.strip():
        contact_points.append(f"RPE pose intent: {pose_intent.strip()}")
    return contact_points


def _state_forbidden_repairs(state: dict) -> list[str]:
    forbidden: list[str] = []
    for key in ("failure_modes", "physical_failures", "contact_graph_failures"):
        for item in _state_list(state, key):
            forbidden.append(f"Observed visual-state failure must not recur: {item}")
    return forbidden


def _state_list(state: dict, key: str) -> list[str]:
    value = state.get(key)
    if isinstance(value, list):
        return _dedupe([str(item) for item in value if str(item).strip()])
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _merge_contact_points(existing: list[str], additions: list[str]) -> list[str]:
    """Merge RPE contact graph repairs, replacing older contacts for the same actor."""
    actors = _contact_actors(additions)
    if not actors:
        return _append_unique(existing, additions)

    filtered = [
        item
        for item in existing
        if not any(_contact_item_mentions_actor(item, actor) for actor in actors)
    ]
    return _append_unique(filtered, additions)


def _contact_actors(items: list[str]) -> set[str]:
    actors: set[str] = set()
    for item in items:
        text = item
        if ":" in text:
            text = text.split(":", 1)[1]
        if "->" in text:
            actor = text.split("->", 1)[0].strip().lower()
            if actor:
                actors.add(actor)
    return actors


def _contact_item_mentions_actor(item: str, actor: str) -> bool:
    aliases = _actor_aliases(actor)
    normalized = item.lower().replace("-", "_")
    return any(alias in normalized for alias in aliases)


def _actor_aliases(actor: str) -> tuple[str, ...]:
    normalized = actor.lower().replace("-", "_").strip()
    if normalized in {"left_hand", "left hand", "viewer_left_hand", "character_left_hand"}:
        return (
            "left_hand",
            "left hand",
            "left-hand",
            "viewer_left_hand",
            "character_left_hand",
            "左手",
        )
    if normalized in {"right_hand", "right hand", "viewer_right_hand", "character_right_hand"}:
        return (
            "right_hand",
            "right hand",
            "right-hand",
            "viewer_right_hand",
            "character_right_hand",
            "右手",
        )
    return (normalized, normalized.replace("_", " "))


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    seen = {item.strip().lower() for item in existing if item.strip()}
    out = [item for item in existing if item.strip()]
    for item in additions:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            out.append(normalized)
            seen.add(key)
    return out


def _dedupe(values: list[str]) -> list[str]:
    return _append_unique([], values)
