"""RPE extraction and semantic diff helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import SVPVideo
from .models import ExpectedRPE, ObservedRPE, RPEProposition, SemanticDiffReport, SemanticIssue


def extract_expected_rpe(svp: SVPVideo) -> ExpectedRPE:
    """Extract expected required/forbidden propositions from an SVP."""
    required: list[RPEProposition] = []
    forbidden: list[RPEProposition] = []

    def add_required(layer: str, values: list[str]) -> None:
        required.extend(_props(layer=layer, values=values, kind="required"))

    def add_forbidden(layer: str, values: list[str]) -> None:
        forbidden.extend(_props(layer=layer, values=values, kind="forbidden"))

    add_required("global.por_core", svp.por_core)
    add_required("global.grv_anchor", svp.grv_anchor)
    add_required("global.identity_locks", svp.identity_locks)
    add_required("global.c3.required", svp.c3.constraints.required)
    add_required("global.hit_list", svp.c3.evaluation_criteria.hit_list)
    add_forbidden("global.c3.forbidden", svp.c3.constraints.forbidden)
    add_forbidden("global.motion_forbidden", svp.c3.constraints.motion_forbidden)
    add_forbidden("global.critical_fail", svp.c3.evaluation_criteria.critical_fail_conditions)

    for layer_name in (
        "composition_layer",
        "face_layer",
        "style_layer",
        "pose_layer",
        "motion_layer",
    ):
        layer = getattr(svp, layer_name)
        add_required(f"{layer_name}.por_core", layer.por_core)
        add_required(f"{layer_name}.required", layer.constraints.required)
        add_forbidden(f"{layer_name}.forbidden", layer.constraints.forbidden)

    add_required("pose_layer.contact_points", svp.pose_layer.contact_points)
    add_required(
        "reference_usage_policy.use_reference_for",
        svp.reference_usage_policy.use_reference_for,
    )
    add_forbidden(
        "reference_usage_policy.do_not_copy",
        svp.reference_usage_policy.do_not_copy_from_reference,
    )
    add_forbidden(
        "reference_usage_policy.object_rules",
        svp.reference_usage_policy.object_instance_rules,
    )

    return ExpectedRPE(required=_dedupe_props(required), forbidden=_dedupe_props(forbidden))


def load_observed_rpe(path: Path) -> ObservedRPE:
    """Load observed RPE from JSON."""
    return ObservedRPE.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))


def diff_rpe(expected: ExpectedRPE, observed: ObservedRPE) -> SemanticDiffReport:
    """Create a semantic diff report from expected and observed RPE."""
    issues: list[SemanticIssue] = []

    for missing in observed.missing:
        expected_prop = _find_best_expected(missing, expected.required)
        issues.append(
            SemanticIssue(
                code="missing_required",
                severity="fail",
                layer=expected_prop.layer if expected_prop is not None else "manual_observation",
                expected=expected_prop.text if expected_prop is not None else missing,
                observed=missing,
                repair_hint="Strengthen this proposition in required constraints and hit_list.",
            )
        )

    for violation in observed.violations:
        expected_prop = _find_best_expected(violation, expected.forbidden)
        issues.append(
            SemanticIssue(
                code="forbidden_violation",
                severity="fail",
                layer=expected_prop.layer if expected_prop is not None else "manual_observation",
                expected=expected_prop.text if expected_prop is not None else violation,
                observed=violation,
                repair_hint=(
                    "Promote this failure into forbidden constraints and critical_fail_conditions."
                ),
            )
        )

    for failure in _state_failures(observed.state):
        issues.append(
            SemanticIssue(
                code="forbidden_violation",
                severity="fail",
                layer="observed_state.failure_modes",
                expected=failure,
                observed=failure,
                repair_hint=(
                    "Promote this observed object/contact failure into pose constraints, "
                    "object rules, and critical_fail_conditions."
                ),
            )
        )

    return SemanticDiffReport(
        gate_status="repair" if issues else "pass",
        issues=issues,
        observed_notes=list(observed.notes),
    )


def write_json(path: Path, payload: object) -> None:
    """Write pydantic model or plain payload as pretty JSON."""
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _props(layer: str, values: list[str], kind: str) -> list[RPEProposition]:
    return [
        RPEProposition(layer=layer, text=value, kind=kind)  # type: ignore[arg-type]
        for value in values
        if value.strip()
    ]


def _dedupe_props(props: list[RPEProposition]) -> list[RPEProposition]:
    seen: set[tuple[str, str, str]] = set()
    out: list[RPEProposition] = []
    for prop in props:
        key = (prop.layer, prop.kind, _normalize(prop.text))
        if key not in seen:
            out.append(prop)
            seen.add(key)
    return out


def _find_best_expected(text: str, candidates: list[RPEProposition]) -> RPEProposition | None:
    needle = _normalize(text)
    for candidate in candidates:
        haystack = _normalize(candidate.text)
        if needle in haystack or haystack in needle:
            return candidate
    for token in needle.split():
        if len(token) < 4:
            continue
        for candidate in candidates:
            if token in _normalize(candidate.text):
                return candidate
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _state_failures(state: dict) -> list[str]:
    """Extract explicit visual-state failures from ObservedRPE.state."""
    failures: list[str] = []
    for key in ("failure_modes", "physical_failures", "contact_graph_failures"):
        value = state.get(key)
        if isinstance(value, list):
            failures.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            failures.append(value)
    return _dedupe_strings(failures)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        key = _normalize(normalized)
        if normalized and key not in seen:
            out.append(normalized)
            seen.add(key)
    return out
