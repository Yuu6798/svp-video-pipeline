"""Image-to-RPE audit helpers.

This module turns a generated image artifact into an ``ObservedRPE`` JSON file.
The first useful mode is manual/Codex-assisted: a human or Codex records state
variables as observed/missing/violating propositions, then the existing repair
pipeline turns those findings into a proposed SVP. An optional OpenAI vision
backend is provided behind the same interface for later automation.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI, OpenAIError

from ..exceptions import ImageAPIError
from ..schema import SVPVideo
from ..utils.timestamps import utc_now_compact
from .models import ExpectedRPE, ObservedRPE
from .rpe import extract_expected_rpe, write_json
from .visual_compare import copy_source_image_to_packet, validate_image_file

ImageAuditBackend = Literal["manual", "openai"]


@dataclass
class ImageAuditResult:
    """Files written by an image audit pass."""

    output_dir: Path
    source_image_path: Path
    expected_rpe_path: Path
    observed_rpe_path: Path
    observed_rpe: ObservedRPE
    expected_rpe: ExpectedRPE


@dataclass
class _ImageAuditContext:
    """Prepared inputs shared across image audit backends."""

    output_dir: Path
    source_image_path: Path
    svp_path: Path
    svp: SVPVideo
    expected_rpe: ExpectedRPE


@dataclass
class _ManualAuditFindings:
    """Caller-supplied findings used by manual/Codex-assisted audit mode."""

    observed: list[str]
    missing: list[str]
    violations: list[str]
    notes: list[str]
    object_states: list[str]
    contact_graph: list[str]
    viewer_contact_graph: list[str]
    anatomical_contact_graph: list[str]
    pose_intent: str | None
    failure_modes: list[str]


def audit_image_to_observed_rpe(
    *,
    svp_path: Path,
    image_path: Path,
    output_root: Path,
    backend: ImageAuditBackend | str = "manual",
    observed: list[str] | None = None,
    missing: list[str] | None = None,
    violations: list[str] | None = None,
    notes: list[str] | None = None,
    object_states: list[str] | None = None,
    contact_graph: list[str] | None = None,
    viewer_contact_graph: list[str] | None = None,
    anatomical_contact_graph: list[str] | None = None,
    pose_intent: str | None = None,
    failure_modes: list[str] | None = None,
    model: str = "gpt-4.1-mini",
    client: OpenAI | None = None,
) -> ImageAuditResult:
    """Audit an image against a target SVP and write ``observed_rpe.json``.

    ``manual`` mode does not call an API. It normalizes caller-provided findings
    into the same schema used by the repair step. ``openai`` mode asks a vision
    model to produce the same JSON shape.
    """
    context = _prepare_image_audit_context(
        svp_path=svp_path,
        image_path=image_path,
        output_root=output_root,
    )
    findings = _build_manual_audit_findings(
        observed=observed,
        missing=missing,
        violations=violations,
        notes=notes,
        object_states=object_states,
        contact_graph=contact_graph,
        viewer_contact_graph=viewer_contact_graph,
        anatomical_contact_graph=anatomical_contact_graph,
        pose_intent=pose_intent,
        failure_modes=failure_modes,
    )
    observed_rpe = _observed_rpe_from_backend(
        context=context,
        backend=backend,
        findings=findings,
        model=model,
        client=client,
    )
    return _write_image_audit_result(
        context=context,
        observed_rpe=observed_rpe,
    )


def _build_manual_audit_findings(
    *,
    observed: list[str] | None,
    missing: list[str] | None,
    violations: list[str] | None,
    notes: list[str] | None,
    object_states: list[str] | None,
    contact_graph: list[str] | None,
    viewer_contact_graph: list[str] | None,
    anatomical_contact_graph: list[str] | None,
    pose_intent: str | None,
    failure_modes: list[str] | None,
) -> _ManualAuditFindings:
    return _ManualAuditFindings(
        observed=observed or [],
        missing=missing or [],
        violations=violations or [],
        notes=notes or [],
        object_states=object_states or [],
        contact_graph=contact_graph or [],
        viewer_contact_graph=viewer_contact_graph or [],
        anatomical_contact_graph=anatomical_contact_graph or [],
        pose_intent=pose_intent,
        failure_modes=failure_modes or [],
    )


def _prepare_image_audit_context(
    *,
    svp_path: Path,
    image_path: Path,
    output_root: Path,
) -> _ImageAuditContext:
    svp_path = Path(svp_path)
    image_path = Path(image_path)
    validate_image_file(image_path)
    output_dir = _make_audit_dir(Path(output_root))
    packet_image_path = copy_source_image_to_packet(image_path, output_dir)
    svp = SVPVideo.model_validate_json(svp_path.read_text(encoding="utf-8"))
    expected = extract_expected_rpe(svp)
    return _ImageAuditContext(
        output_dir=output_dir,
        source_image_path=packet_image_path,
        svp_path=svp_path,
        svp=svp,
        expected_rpe=expected,
    )


def _observed_rpe_from_backend(
    *,
    context: _ImageAuditContext,
    backend: ImageAuditBackend | str,
    findings: _ManualAuditFindings,
    model: str,
    client: OpenAI | None,
) -> ObservedRPE:
    if backend == "manual":
        return _manual_observed_rpe(
            svp=context.svp,
            svp_path=context.svp_path,
            image_path=context.source_image_path,
            findings=findings,
        )
    if backend == "openai":
        return _openai_observed_rpe(
            svp=context.svp,
            svp_path=context.svp_path,
            expected=context.expected_rpe,
            image_path=context.source_image_path,
            model=model,
            client=client,
        )
    raise ValueError(f"Unknown image audit backend: {backend!r}")


def _write_image_audit_result(
    *,
    context: _ImageAuditContext,
    observed_rpe: ObservedRPE,
) -> ImageAuditResult:
    expected_rpe_path = context.output_dir / "expected_rpe.json"
    observed_rpe_path = context.output_dir / "observed_rpe.json"
    write_json(expected_rpe_path, context.expected_rpe)
    write_json(observed_rpe_path, observed_rpe)
    return ImageAuditResult(
        output_dir=context.output_dir,
        source_image_path=context.source_image_path,
        expected_rpe_path=expected_rpe_path,
        observed_rpe_path=observed_rpe_path,
        observed_rpe=observed_rpe,
        expected_rpe=context.expected_rpe,
    )


def _manual_observed_rpe(
    *,
    svp: SVPVideo,
    svp_path: Path,
    image_path: Path,
    findings: _ManualAuditFindings,
) -> ObservedRPE:
    return ObservedRPE(
        artifact=str(image_path),
        observed=_dedupe(findings.observed),
        missing=_dedupe(findings.missing),
        violations=_dedupe(findings.violations),
        notes=_dedupe(
            [
                *findings.notes,
                "manual image audit; state variables supplied by caller",
            ]
        ),
        state=_manual_observed_state(
            svp=svp,
            svp_path=svp_path,
            image_path=image_path,
            findings=findings,
        ),
    )


def _manual_observed_state(
    *,
    svp: SVPVideo,
    svp_path: Path,
    image_path: Path,
    findings: _ManualAuditFindings,
) -> dict[str, Any]:
    state = _state_scaffold(svp=svp, svp_path=svp_path, image_path=image_path)
    if findings.object_states:
        state["object_graph"] = _dedupe(findings.object_states)
    if findings.contact_graph:
        state["contact_graph"] = _dedupe(findings.contact_graph)
    if findings.viewer_contact_graph:
        state["viewer_contact_graph"] = _dedupe(findings.viewer_contact_graph)
    if findings.anatomical_contact_graph:
        state["anatomical_contact_graph"] = _dedupe(findings.anatomical_contact_graph)
    if findings.pose_intent and findings.pose_intent.strip():
        state["pose_intent"] = findings.pose_intent.strip()
    if findings.failure_modes:
        state["failure_modes"] = _dedupe(findings.failure_modes)
    return state


def _openai_observed_rpe(
    *,
    svp: SVPVideo,
    svp_path: Path,
    expected: ExpectedRPE,
    image_path: Path,
    model: str,
    client: OpenAI | None,
) -> ObservedRPE:
    client = client or OpenAI()
    prompt = _audit_prompt(svp=svp, expected=expected)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
    except OpenAIError as exc:
        raise ImageAPIError("image audit backend failed") from exc

    content = response.choices[0].message.content
    if not content:
        raise ImageAPIError("image audit backend returned an empty response")
    payload = json.loads(content)
    payload["artifact"] = str(image_path)
    payload.setdefault("state", {})
    payload["state"] = {
        **_state_scaffold(svp=svp, svp_path=svp_path, image_path=image_path),
        **payload["state"],
    }
    return ObservedRPE.model_validate(payload)


def _audit_prompt(*, svp: SVPVideo, expected: ExpectedRPE) -> str:
    required = "\n".join(f"- {prop.text}" for prop in expected.required[:40])
    forbidden = "\n".join(f"- {prop.text}" for prop in expected.forbidden[:40])
    return f"""Audit the image as state variables for SVP regeneration.

Return strict JSON with these keys only:
- artifact: string
- observed: string[]
- missing: string[]
- violations: string[]
- notes: string[]
- state: object

Focus on identity, objects, contact graph, background, style, and physical
consistency. Do not write general art critique. Write propositions that can be
fed back into a structured prompt.

Target identity:
{svp.por_identity}

Expected required propositions:
{required}

Forbidden / critical failure propositions:
{forbidden}

State keys to fill when visible:
identity, object_graph, viewer_contact_graph, anatomical_contact_graph,
contact_graph, pose_intent, background, style, failure_modes, physical_failures.

For weapon/tool poses, be explicit about object roles and contacts. Example:
- object_graph: [
  "katana_blade: single glowing drawn blade",
  "scabbard: separate dark sheath at upper-left"
]
- contact_graph: ["right_hand -> katana_handle", "left_hand -> scabbard"]
- viewer_contact_graph: ["viewer_left_hand -> katana_handle"]
- anatomical_contact_graph: ["character_right_hand -> katana_handle"]
- pose_intent: "unsheathing / draw-pose"
- failure_modes: ["left hand grips the wrong object", "blade and scabbard are fused"]
"""


def _state_scaffold(*, svp: SVPVideo, svp_path: Path, image_path: Path) -> dict[str, Any]:
    return {
        "target_svp": str(svp_path),
        "image": str(image_path),
        "expected_identity": [
            svp.por_identity,
            *svp.identity_locks,
            *svp.face_layer.distinctive_features,
        ],
        "expected_objects": list(svp.pose_layer.contact_points),
        "expected_background": [
            svp.c3.context,
            *svp.composition_layer.depth_layers,
            *svp.reference_usage_policy.background_quality_rules,
        ],
        "expected_forbidden": [
            *svp.c3.constraints.forbidden,
            *svp.c3.evaluation_criteria.critical_fail_conditions,
            *svp.reference_usage_policy.do_not_copy_from_reference,
            *svp.reference_usage_policy.object_instance_rules,
        ],
    }


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _make_audit_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"audit-{utc_now_compact()}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}-{index:02d}"
        index += 1
    candidate.mkdir()
    return candidate


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            out.append(normalized)
            seen.add(key)
    return out
