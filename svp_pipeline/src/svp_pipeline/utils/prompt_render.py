"""Render SVP into a structured prompt for image generation."""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import SVPVideo


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _is_no_subject(*values: str | None) -> bool:
    joined = " ".join(v for v in values if v).lower()
    return "no subject" in joined or joined.strip().startswith("n/a")


def _render_face_block(svp: SVPVideo) -> list[str]:
    face = svp.face_layer
    if _is_no_subject(face.expression, face.eye_direction):
        return ["No human subject (still life / landscape)."]

    lines = [
        f"- Expression: {face.expression}",
        f"- Eye direction: {face.eye_direction}",
    ]
    if face.age_range:
        lines.append(f"- Age range: {face.age_range}")
    if face.distinctive_features:
        lines.append(f"- Distinctive features: {', '.join(face.distinctive_features)}")
    return lines


def _render_pose_block(svp: SVPVideo) -> list[str]:
    pose = svp.pose_layer
    if _is_no_subject(pose.body_pose, pose.hand_state):
        return ["No human pose constraints (no subject)."]

    lines = [
        f"- Body pose: {pose.body_pose}",
        f"- Hand state: {pose.hand_state}",
    ]
    if pose.contact_points:
        lines.append(f"- Contact points: {', '.join(pose.contact_points)}")
    return lines


def _collect_forbidden(svp: SVPVideo) -> list[str]:
    merged: list[str] = []
    merged.extend(svp.composition_layer.constraints.forbidden)
    merged.extend(svp.face_layer.constraints.forbidden)
    merged.extend(svp.style_layer.constraints.forbidden)
    merged.extend(svp.pose_layer.constraints.forbidden)
    merged.extend(svp.c3.constraints.forbidden)
    merged.extend(svp.c3.evaluation_criteria.critical_fail_conditions)
    return _dedupe_keep_order(merged)


def _collect_required(svp: SVPVideo) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    layer_required = [
        ("composition", svp.composition_layer.constraints.required),
        ("face", svp.face_layer.constraints.required),
        ("style", svp.style_layer.constraints.required),
        ("pose", svp.pose_layer.constraints.required),
        ("global", svp.c3.constraints.required),
    ]
    for layer, required_items in layer_required:
        for item in required_items:
            normalized = item.strip()
            if normalized:
                items.append((layer, normalized))
    return items


def _collect_critical_pose_geometry(svp: SVPVideo) -> dict[str, list[str]]:
    """Extract RPE-derived object/contact geometry and expand it for image models."""
    object_states = _filter_prefixed(
        svp.c3.constraints.required,
        "Object state must be preserved:",
    )
    contact_graph = _filter_prefixed(
        svp.c3.constraints.required,
        "Contact graph must be preserved:",
    )
    viewer_contact_graph = _filter_prefixed(
        svp.c3.constraints.required,
        "Viewer contact graph must be preserved:",
    )
    anatomical_contact_graph = _filter_prefixed(
        svp.c3.constraints.required,
        "Anatomical contact graph must be preserved:",
    )
    pose_intents = _filter_prefixed(
        svp.pose_layer.contact_points,
        "RPE pose intent:",
    )
    contact_points = _filter_prefixed(
        svp.pose_layer.contact_points,
        "RPE contact graph:",
    )
    viewer_contact_points = _filter_prefixed(
        svp.pose_layer.contact_points,
        "RPE viewer contact graph:",
    )
    anatomical_contact_points = _filter_prefixed(
        svp.pose_layer.contact_points,
        "RPE anatomical contact graph:",
    )
    failures = _filter_prefixed(
        [
            *svp.pose_layer.constraints.forbidden,
            *svp.c3.evaluation_criteria.critical_fail_conditions,
        ],
        "Observed visual-state failure must not recur:",
    )
    observed_failures = _filter_prefixed(
        svp.c3.evaluation_criteria.critical_fail_conditions,
        "Observed failure:",
    )
    failures = _dedupe_keep_order([*failures, *observed_failures])
    expanded = _expand_pose_geometry(
        object_states=object_states,
        contact_graph=_dedupe_keep_order([*contact_graph, *contact_points]),
        viewer_contact_graph=_dedupe_keep_order(
            [*viewer_contact_graph, *viewer_contact_points]
        ),
        anatomical_contact_graph=_dedupe_keep_order(
            [*anatomical_contact_graph, *anatomical_contact_points]
        ),
        pose_intents=pose_intents,
        failures=failures,
    )
    if not any(expanded.values()):
        return {}
    return expanded


def _filter_prefixed(items: Iterable[str], prefix: str) -> list[str]:
    values: list[str] = []
    for item in items:
        stripped = item.strip()
        if stripped.startswith(prefix):
            values.append(stripped.removeprefix(prefix).strip())
    return _dedupe_keep_order(values)


def _expand_pose_geometry(
    *,
    object_states: list[str],
    contact_graph: list[str],
    viewer_contact_graph: list[str],
    anatomical_contact_graph: list[str],
    pose_intents: list[str],
    failures: list[str],
) -> dict[str, list[str]]:
    required: list[str] = []
    avoid: list[str] = []
    if pose_intents:
        joined = "; ".join(pose_intents)
        required.append(f"Pose intent: {joined}. This pose intent overrides generic grip poses.")
        if _mentions_any(joined, ("unsheathing", "draw-pose", "draw pose")):
            required.append(
                "This is an unsheathing / draw-pose, not a two-handed guard stance."
            )
            avoid.append("Do not convert the draw-pose into both hands gripping the same blade.")

    for item in object_states:
        required.append(f"Object role: {item}.")
        if _mentions_any(item, ("scabbard", "sheath")):
            required.append(
                "The scabbard/sheath must remain a separate dark object, not fused with the blade."
            )
        if _mentions_any(item, ("katana_blade", "blade")):
            required.append(
                "The katana blade must read as one single drawn glowing blade."
            )

    for item in contact_graph:
        required.append(f"Required contact relation: {item}.")
        lowered = item.lower()
        if "left_hand" in lowered and _mentions_any(lowered, ("scabbard", "sheath")):
            required.append(
                "Left hand grips only the separate scabbard/sheath; it does not grip the "
                "glowing blade or the same handle as the right hand."
            )
        if "right_hand" in lowered and _mentions_any(lowered, ("handle", "katana")):
            required.append("Right hand grips the katana handle near the guard.")

    for item in viewer_contact_graph:
        required.append(
            f"Viewer-relative contact relation: {item}. "
            "This describes where the contact appears in the image, not anatomy."
        )
        lowered = item.lower()
        if "viewer_left" in lowered:
            required.append(
                "On the viewer-left side of the image, preserve this hand/object contact."
            )
        if "viewer_right" in lowered:
            required.append(
                "On the viewer-right side of the image, preserve this hand/object contact."
            )

    for item in anatomical_contact_graph:
        required.append(
            f"Anatomical contact relation: {item}. "
            "This describes the character's body side, not screen position."
        )

    if viewer_contact_graph or anatomical_contact_graph:
        required.append(
            "Do not swap viewer-left/viewer-right with anatomical-left/anatomical-right; "
            "keep both coordinate systems consistent."
        )
        avoid.append("Do not mirror or swap the specified hand/object roles.")

    for item in failures:
        avoid.append(item)

    return {
        "required": _dedupe_keep_order(required),
        "avoid": _dedupe_keep_order(avoid),
    }


def _mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _collect_motion_forbidden(svp: SVPVideo) -> list[str]:
    merged: list[str] = []
    merged.extend(svp.motion_layer.constraints.forbidden)
    merged.extend(svp.c3.constraints.forbidden)
    merged.extend(svp.c3.constraints.motion_forbidden)
    return _dedupe_keep_order(merged)


def _collect_motion_required(svp: SVPVideo) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    layer_required = [
        ("motion", svp.motion_layer.constraints.required),
        ("global", svp.c3.constraints.required),
    ]
    for layer, required_items in layer_required:
        for item in required_items:
            normalized = item.strip()
            if normalized:
                items.append((layer, normalized))
    return items


def render_image_prompt(svp: SVPVideo) -> str:
    """Render image-layer prompt text from SVPVideo.

    The image renderer intentionally excludes ``motion_layer`` fields.
    """

    por_core_text = ", ".join(svp.por_core)
    grv_anchor_text = ", ".join(svp.grv_anchor)
    required_items = _collect_required(svp)
    critical_pose_geometry = _collect_critical_pose_geometry(svp)
    avoid_items = _collect_forbidden(svp)
    avoid_text = ", ".join(avoid_items) if avoid_items else "None"

    lines: list[str] = [
        "# Image Generation Brief",
        "",
        "## Essence (PoR)",
        f"This image is defined by: {por_core_text}. Without these, the image is not that image.",
        "",
    ]

    if svp.identity_locks:
        lines.extend(
            [
                "## Identity Lock",
                (
                    "Preserve these subject-defining details exactly. Do not generalize, "
                    "swap, or reinterpret them:"
                ),
                *(f"- {item}" for item in svp.identity_locks),
                "",
            ]
        )

    lines.extend(
        [
        "## Subject Identity",
        svp.por_identity,
        "",
        "## Composition",
        f"- Camera angle: {svp.composition_layer.camera_angle}",
        f"- Framing: {svp.composition_layer.framing}",
        (
            "- Depth layers: "
            + (
                ", ".join(svp.composition_layer.depth_layers)
                if svp.composition_layer.depth_layers
                else "None"
            )
        ),
        f"- Visual focus (grv_anchor): {grv_anchor_text}",
        "",
        "## Face",
        *_render_face_block(svp),
        "",
        "## Style",
        f"- Line density: {svp.style_layer.line_density}",
        f"- Specular: {svp.style_layer.specular_reflect}",
        f"- Glow radius: {svp.style_layer.glow_radius}",
        f"- Entropy: {svp.style_layer.entropy}",
        "",
        "## Pose",
        *_render_pose_block(svp),
        "",
        ]
    )

    if critical_pose_geometry:
        lines.extend(
            [
                "## Critical Pose Geometry",
                (
                    "These RPE-derived object/contact constraints are high priority. "
                    "They override generic pose interpretation:"
                ),
            ]
        )
        lines.extend(f"- {item}" for item in critical_pose_geometry.get("required", []))
        if critical_pose_geometry.get("avoid"):
            lines.extend(
                [
                    "",
                    "Critical pose failures to avoid:",
                    *(f"- {item}" for item in critical_pose_geometry["avoid"]),
                ]
            )
        lines.append("")

    lines.extend(
        [
        "## Color & Texture",
        f"- Colors: {', '.join(svp.color_axis)}",
        f"- Textures: {', '.join(svp.texture_axis)}",
        "",
        "## Context",
        svp.c3.context,
        "",
        "## Consistency Rules",
        ]
    )

    if svp.c3.consistency:
        lines.extend(f"- {item}" for item in svp.c3.consistency)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Evaluation Hit List",
        ]
    )
    if svp.c3.evaluation_criteria.hit_list:
        lines.extend(f"- {item}" for item in svp.c3.evaluation_criteria.hit_list)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Required Constraints",
        ]
    )
    if required_items:
        lines.extend(f"- [{layer}] {item}" for layer, item in required_items)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Avoid",
            f"Avoid: {avoid_text}",
        ]
    )
    return "\n".join(lines).strip()


def append_reference_usage_policy(prompt: str, svp: SVPVideo) -> str:
    """Append instructions that constrain how optional reference images are used."""
    policy = svp.reference_usage_policy
    lines = [
        prompt.strip(),
        "",
        "## Reference Image Usage Policy",
        (
            "If a reference image is provided, use it only for these character "
            "identity details:"
        ),
        *(f"- {item}" for item in policy.use_reference_for),
        "",
        "Do not copy these elements from the reference image:",
        *(f"- {item}" for item in policy.do_not_copy_from_reference),
        "",
        f"Background source: {policy.background_source}.",
        f"Identity transfer strength: {policy.identity_strength}.",
        f"Scene/background transfer strength: {policy.scene_transfer_strength}.",
        "",
        "Object instance rules:",
        *(f"- {item}" for item in policy.object_instance_rules),
        "",
        "Background quality rules:",
        *(f"- {item}" for item in policy.background_quality_rules),
    ]
    if (
        policy.failure_preset_ids
        or policy.positive_anchors
        or policy.negative_anchors
        or policy.render_priority
        or policy.conflict_resolution
    ):
        lines.extend(
            [
                "",
                "## Failure Prevention Presets",
                "Active preset ids:",
                *(
                    f"- {preset_id}"
                    for preset_id in (policy.failure_preset_ids or ["none"])
                ),
            ]
        )
        if policy.positive_anchors:
            lines.extend(["", "Positive anchors to preserve:"])
            lines.extend(f"- {item}" for item in policy.positive_anchors)
        if policy.negative_anchors:
            lines.extend(["", "Negative anchors to suppress:"])
            lines.extend(f"- {item}" for item in policy.negative_anchors)
        if policy.render_priority:
            lines.extend(["", "Render priority:"])
            lines.append(" > ".join(policy.render_priority))
        if policy.conflict_resolution:
            lines.extend(["", "Conflict resolution rules:"])
            lines.extend(f"- {item}" for item in policy.conflict_resolution)
    lines.append(
        "Generate a single final image, not a collage, contact sheet, "
        "split panel, or numbered panel layout."
    )
    return "\n".join(lines).strip()


def render_motion_prompt(svp: SVPVideo) -> str:
    """Render motion-layer prompt text from SVPVideo for Seedance r2v."""

    avoid_items = _collect_motion_forbidden(svp)
    required_items = _collect_motion_required(svp)
    avoid_text = ", ".join(avoid_items) if avoid_items else "None"

    lines: list[str] = [
        "# Video Generation Brief",
        "",
        "## Reference",
        (
            "Use @Image1 as the primary visual reference. Preserve subject identity, "
            "composition, lighting, and style continuity through the full clip."
        ),
        "",
        "## Essence to Preserve (PoR)",
        "Throughout the video, keep these semantic core elements visible:",
    ]
    lines.extend(f"- {item}" for item in svp.por_core)

    lines.extend(
        [
            "",
            "## Visual Anchors (grv)",
            "The camera focus should stay on:",
        ]
    )
    lines.extend(f"- {item}" for item in svp.grv_anchor)

    camera = svp.motion_layer.camera_movement
    lines.extend(
        [
            "",
            "## Camera Movement",
            f"- Type: {camera.type}",
            f"- Speed: {camera.speed}",
        ]
    )

    lines.extend(
        [
            "",
            "## Subject Motion",
        ]
    )
    if svp.motion_layer.subject_motion:
        for movement in svp.motion_layer.subject_motion:
            lines.extend(
                [
                    f"- Subject: {movement.subject}",
                    f"  Action: {movement.action}",
                    f"  Intensity: {movement.intensity}",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Timeline",
        ]
    )
    if svp.motion_layer.temporal_anchors:
        lines.extend(
            f"- {anchor.time_range}: {anchor.description}"
            for anchor in svp.motion_layer.temporal_anchors
        )
    else:
        lines.append("- None")

    style_pack = svp.style_pack or "None"
    lines.extend(
        [
            "",
            "## Duration",
            f"{svp.motion_layer.duration_seconds} seconds",
            "",
            "## Style Continuity",
            f"Maintain the style defined by: {svp.style_family} / {style_pack}",
            "",
            "## Consistency Rules",
        ]
    )
    if svp.c3.consistency:
        lines.extend(f"- {item}" for item in svp.c3.consistency)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Required Constraints",
        ]
    )
    if required_items:
        lines.extend(f"- [{layer}] {item}" for layer, item in required_items)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Continuity Guardrails",
            "- Keep PoR core elements visible throughout the timeline.",
            "- Keep primary grv anchors inside the frame throughout the timeline.",
            "",
            "## Avoid",
            f"Avoid: {avoid_text}",
        ]
    )
    return "\n".join(lines).strip()
