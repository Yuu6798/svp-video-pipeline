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


def render_image_prompt(svp: SVPVideo) -> str:
    """Render image-layer prompt text from SVPVideo.

    The image renderer intentionally excludes ``motion_layer`` fields.
    """

    por_core_text = ", ".join(svp.por_core)
    grv_anchor_text = ", ".join(svp.grv_anchor)
    required_items = _collect_required(svp)
    avoid_items = _collect_forbidden(svp)
    avoid_text = ", ".join(avoid_items) if avoid_items else "None"

    lines: list[str] = [
        "# Image Generation Brief",
        "",
        "## Essence (PoR)",
        f"This image is defined by: {por_core_text}. Without these, the image is not that image.",
        "",
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
        "## Color & Texture",
        f"- Colors: {', '.join(svp.color_axis)}",
        f"- Textures: {', '.join(svp.texture_axis)}",
        "",
        "## Context",
        svp.c3.context,
        "",
        "## Consistency Rules",
    ]

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
            "## Avoid",
            f"Avoid: {avoid_text}",
        ]
    )
    return "\n".join(lines).strip()
