"""SVP.v4x-five-layer.video — Pydantic v2 schema for Structured Video Prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DEProfile(BaseModel):
    target_mean: float = Field(default=0.05, ge=0.0, le=1.0)
    tolerance: float = Field(default=0.03, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class LayerConstraints(BaseModel):
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class LayerBase(BaseModel):
    """Common structure shared by every layer. Subclasses add layer-specific fields."""

    por_core: list[str] = Field(default_factory=list)
    constraints: LayerConstraints = Field(default_factory=LayerConstraints)

    model_config = ConfigDict(extra="forbid")


class CompositionLayer(LayerBase):
    """Top-priority layer: camera angle, framing, depth."""

    camera_angle: Literal[
        "low_angle", "eye_level", "high_angle", "birds_eye", "dutch_angle"
    ]
    framing: Literal[
        "extreme_close_up",
        "close_up",
        "medium_close_up",
        "medium_shot",
        "medium_full_shot",
        "full_shot",
        "long_shot",
        "extreme_long_shot",
    ]
    # TODO(M3): When aspect_ratio is not "1:1" / "16:9" / "9:16",
    # gpt-image-2 does not accept it directly. image.py must translate
    # it via an ASPECT_TO_SIZE map before calling the image API.
    aspect_ratio: Literal[
        "auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"
    ] = "16:9"
    depth_layers: list[str] = Field(default_factory=list)


class FaceLayer(LayerBase):
    """Identity preservation."""

    expression: str
    eye_direction: str
    age_range: str | None = None
    distinctive_features: list[str] = Field(default_factory=list)


class StyleLayer(LayerBase):
    """Stylistic coherence."""

    line_density: Literal["low", "medium", "high"]
    specular_reflect: Literal["low", "medium", "high"]
    glow_radius: Literal["none", "narrow", "medium", "wide"]
    entropy: Literal["low", "medium", "high"]


class PoseLayer(LayerBase):
    """Hands, legs, contact points.

    For special poses, record priority bumps in ``constraints.notes``.
    """

    body_pose: str
    hand_state: str
    contact_points: list[str] = Field(default_factory=list)


class CameraMovement(BaseModel):
    type: Literal[
        "static",
        "dolly_in",
        "dolly_out",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "orbit",
        "handheld",
    ]
    speed: Literal["slow", "medium", "fast"] = "medium"

    model_config = ConfigDict(extra="forbid")


class SubjectMotion(BaseModel):
    subject: str
    action: str
    intensity: Literal["subtle", "moderate", "dramatic"] = "moderate"

    model_config = ConfigDict(extra="forbid")


class TemporalAnchor(BaseModel):
    time_range: str
    description: str

    model_config = ConfigDict(extra="forbid")


class MotionLayer(LayerBase):
    """The temporal semantic core — the fifth layer added for video."""

    duration_seconds: int = Field(ge=4, le=15)
    camera_movement: CameraMovement
    subject_motion: list[SubjectMotion] = Field(default_factory=list)
    temporal_anchors: list[TemporalAnchor] = Field(default_factory=list)


class GlobalConstraints(BaseModel):
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    motion_forbidden: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EvaluationCriteria(BaseModel):
    """Human-facing preservation criteria carried through generation logs/prompts."""

    hit_list: list[str] = Field(default_factory=list)
    por_similarity_target: float = Field(default=0.95, ge=0.0, le=1.0)
    delta_e_target: str = "0.03-0.06"
    critical_fail_conditions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class C3(BaseModel):
    context: str
    constraints: GlobalConstraints
    consistency: list[str] = Field(default_factory=list)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)

    model_config = ConfigDict(extra="forbid")


class RoleVisualCue(BaseModel):
    """Visual role cues borrowed from the canonical illustration SVP template."""

    role: str | None = None
    visual_elements: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VariationPolicy(BaseModel):
    """Allowed variation range for identity-preserving generation."""

    camera_offset_variation: str = "small"
    clothing_variation: str = "small"
    expression_range: str = "minimal"
    pose_variation: str = "minimal"
    background_structure_variation: str = "minimal"
    color_variation: str = "small"

    model_config = ConfigDict(extra="forbid")


class ReferenceUsagePolicy(BaseModel):
    """How an optional reference image should and should not influence generation."""

    use_reference_for: list[str] = Field(
        default_factory=lambda: [
            "character identity",
            "hair color and hairstyle",
            "eye color",
            "outfit silhouette and pattern",
            "overall character color palette",
        ]
    )
    do_not_copy_from_reference: list[str] = Field(
        default_factory=lambda: [
            "background",
            "panel layout",
            "number labels",
            "duplicate character poses",
            "extra swords or weapon trails",
            "compression artifacts",
            "texture noise",
        ]
    )
    background_source: str = "SVP prompt, not reference image"
    identity_strength: Literal["low", "medium", "high"] = "high"
    scene_transfer_strength: Literal["low", "medium", "high"] = "low"
    object_instance_rules: list[str] = Field(
        default_factory=lambda: [
            "main weapon or prop count must match the prompt",
            "main weapon or prop must stay attached to the character",
            "no weapon-like silhouettes in the background",
        ]
    )
    background_quality_rules: list[str] = Field(
        default_factory=lambda: [
            "clean cinematic background",
            "smooth distant background details",
            "reduced distant micro-line density",
            "no scratch-like line artifacts",
            "no compression-like speckles",
        ]
    )

    model_config = ConfigDict(extra="forbid")


class Axes(BaseModel):
    composition: str
    light_air: str
    expression: str
    stroke: str
    motion: str
    material: str
    narrative: str
    emotion_symbol: str

    model_config = ConfigDict(extra="forbid")


class SVPVideo(BaseModel):
    schema_version: Literal["SVP.v4x-five-layer.video"] = "SVP.v4x-five-layer.video"

    por_identity: str = Field(min_length=10, max_length=500)
    por_core: list[str] = Field(min_length=3, max_length=6)
    grv_anchor: list[str] = Field(min_length=2, max_length=4)
    identity_locks: list[str] = Field(default_factory=list)
    de_profile: DEProfile

    role_visual_cue: RoleVisualCue = Field(default_factory=RoleVisualCue)
    variation_policy: VariationPolicy = Field(default_factory=VariationPolicy)
    reference_usage_policy: ReferenceUsagePolicy = Field(default_factory=ReferenceUsagePolicy)
    layer_priority: list[str] = Field(
        default_factory=lambda: [
            "composition_layer",
            "face_layer",
            "style_layer",
            "pose_layer",
            "motion_layer",
        ]
    )

    composition_layer: CompositionLayer
    face_layer: FaceLayer
    style_layer: StyleLayer
    pose_layer: PoseLayer
    motion_layer: MotionLayer

    style_family: str
    style_pack: str | None = None
    color_axis: list[str] = Field(min_length=1)
    texture_axis: list[str] = Field(min_length=1)

    c3: C3
    axes: Axes

    model_config = ConfigDict(extra="forbid")
