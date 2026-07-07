"""Declarative scenario rule tables + a single application engine for the planner.

Each of the planner's 4 post-processing scenarios (character lock, background
noise control, umbrella/katana object contact, katana reflection policy) used
to be implemented as one hand-written ``_apply_*``/``_build_*`` method pair per
touched SVP layer. That produced 4 near-identical "read constraints -> append
unique items -> model_copy" call chains.

This module collapses that pattern into data: a :class:`ScenarioContext`
(precomputed, scenario-agnostic flags), a :class:`LayerRule` (one declarative
instruction: which field to touch, what to append/set, and an optional
guard), and :func:`apply_scenario` (the single engine that walks a tuple of
``LayerRule`` and folds it onto an ``SVPVideo``).

``planner.py`` keeps ownership of prompt-detection (regex heuristics) and the
per-scenario early-return gates; this module owns only what the SVP looks like given the flags.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import SVPVideo


@dataclass(frozen=True)
class ScenarioContext:
    """Precomputed flags consumed by rule callables.

    A single shared shape is used across all 4 scenarios; only the fields
    relevant to the active scenario need to be populated, the rest keep their
    defaults.
    """

    locks: tuple[str, ...] = ()
    single_subject_intent: bool = False
    risk_flags: frozenset[str] = frozenset()
    detailed_background: bool = False
    background_forbidden: tuple[str, ...] = ()
    character_weapon_contact: bool = False
    reflection_forbidden: tuple[str, ...] = ()
    role_visual_cue_role: str | None = None


ListAppendSpec = tuple[str, ...] | Callable[[ScenarioContext], list[str]]
SetSpec = Any  # a literal scalar, or Callable[[ScenarioContext], Any]


@dataclass(frozen=True)
class LayerRule:
    """One declarative instruction: touch ``layer``, append/set some fields.

    ``layer`` is a dotted attribute path rooted at the ``SVPVideo`` being
    updated (e.g. ``"face_layer"``, ``"c3"``, ``"identity_locks"``).
    ``list_appends`` keys are dotted sub-paths *relative to* ``layer``
    (empty string ``""`` means "the layer path itself is the list", used for
    top-level list fields such as ``identity_locks``). Each value is either a
    static tuple of strings or a ``Callable[[ScenarioContext], list[str]]``
    computing the additions; additions are folded onto the existing list with
    :func:`_append_unique` (case-insensitive de-dupe, order-preserving).
    ``sets`` keys are dotted sub-paths for non-list (scalar) fields; each
    value is either a literal or a ``Callable[[ScenarioContext], Any]``.
    ``when``, if given, gates the whole rule: the rule is skipped entirely
    (leaving that field untouched) unless ``when(ctx)`` is true.
    """

    layer: str
    list_appends: Mapping[str, ListAppendSpec] = field(default_factory=dict)
    sets: Mapping[str, SetSpec] = field(default_factory=dict)
    when: Callable[[ScenarioContext], bool] | None = None


def _split_path(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        obj = getattr(obj, part)
    return obj


def _set_path(obj: Any, path: tuple[str, ...], value: Any) -> Any:
    if not path:
        return value
    head, *rest = path
    child = getattr(obj, head)
    new_child = _set_path(child, tuple(rest), value)
    return obj.model_copy(update={head: new_child})


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


def apply_scenario(svp: SVPVideo, rules: Sequence[LayerRule], ctx: ScenarioContext) -> SVPVideo:
    """Fold a tuple of :class:`LayerRule` onto ``svp`` in order.

    Unknown/misspelled ``layer`` or sub-paths fail fast with ``AttributeError``
    (via the underlying ``getattr``) rather than silently no-op-ing, since a
    typo'd rule path is a programmer error in the rule table, not a runtime
    data condition.
    """
    for rule in rules:
        if rule.when is not None and not rule.when(ctx):
            continue
        layer_path = _split_path(rule.layer)

        for subpath, spec in rule.list_appends.items():
            full_path = layer_path + _split_path(subpath)
            additions = spec(ctx) if callable(spec) else list(spec)
            current = list(_get_path(svp, full_path))
            updated = _append_unique(current, additions)
            svp = _set_path(svp, full_path, updated)

        for subpath, spec in rule.sets.items():
            full_path = layer_path + _split_path(subpath)
            value = spec(ctx) if callable(spec) else spec
            svp = _set_path(svp, full_path, value)

    return svp


# ---------------------------------------------------------------------------
# Rule-value helpers (conditional list construction extracted verbatim from
# the former _build_* methods so apply_scenario's callables stay one-liners).
# ---------------------------------------------------------------------------


def _character_locked_c3_forbidden(ctx: ScenarioContext) -> list[str]:
    has_female_lock = "female character" in ctx.locks
    has_male_lock = "male character" in ctx.locks
    items = ["collage, contact sheet, split panel, or numbered panel layout"]
    if ctx.single_subject_intent:
        if has_female_lock:
            items.append("male character if a female character was specified")
        if has_male_lock:
            items.append("female character if a male character was specified")
        items.extend(["extra characters", "duplicated background character"])
    return items


def _character_locked_c3_critical_fail(ctx: ScenarioContext) -> list[str]:
    items = [
        "subject gender changes",
        "hair color or hairstyle changes",
        "eye color changes",
        "outfit identity changes",
    ]
    if ctx.single_subject_intent:
        items.append("extra or duplicated characters appear")
    return items


def _character_locked_composition_required(ctx: ScenarioContext) -> list[str]:
    if ctx.single_subject_intent:
        return ["single primary character only", *ctx.locks]
    return list(ctx.locks)


def _character_locked_composition_forbidden(ctx: ScenarioContext) -> list[str]:
    items = ["collage layout", "multi-panel grid"]
    if ctx.single_subject_intent:
        items.extend(["extra characters", "duplicate character"])
    return items


def _background_depth_layers(
    risk_flags: set[str],
    *,
    single_subject_intent: bool,
    detailed_background: bool,
) -> list[str]:
    foreground = (
        "foreground: single character in sharp detail"
        if single_subject_intent
        else "foreground: requested subject(s) in sharp detail"
    )
    background = (
        "background: organized requested city detail kept subordinate to the PoR"
        if detailed_background
        else "background: simplified dark silhouettes with sparse soft light blocks"
    )
    layers = [foreground, background]
    if "wet_reflection" in risk_flags:
        layers.append("midground: broad smooth wet reflection bands")
    if "dense_city" in risk_flags and not detailed_background:
        layers.append("background: sparse neon blocks instead of dense signage")
    if "transparent_object" in risk_flags:
        layers.append("transparent objects stay clean and do not multiply background detail")
    return layers


def _background_forbidden_items(
    risk_flags: set[str],
    detailed_background: bool = False,
) -> list[str]:
    items = [
        "speckled light noise",
        "gritty background texture",
        "scratch-like background artifacts",
        "background silhouettes resembling the character",
    ]
    if not detailed_background:
        items.extend(["dense signage", "tiny readable text"])
    if "wet_reflection" in risk_flags:
        items.extend(
            [
                "fragmented noisy wet reflections",
                "small glitter-like reflection speckles",
            ]
        )
    if "weapon" in risk_flags:
        items.extend(
            [
                "weapon-like reflections in the background",
                "duplicated weapon silhouettes",
                "diagonal blade trails behind the character",
            ]
        )
    if "transparent_object" in risk_flags:
        items.extend(
            [
                "duplicated umbrella ribs",
                "transparent object filled with noisy background clutter",
            ]
        )
    return items


def _background_quality_rules(
    risk_flags: set[str],
    *,
    detailed_background: bool,
) -> list[str]:
    if detailed_background:
        rules = [
            "background detail remains organized and subordinate to character detail",
            "readable signs appear only where explicitly requested",
            "background micro-detail is grouped into clean blocks",
        ]
    else:
        rules = [
            "background acts as smooth lighting support",
            "simplified dark building silhouettes",
            "background detail stays subordinate to character detail",
        ]
    if "wet_reflection" in risk_flags:
        rules.append("broad smooth wet reflection bands")
    if "dense_city" in risk_flags:
        rules.append("sparse soft neon blocks")
    return rules


def _katana_reflection_forbidden_items() -> list[str]:
    return [
        "katana reflection on floor",
        "katana shadow or silhouette on wall",
        "blade-like line outside the waist sheath",
        "object-shaped katana floor reflection",
        "sharp linear katana reflection",
        "katana reflection on glass, umbrella, rain, floor, wall, or background",
    ]


def _background_noise_composition_required(ctx: ScenarioContext) -> list[str]:
    items = [
        "background acts as smooth lighting support, not the subject",
        "character detail has priority over background detail",
    ]
    if ctx.detailed_background:
        items.append("background detail remains organized and subordinate to the PoR")
    else:
        items.append("distant background uses sparse simplified shapes")
    return items


def _background_noise_style_required(ctx: ScenarioContext) -> list[str]:
    items = [
        "background uses broad smooth shapes and controlled gradients",
        "background micro-line density stays low",
    ]
    if "dense_city" in ctx.risk_flags:
        items.append("neon atmosphere is carried by large clean light blocks")
    if ctx.detailed_background:
        items.append("requested city detail is grouped into clean blocks")
    return items


def _background_noise_c3_required(ctx: ScenarioContext) -> list[str]:
    if ctx.detailed_background:
        return [
            (
                "Priority rule: character detail > requested background organization > "
                "lighting atmosphere > wet reflections"
            ),
            "background detail remains readable only where requested",
            "background remains subordinate to the PoR",
        ]
    return [
        (
            "Priority rule: character detail > lighting atmosphere > "
            "wet reflections > background simplicity"
        ),
        "background simplicity has higher priority than background detail",
        "background remains a clean support field for the PoR",
    ]


def _background_noise_reference_object_instance_rules(ctx: ScenarioContext) -> list[str]:
    if "weapon" in ctx.risk_flags:
        return [
            "no weapon-like reflections in the background",
            "no duplicated prop silhouettes",
        ]
    return []


# ---------------------------------------------------------------------------
# Scenario rule tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScenarioTables:
    character_lock: tuple[LayerRule, ...]
    background_noise: tuple[LayerRule, ...]
    umbrella_katana_contact: tuple[LayerRule, ...]
    katana_reflection: tuple[LayerRule, ...]


SCENARIOS = _ScenarioTables(
    character_lock=(
        LayerRule(
            layer="identity_locks",
            list_appends={"": lambda ctx: list(ctx.locks)},
        ),
        LayerRule(
            layer="face_layer",
            list_appends={
                "constraints.required": lambda ctx: list(ctx.locks),
                "constraints.forbidden": (
                    "wrong gender",
                    "wrong hair color",
                    "wrong eye color",
                    "different outfit",
                    "missing key weapon or prop",
                ),
                "distinctive_features": lambda ctx: list(ctx.locks),
            },
        ),
        LayerRule(
            layer="composition_layer",
            list_appends={
                "constraints.required": _character_locked_composition_required,
                "constraints.forbidden": _character_locked_composition_forbidden,
            },
        ),
        LayerRule(
            layer="c3",
            list_appends={
                "constraints.required": lambda ctx: list(ctx.locks),
                "constraints.forbidden": _character_locked_c3_forbidden,
                "evaluation_criteria.hit_list": lambda ctx: list(ctx.locks),
                "evaluation_criteria.critical_fail_conditions": _character_locked_c3_critical_fail,
            },
        ),
        LayerRule(
            layer="role_visual_cue",
            list_appends={"visual_elements": lambda ctx: list(ctx.locks)},
            sets={"role": lambda ctx: ctx.role_visual_cue_role or "character"},
        ),
        LayerRule(
            layer="variation_policy",
            sets={
                "clothing_variation": "none",
                "pose_variation": "minimal",
                "background_structure_variation": "minimal",
                "color_variation": "small",
            },
        ),
        LayerRule(
            layer="reference_usage_policy",
            list_appends={
                "use_reference_for": (
                    "character identity",
                    "hair color and hairstyle",
                    "eye color",
                    "outfit silhouette and pattern",
                    "weapon or held prop identity",
                ),
                "do_not_copy_from_reference": (
                    "reference background",
                    "collage or contact sheet layout",
                    "number labels",
                    "duplicate character poses",
                    "extra swords or weapon trails",
                    "texture noise",
                    "scratch-like line artifacts",
                    "compression-like speckles",
                ),
                "object_instance_rules": (
                    "katana count = exactly one when a katana is specified",
                    "katana location = character waist or character hand only",
                    "no sword-like shapes in the background",
                    "no diagonal blade reflections behind the character",
                    "no duplicated katana silhouettes",
                ),
                "background_quality_rules": (
                    "clean background matching the SVP scene context",
                    "smooth distant background details",
                    "reduced background micro-line density",
                    "no gritty texture noise",
                    "no scratch-like artifacts",
                    "no compression-like speckles",
                ),
            },
            sets={
                "background_source": "SVP prompt, not reference image",
                "identity_strength": "high",
                "scene_transfer_strength": "low",
            },
        ),
    ),
    background_noise=(
        LayerRule(
            layer="composition_layer",
            list_appends={
                "depth_layers": lambda ctx: _background_depth_layers(
                    set(ctx.risk_flags),
                    single_subject_intent=ctx.single_subject_intent,
                    detailed_background=ctx.detailed_background,
                ),
                "constraints.required": _background_noise_composition_required,
                "constraints.forbidden": lambda ctx: list(ctx.background_forbidden),
            },
        ),
        LayerRule(
            layer="style_layer",
            list_appends={
                "constraints.required": _background_noise_style_required,
                "constraints.forbidden": lambda ctx: list(ctx.background_forbidden),
            },
        ),
        LayerRule(
            layer="pose_layer",
            when=lambda ctx: "weapon" in ctx.risk_flags and ctx.character_weapon_contact,
            list_appends={
                "constraints.required": (
                    "main weapon is a single physical object",
                    "main weapon stays attached to the specified hand, waist, or contact point",
                ),
                "constraints.forbidden": (
                    "duplicated weapon",
                    "weapon trail",
                    "weapon-like background reflection",
                    "wrong weapon contact point",
                ),
                "contact_points": ("main weapon attached to character contact point",),
            },
        ),
        LayerRule(
            layer="c3",
            list_appends={
                "constraints.required": _background_noise_c3_required,
                "constraints.forbidden": lambda ctx: list(ctx.background_forbidden),
                "evaluation_criteria.critical_fail_conditions": (
                    "background detail overwhelms the character",
                    "background becomes gritty, speckled, or noisy",
                    "background contains character-like silhouettes",
                ),
            },
        ),
        LayerRule(
            layer="reference_usage_policy",
            list_appends={
                "do_not_copy_from_reference": lambda ctx: list(ctx.background_forbidden),
                "background_quality_rules": lambda ctx: _background_quality_rules(
                    set(ctx.risk_flags),
                    detailed_background=ctx.detailed_background,
                ),
                "object_instance_rules": _background_noise_reference_object_instance_rules,
            },
        ),
        LayerRule(
            layer="variation_policy",
            sets={
                "background_structure_variation": "minimal",
                "color_variation": "small",
            },
        ),
    ),
    umbrella_katana_contact=(
        LayerRule(
            layer="pose_layer",
            list_appends={
                "constraints.required": (
                    "one hand holds the umbrella handle",
                    "the other hand stays relaxed near the waist",
                    "katana is sheathed and attached to the waist belt",
                    "only the katana hilt and sheath may be visible",
                    "hands do not hold the katana while holding the umbrella",
                ),
                "constraints.forbidden": (
                    "floating katana",
                    "unsheathed blade",
                    "drawn katana",
                    "katana held while both hands hold umbrella",
                    "extra sword",
                    "sword-like background reflection",
                ),
                "contact_points": (
                    "one hand <-> umbrella handle",
                    "katana sheath <-> waist belt",
                ),
            },
            sets={
                "hand_state": (
                    "one hand holds the umbrella handle; the other hand stays relaxed "
                    "near the waist; no hand holds the katana"
                ),
            },
        ),
        LayerRule(
            layer="composition_layer",
            list_appends={
                "constraints.required": (
                    "umbrella and katana use separate non-conflicting contact points",
                    "katana remains fixed to the waist belt while the umbrella is held",
                ),
                "constraints.forbidden": (
                    "floating katana",
                    "extra sword",
                    "sword-like background reflection",
                ),
            },
        ),
        LayerRule(
            layer="c3",
            list_appends={
                "constraints.required": (
                    (
                        "Object-contact proposition: one hand holds umbrella; "
                        "katana remains sheathed at waist"
                    ),
                    "no extra sword-like object may appear outside the waist sheath",
                ),
                "constraints.forbidden": (
                    "floating katana",
                    "unsheathed blade",
                    "drawn katana",
                    "katana held while both hands hold umbrella",
                    "extra sword",
                    "sword-like background reflection",
                ),
                "evaluation_criteria.critical_fail_conditions": (
                    "katana floats away from waist sheath",
                    "both hands hold umbrella while katana appears unsheathed or floating",
                    "background produces sword-like reflections or extra blades",
                ),
            },
        ),
        LayerRule(
            layer="reference_usage_policy",
            list_appends={
                "object_instance_rules": (
                    "umbrella count = exactly one",
                    "katana count = exactly one",
                    "katana must be sheathed and attached to waist, not floating",
                    "katana may not be held if umbrella occupies a hand",
                    "no sword-like background reflection",
                ),
                "do_not_copy_from_reference": (
                    "floating swords from reference background",
                    "drawn blade pose unless explicitly requested",
                    "duplicate weapon silhouettes",
                ),
            },
        ),
    ),
    katana_reflection=(
        LayerRule(
            layer="pose_layer",
            list_appends={
                "constraints.required": (
                    "katana visible area is limited to physical waist hilt and sheath only",
                    "katana does not cast a distinct reflection, shadow, trail, or silhouette",
                ),
                "constraints.forbidden": (
                    "katana reflection on floor",
                    "katana shadow or silhouette on wall",
                    "blade-like line outside the waist sheath",
                    "object-shaped katana floor reflection",
                    "sharp linear katana reflection",
                ),
            },
        ),
        LayerRule(
            layer="composition_layer",
            list_appends={
                "constraints.forbidden": lambda ctx: list(ctx.reflection_forbidden),
                "constraints.required": (
                    (
                        "floor and background may show soft light only, never "
                        "object-shaped katana reflections"
                    ),
                ),
            },
        ),
        LayerRule(
            layer="style_layer",
            list_appends={
                "constraints.forbidden": lambda ctx: list(ctx.reflection_forbidden),
                "constraints.required": (
                    (
                        "katana-adjacent reflections are diffuse lighting patches, "
                        "not blade-shaped marks"
                    ),
                ),
            },
        ),
        LayerRule(
            layer="c3",
            list_appends={
                "constraints.forbidden": lambda ctx: list(ctx.reflection_forbidden),
                "constraints.required": (
                    (
                        "Object-instance rule: katana exists only as the physical waist "
                        "hilt/sheath; reflections must not duplicate it"
                    ),
                ),
                "evaluation_criteria.critical_fail_conditions": (
                    "katana appears anywhere except physical waist hilt/sheath",
                    "floor or background contains a blade-like reflection",
                    "katana reflection, shadow, trail, or silhouette reads as a second weapon",
                ),
            },
        ),
        LayerRule(
            layer="reference_usage_policy",
            list_appends={
                "object_instance_rules": (
                    "katana visible area is limited to physical waist hilt and sheath only",
                    "katana casts no distinct reflection, shadow, trail, or silhouette",
                    (
                        "no blade-like line may appear on floor, wall, glass, umbrella, "
                        "rain, or background"
                    ),
                ),
                "do_not_copy_from_reference": (
                    "weapon reflections from reference image",
                    "blade-like floor or background lines from reference image",
                ),
            },
        ),
    ),
)
