"""Planner that converts natural language prompts into validated SVPVideo objects."""

from __future__ import annotations

import re
from importlib import resources
from typing import Any, Literal

import anthropic
from anthropic import Anthropic
from pydantic import ValidationError

from ..exceptions import PlannerAPIError, PlannerSchemaError
from ..schema import SVPVideo

PlannerModel = Literal["claude-opus-4-7", "claude-opus-4-6", "claude-haiku-4-5"]


class Planner:
    """Natural-language prompt -> SVPVideo planner."""

    DEFAULT_MODEL: PlannerModel = "claude-opus-4-7"
    TOOL_NAME = "generate_svp_video"
    _MODEL_ALIASES: dict[str, str] = {
        "claude-opus-4-7": "claude-opus-4-6",
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-haiku-4-5": "claude-haiku-4-5",
    }
    REQUIRED_MOTION_FORBIDDEN: tuple[str, ...] = (
        "PoR_core要素のフレームアウト",
        "grv_anchor主要要素の画面外移動",
    )

    def __init__(
        self,
        model: PlannerModel | str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Anthropic | Any | None = None,
        character_lock: bool = True,
    ) -> None:
        if model not in self._MODEL_ALIASES:
            raise ValueError(f"unsupported planner model: {model}")
        self.requested_model = model
        self.model = self._MODEL_ALIASES[model]
        self.character_lock = character_lock
        self._client = client if client is not None else Anthropic(api_key=api_key)
        self._system_prompt = _load_system_prompt()

    def plan(
        self,
        user_prompt: str,
        duration: int | None = None,
    ) -> SVPVideo:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_user_prompt(user_prompt=user_prompt, duration=duration),
            }
        ]

        for attempt in range(2):
            response = self._call_messages_api(messages=messages)
            tool_use_block, tool_input = self._extract_tool_input(response)

            try:
                svp = SVPVideo.model_validate(tool_input)
            except ValidationError as exc:
                if attempt == 0:
                    self._append_retry_messages(
                        messages=messages,
                        response=response,
                        fallback_tool_use_id=getattr(tool_use_block, "id", "tool_use_retry"),
                        validation_error=exc,
                    )
                    continue
                raise PlannerSchemaError("planner output failed schema validation twice") from exc

            svp = self._inject_motion_forbidden(svp)
            if self.character_lock:
                svp = self._apply_character_locks(svp=svp, user_prompt=user_prompt)
            svp = self._enforce_requested_duration(svp=svp, duration=duration)
            return svp

        raise PlannerSchemaError("planner failed to produce a valid SVP output")

    def _call_messages_api(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._system_prompt,
                tools=[
                    {
                        "name": self.TOOL_NAME,
                        "description": "Generate a valid SVP.v4x-five-layer.video object.",
                        "input_schema": SVPVideo.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": self.TOOL_NAME},
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise PlannerAPIError("anthropic API request failed in planner") from exc

    def _extract_tool_input(self, response: Any) -> tuple[Any, dict[str, Any]]:
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            raise PlannerSchemaError("planner response content is missing or invalid")

        selected: Any | None = None
        for block in content:
            is_tool_use = getattr(block, "type", None) == "tool_use"
            is_target_tool = getattr(block, "name", None) == self.TOOL_NAME
            if is_tool_use and is_target_tool:
                selected = block
                break
        if selected is None:
            for block in content:
                if getattr(block, "type", None) == "tool_use":
                    selected = block
                    break
        if selected is None:
            raise PlannerSchemaError("planner response does not include tool_use block")

        tool_input = getattr(selected, "input", None)
        if isinstance(tool_input, dict):
            return selected, tool_input
        raise PlannerSchemaError("tool_use input is not an object")

    def _append_retry_messages(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        fallback_tool_use_id: str,
        validation_error: ValidationError,
    ) -> None:
        assistant_content = getattr(response, "content", [])
        messages.append({"role": "assistant", "content": assistant_content})

        tool_use_ids = [
            getattr(block, "id", None)
            for block in assistant_content
            if getattr(block, "type", None) == "tool_use"
        ]
        tool_use_ids = [tool_use_id for tool_use_id in tool_use_ids if tool_use_id]
        if not tool_use_ids:
            tool_use_ids = [fallback_tool_use_id]

        tool_results = []
        for tool_use_id in tool_use_ids:
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": (
                        "Schema validation failed. Return one corrected "
                        "generate_svp_video tool call only. "
                        f"Error details: {validation_error.errors()}"
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }
        )

    def _build_user_prompt(self, user_prompt: str, duration: int | None) -> str:
        if duration is None:
            return user_prompt
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration must be an integer between 4 and 15 seconds")
        if not 4 <= duration <= 15:
            raise ValueError("duration must be between 4 and 15 seconds")
        return f"{user_prompt}\n\n追加要件: duration_seconds は {duration} 秒にしてください。"

    def _inject_motion_forbidden(self, svp: SVPVideo) -> SVPVideo:
        current = list(svp.motion_layer.constraints.forbidden)
        updated = False
        for required in self.REQUIRED_MOTION_FORBIDDEN:
            if required not in current:
                current.append(required)
                updated = True
        if not updated:
            return svp

        new_constraints = svp.motion_layer.constraints.model_copy(update={"forbidden": current})
        new_motion_layer = svp.motion_layer.model_copy(update={"constraints": new_constraints})
        return svp.model_copy(update={"motion_layer": new_motion_layer})

    def _enforce_requested_duration(self, svp: SVPVideo, duration: int | None) -> SVPVideo:
        if duration is None:
            return svp
        if svp.motion_layer.duration_seconds == duration:
            return svp
        new_motion_layer = svp.motion_layer.model_copy(update={"duration_seconds": duration})
        return svp.model_copy(update={"motion_layer": new_motion_layer})

    def _apply_character_locks(self, svp: SVPVideo, user_prompt: str) -> SVPVideo:
        locks = _extract_identity_locks(user_prompt)
        if not locks:
            return svp

        single_subject_intent = _prompt_indicates_single_subject(user_prompt)
        identity_locks = _append_unique(list(svp.identity_locks), locks)

        face_required = _append_unique(list(svp.face_layer.constraints.required), locks)
        face_forbidden = _append_unique(
            list(svp.face_layer.constraints.forbidden),
            [
                "wrong gender",
                "wrong hair color",
                "wrong eye color",
                "different outfit",
                "missing key weapon or prop",
            ],
        )
        face_constraints = svp.face_layer.constraints.model_copy(
            update={"required": face_required, "forbidden": face_forbidden}
        )
        face_features = _append_unique(list(svp.face_layer.distinctive_features), locks)
        face_layer = svp.face_layer.model_copy(
            update={
                "constraints": face_constraints,
                "distinctive_features": face_features,
            }
        )

        composition_required = _append_unique(
            list(svp.composition_layer.constraints.required),
            (
                ["single primary character only", *locks]
                if single_subject_intent
                else locks
            ),
        )
        composition_forbidden_additions = ["collage layout", "multi-panel grid"]
        if single_subject_intent:
            composition_forbidden_additions.extend(["extra characters", "duplicate character"])
        composition_forbidden = _append_unique(
            list(svp.composition_layer.constraints.forbidden),
            composition_forbidden_additions,
        )
        composition_constraints = svp.composition_layer.constraints.model_copy(
            update={"required": composition_required, "forbidden": composition_forbidden}
        )
        composition_layer = svp.composition_layer.model_copy(
            update={"constraints": composition_constraints}
        )

        global_required = _append_unique(list(svp.c3.constraints.required), locks)
        global_forbidden_additions = [
            "collage, contact sheet, split panel, or numbered panel layout",
        ]
        if single_subject_intent:
            global_forbidden_additions.extend(
                [
                    "male character if a female character was specified",
                    "extra characters",
                    "duplicated background character",
                ]
            )
        global_forbidden = _append_unique(
            list(svp.c3.constraints.forbidden),
            global_forbidden_additions,
        )
        global_constraints = svp.c3.constraints.model_copy(
            update={"required": global_required, "forbidden": global_forbidden}
        )

        hit_list = _append_unique(list(svp.c3.evaluation_criteria.hit_list), locks)
        critical_fail_additions = [
            "subject gender changes",
            "hair color or hairstyle changes",
            "eye color changes",
            "outfit identity changes",
        ]
        if single_subject_intent:
            critical_fail_additions.append("extra or duplicated characters appear")
        critical_fail_conditions = _append_unique(
            list(svp.c3.evaluation_criteria.critical_fail_conditions),
            critical_fail_additions,
        )
        evaluation_criteria = svp.c3.evaluation_criteria.model_copy(
            update={
                "hit_list": hit_list,
                "critical_fail_conditions": critical_fail_conditions,
            }
        )
        c3 = svp.c3.model_copy(
            update={
                "constraints": global_constraints,
                "evaluation_criteria": evaluation_criteria,
            }
        )

        role_visual_cue = svp.role_visual_cue.model_copy(
            update={
                "role": svp.role_visual_cue.role or "character",
                "visual_elements": _append_unique(
                    list(svp.role_visual_cue.visual_elements),
                    locks,
                ),
            }
        )
        variation_policy = svp.variation_policy.model_copy(
            update={
                "clothing_variation": "none",
                "pose_variation": "minimal",
                "background_structure_variation": "minimal",
                "color_variation": "small",
            }
        )
        reference_usage_policy = svp.reference_usage_policy.model_copy(
            update={
                "use_reference_for": _append_unique(
                    list(svp.reference_usage_policy.use_reference_for),
                    [
                        "character identity",
                        "hair color and hairstyle",
                        "eye color",
                        "outfit silhouette and pattern",
                        "weapon or held prop identity",
                    ],
                ),
                "do_not_copy_from_reference": _append_unique(
                    list(svp.reference_usage_policy.do_not_copy_from_reference),
                    [
                        "reference background",
                        "collage or contact sheet layout",
                        "number labels",
                        "duplicate character poses",
                        "extra swords or weapon trails",
                        "texture noise",
                        "scratch-like line artifacts",
                        "compression-like speckles",
                    ],
                ),
                "background_source": "SVP prompt, not reference image",
                "identity_strength": "high",
                "scene_transfer_strength": "low",
                "object_instance_rules": _append_unique(
                    list(svp.reference_usage_policy.object_instance_rules),
                    [
                        "katana count = exactly one when a katana is specified",
                        "katana location = character waist or character hand only",
                        "no sword-like shapes in the background",
                        "no diagonal blade reflections behind the character",
                        "no duplicated katana silhouettes",
                    ],
                ),
                "background_quality_rules": _append_unique(
                    list(svp.reference_usage_policy.background_quality_rules),
                    [
                        "clean background matching the SVP scene context",
                        "smooth distant background details",
                        "reduced background micro-line density",
                        "no gritty texture noise",
                        "no scratch-like artifacts",
                        "no compression-like speckles",
                    ],
                ),
            }
        )

        return svp.model_copy(
            update={
                "identity_locks": identity_locks,
                "face_layer": face_layer,
                "composition_layer": composition_layer,
                "c3": c3,
                "role_visual_cue": role_visual_cue,
                "variation_policy": variation_policy,
                "reference_usage_policy": reference_usage_policy,
            }
        )


def _load_system_prompt() -> str:
    return (
        resources.files("svp_pipeline.prompts")
        .joinpath("planner_system.md")
        .read_text(encoding="utf-8")
        .strip()
    )


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


def _prompt_indicates_single_subject(user_prompt: str) -> bool:
    prompt = " ".join(user_prompt.replace(";", ",").split())
    lower = " ".join(user_prompt.lower().replace(";", ",").split())
    multi_subject_patterns = [
        r"\b(?:two|three|four|multiple|several)\s+(?:\w+\s+){0,2}"
        r"(?:people|persons|characters|women|girls|men|boys|subjects)\b",
        r"\b(?:pair|duo|couple|group|crowd|team|twins)\b",
        r"\b(?:woman|girl|man|boy|person|character)\s+and\s+"
        r"(?:woman|girl|man|boy|person|character)\b",
        r"\bwith\s+another\s+(?:person|character|woman|girl|man|boy)\b",
    ]
    if any(re.search(pattern, lower) for pattern in multi_subject_patterns):
        return False
    multi_subject_literals = (
        "二人",
        "二名",
        "2人",
        "2名",
        "複数",
        "複数人",
        "群衆",
        "グループ",
        "ペア",
        "カップル",
        "双子",
    )
    if any(_contains_unnegated_literal(prompt, literal) for literal in multi_subject_literals):
        return False

    single_subject_patterns = [
        r"\b(?:a|one|single|solo|lone)\s+(?:young\s+adult\s+)?"
        r"(?:woman|girl|man|boy|person|character|subject)\b",
        r"\b(?:young adult woman|young woman|female character|woman|girl)\b",
        r"\b(?:young adult man|young man|male character|man|boy)\b",
    ]
    if any(re.search(pattern, lower) for pattern in single_subject_patterns):
        return True
    single_subject_literals = (
        "単独",
        "単独の女性",
        "一人",
        "一名",
        "1人",
        "1名",
        "ひとり",
        "ソロ",
    )
    return any(_contains_unnegated_literal(prompt, literal) for literal in single_subject_literals)


def _find_unnegated_match(pattern: str, lower_prompt: str) -> re.Match[str] | None:
    for match in re.finditer(pattern, lower_prompt):
        if not _has_negation_context(lower_prompt, match.start(), match.end()):
            return match
    return None


def _contains_unnegated(pattern: str, lower_prompt: str) -> bool:
    return _find_unnegated_match(pattern, lower_prompt) is not None


def _contains_unnegated_literal(prompt: str, literal: str) -> bool:
    start = 0
    while True:
        index = prompt.find(literal, start)
        if index < 0:
            return False
        end = index + len(literal)
        if not _has_negation_context(prompt, index, end):
            return True
        start = end


def _has_negation_context(prompt: str, start: int, end: int) -> bool:
    if _has_negation_prefix(prompt.lower(), start):
        return True

    prefix = re.split(r"[,;:.、。()（）]", prompt[max(0, start - 16) : start])[-1]
    suffix = re.split(r"[,;:.、。()（）]", prompt[end : end + 16], maxsplit=1)[0]
    english_negation_after = (
        r"^\s*(?:is|are|was|were)?\s*"
        r"(?:not|never)\s+(?:allowed|desired|wanted|needed|included|permitted|present)\b"
    )
    japanese_negation_after = (
        r"^.{0,4}(?:ではない|じゃない|でない|ではなく|じゃなく|でなく|"
        r"不要|なし|無し|ない|持たない|含めない|避ける|除外|禁止)"
    )
    japanese_negation_before = r"(?:不要な|不要の|禁止の|禁止された|避けるべき)$"
    return bool(
        re.search(english_negation_after, suffix.lower())
        or re.search(japanese_negation_after, suffix)
        or re.search(japanese_negation_before, prefix)
    )


def _has_negation_prefix(lower_prompt: str, start: int) -> bool:
    window = lower_prompt[max(0, start - 60) : start]
    negation_pattern = (
        r"(?:^|[\s,;:.()])"
        r"(?:no|not|without|avoid|exclude|excluding|never)\s+"
        r"(?:\w+\s+){0,3}$"
    )
    return re.search(negation_pattern, window) is not None


def _extract_identity_locks(user_prompt: str) -> list[str]:
    """Extract literal subject traits that must survive Claude's SVP abstraction."""
    prompt = " ".join(user_prompt.replace(";", ",").split())
    lower = prompt.lower()
    locks: list[str] = []

    phrase_patterns = [
        r"young adult woman",
        r"young woman",
        r"\bfemale\b",
        r"\bwoman\b",
        r"\bgirl\b",
        r"silver[- ]gray high ponytail",
        r"silver[- ]gray ponytail",
        r"silver high ponytail",
        r"silver ponytail",
        r"vivid red eyes",
        r"red eyes",
        r"black and indigo floral kimono coat",
        r"black[- ]indigo floral kimono coat",
        r"floral kimono coat",
        r"kimono coat",
        r"katana at (?:her|his|their) waist",
        r"\bkatana\b",
    ]
    for pattern in phrase_patterns:
        match = _find_unnegated_match(pattern, lower)
        if match:
            locks.append(prompt[match.start() : match.end()])

    japanese_markers = [
        "若い成人女性",
        "女性",
        "少女",
        "銀灰色",
        "銀髪",
        "ポニーテール",
        "赤い瞳",
        "赤目",
        "黒藍",
        "花柄",
        "着物",
        "ロングコート",
        "日本刀",
        "刀",
    ]
    for marker in japanese_markers:
        if _contains_unnegated_literal(prompt, marker):
            locks.append(marker)

    has_silver = _contains_unnegated(r"\bsilver\b", lower) or _contains_unnegated_literal(
        prompt, "銀"
    )
    has_ponytail = _contains_unnegated(
        r"\bponytail\b", lower
    ) or _contains_unnegated_literal(prompt, "ポニーテール")
    if has_silver and has_ponytail:
        locks.append("silver-gray high ponytail")

    has_red_eyes = (
        _contains_unnegated(r"\bred eyes?\b", lower)
        or _contains_unnegated_literal(prompt, "赤い瞳")
        or _contains_unnegated_literal(prompt, "赤目")
    )
    if has_red_eyes:
        locks.append("red eyes")

    has_female_subject = (
        _contains_unnegated(r"\bwoman\b", lower)
        or _contains_unnegated(r"\bfemale\b", lower)
        or _contains_unnegated_literal(prompt, "女性")
        or _contains_unnegated_literal(prompt, "少女")
    )
    if has_female_subject:
        locks.append("female character")

    if (
        _contains_unnegated(r"\bkatana\b", lower)
        or _contains_unnegated_literal(prompt, "刀")
        or _contains_unnegated_literal(prompt, "日本刀")
    ):
        locks.append("katana")

    return _append_unique([], locks)
