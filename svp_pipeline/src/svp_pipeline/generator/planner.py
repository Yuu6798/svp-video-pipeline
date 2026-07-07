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
from .planner_rules import (
    SCENARIOS,
    ScenarioContext,
    _append_unique,
    _background_forbidden_items,
    _katana_reflection_forbidden_items,
    apply_scenario,
)

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
            svp = self._apply_background_noise_controls(svp=svp, user_prompt=user_prompt)
            svp = self._apply_object_contact_audit(svp=svp, user_prompt=user_prompt)
            svp = self._apply_katana_reflection_policy(svp=svp, user_prompt=user_prompt)
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

        ctx = ScenarioContext(
            locks=tuple(locks),
            single_subject_intent=_prompt_indicates_single_subject(user_prompt),
            role_visual_cue_role=svp.role_visual_cue.role,
        )
        return apply_scenario(svp, SCENARIOS.character_lock, ctx)

    def _apply_background_noise_controls(self, svp: SVPVideo, user_prompt: str) -> SVPVideo:
        risk_flags = _detect_background_noise_risk(user_prompt)
        if not risk_flags:
            return svp

        detailed_background = _detect_detailed_background_request(user_prompt)
        ctx = ScenarioContext(
            single_subject_intent=_prompt_indicates_single_subject(user_prompt),
            risk_flags=frozenset(risk_flags),
            detailed_background=detailed_background,
            background_forbidden=tuple(
                _background_forbidden_items(risk_flags, detailed_background)
            ),
            character_weapon_contact=_prompt_indicates_character_weapon_contact(user_prompt),
        )
        return apply_scenario(svp, SCENARIOS.background_noise, ctx)

    def _apply_object_contact_audit(self, svp: SVPVideo, user_prompt: str) -> SVPVideo:
        object_flags = _detect_object_contact_risk(user_prompt)
        if not {"umbrella", "katana"}.issubset(object_flags):
            return svp
        if _detect_drawn_weapon_request(user_prompt):
            return svp

        return apply_scenario(svp, SCENARIOS.umbrella_katana_contact, ScenarioContext())

    def _apply_katana_reflection_policy(self, svp: SVPVideo, user_prompt: str) -> SVPVideo:
        object_flags = _detect_object_contact_risk(user_prompt)
        if "katana" not in object_flags:
            return svp
        if _detect_drawn_weapon_request(user_prompt):
            return svp
        if not _prompt_indicates_character_weapon_contact(user_prompt):
            return svp
        if not _prompt_indicates_waist_katana(user_prompt):
            return svp

        ctx = ScenarioContext(reflection_forbidden=tuple(_katana_reflection_forbidden_items()))
        return apply_scenario(svp, SCENARIOS.katana_reflection, ctx)


def _load_system_prompt() -> str:
    return (
        resources.files("svp_pipeline.prompts")
        .joinpath("planner_system.md")
        .read_text(encoding="utf-8")
        .strip()
    )


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


def _detect_background_noise_risk(user_prompt: str) -> set[str]:
    lower_prompt = " ".join(user_prompt.lower().replace(";", ",").split())
    flags: set[str] = set()

    if _contains_unnegated(r"\b(?:cyberpunk|neon|night city|cityscape|urban)\b", lower_prompt):
        flags.add("dense_city")
    wet_surfaces = (
        r"(?:floor|floors|pavement|pavements|street|streets|road|roads|"
        r"sidewalk|sidewalks|walkway|walkways|surface|surfaces|ground)"
    )
    wet_reflection_patterns = (
        r"\b(?:rain|rainy)\b",
        rf"\bwet\s+(?:reflection|reflections|{wet_surfaces})\b",
        rf"\b{wet_surfaces}\s+(?:(?:is|are|was|were)\s+)?wet\b",
        rf"\b{wet_surfaces}\s+reflection(?:s)?\b",
        r"\breflection(?:s)?\b",
    )
    if any(_contains_unnegated(pattern, lower_prompt) for pattern in wet_reflection_patterns):
        flags.add("wet_reflection")
    if _contains_unnegated(r"\b(?:glass|transparent|umbrella)\b", lower_prompt):
        flags.add("transparent_object")
    if _contains_unnegated(r"\b(?:katana|sword|blade|gun|weapon)\b", lower_prompt):
        flags.add("weapon")
    if _contains_unnegated(
        r"\b(?:noise|noisy|grain|gritty|speckle|speckled|busy background)\b",
        lower_prompt,
    ):
        flags.add("explicit_noise_control")

    return flags


def _detect_detailed_background_request(user_prompt: str) -> bool:
    lower_prompt = " ".join(user_prompt.lower().replace(";", ",").split())
    detail_patterns = [
        r"\b(?:detailed|highly detailed|intricate|dense|busy)\s+"
        r"(?:cityscape|background|neon signage|signage|city|street)\b",
        r"\b(?:readable|legible)\s+(?:neon\s+)?(?:signage|signs|text)\b",
        r"\b(?:many|dense|crowded)\s+(?:neon\s+)?(?:signs|signage)\b",
    ]
    return any(_contains_unnegated(pattern, lower_prompt) for pattern in detail_patterns)


def _prompt_indicates_character_weapon_contact(user_prompt: str) -> bool:
    lower_prompt = " ".join(user_prompt.lower().replace(";", ",").split())
    character_patterns = [
        r"\b(?:person|character|woman|girl|man|boy|subject|human|samurai|ninja)\b",
        r"\b(?:her|his|their)\s+"
        r"(?:hand|hands|waist(?!-)|belt(?!-)|hip(?!-)|back|shoulder)\b",
        r"\b(?:hand|hands)\s+(?:holding|gripping|wielding)\b",
        r"\b(?:katana|sword|blade|gun|weapon)\s+in\s+(?:hand|hands)\b",
        r"\b(?:holding|gripping|wielding)\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?(?:katana|sword|blade|gun|weapon)\b",
    ]
    weapon_contact_patterns = [
        r"\b(?:katana|sword|blade|gun|weapon)\b",
        r"\b(?:hand|hands|waist(?!-)|belt(?!-)|hip(?!-)|back|shoulder|grip|holding|wielding)\b",
        r"\b(?:katana|sword|blade|gun|weapon)\s+at\s+(?:her|his|their|the)\s+"
        r"(?:waist(?!-)|belt(?!-)|hip(?!-)|back|hand)\b",
    ]
    has_character = any(
        _contains_unnegated(pattern, lower_prompt) for pattern in character_patterns
    )
    has_weapon_contact = any(
        _contains_unnegated(pattern, lower_prompt) for pattern in weapon_contact_patterns
    )
    return has_character and has_weapon_contact


def _prompt_indicates_waist_katana(user_prompt: str) -> bool:
    lower_prompt = " ".join(user_prompt.lower().replace(";", ",").split())
    side_location = (
        r"(?:(?:left|right|front|back|rear|side)\s+)?"
        r"(?:waist|belt|hip)"
        r"(?![-\s]+(?:high|height|level|display|stand|table|shelf|rack)\b)\b"
    )
    waist_katana_patterns = [
        r"\bkatana\s+(?:is\s+)?(?:sheathed\s+)?"
        r"(?:at|on|attached\s+to|fixed\s+to|hanging\s+from|strapped\s+to)\s+"
        rf"(?:her|his|their|the)\s+{side_location}",
        r"\bkatana\s+(?:is\s+)?(?:sheathed\s+)?(?:at|on)\s+"
        rf"{side_location}",
        r"\bkatana\s+(?:is\s+)?sheathed\s+(?:at|on)\s+"
        rf"{side_location}",
        r"\bkatana\s+(?:is\s+)?(?:sheathed\s+)?"
        r"(?:attached\s+to|fixed\s+to|hanging\s+from|strapped\s+to)\s+"
        rf"{side_location}",
        rf"\b{side_location}\s+"
        r"(?:katana\s+sheath|katana\s+scabbard|sheathed\s+katana)\b"
        r"(?!\s+(?:tattoo|print|pattern|emblem|logo|graphic|design)\b)",
        r"\b(?:sheathed|sheathe|scabbard(?:ed)?)\s+katana\s+"
        rf"(?:at|on)\s+(?:her|his|their|the)\s+{side_location}",
    ]
    return any(_contains_unnegated(pattern, lower_prompt) for pattern in waist_katana_patterns)


def _detect_drawn_weapon_request(user_prompt: str) -> bool:
    lower_prompt = " ".join(user_prompt.lower().replace(";", ",").split())
    drawn_patterns = [
        r"\b(?:drawn|unsheathed|raised|brandished)\s+(?:katana|sword|blade)\b",
        r"\b(?:draw|draws|drawing|pull|pulls|pulling)\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?"
        r"(?:katana|sword|blade)\s+(?:from|out\s+of)\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?(?:waist\s+)?(?:sheath|scabbard)\b",
        r"\b(?:she|he|they|woman|girl|man|boy|character|subject|samurai|ninja)\s+"
        r"(?:(?:(?:is|was|were)\s+)?"
        r"(?:draw|draws|drawing|drew|pull|pulls|pulling|pulled)|"
        r"(?:(?:has|had)\s+(?:drawn|pulled)))\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?(?:katana|sword|blade)\b",
        r"\bunsheath(?:e|es|ed|ing)?\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?(?:katana|sword|blade)\b",
        r"\b(?:katana|sword|blade)\s+(?:drawn|unsheathed|in hand|held|raised)\b",
        r"\b(?:holding|wielding|gripping)\s+"
        r"(?:(?:a|the|her|his|their|its)\s+)?(?:katana|sword|blade)\b",
    ]
    return any(_contains_unnegated(pattern, lower_prompt) for pattern in drawn_patterns)


def _detect_object_contact_risk(user_prompt: str) -> set[str]:
    prompt = " ".join(user_prompt.replace(";", ",").split())
    lower = prompt.lower()
    flags: set[str] = set()

    if _contains_unnegated(r"\b(?:umbrella|parasol)\b", lower):
        flags.add("umbrella")
    if _contains_unnegated(r"\b(?:katana|sword|blade)\b", lower):
        flags.add("katana")

    return flags


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
    locks.extend(_extract_phrase_identity_locks(prompt, lower))
    locks.extend(_extract_literal_identity_markers(prompt))
    locks.extend(_extract_derived_identity_locks(prompt, lower))
    return _append_unique([], locks)


def _extract_phrase_identity_locks(prompt: str, lower_prompt: str) -> list[str]:
    phrase_patterns = [
        r"young adult woman",
        r"young woman",
        r"\bfemale\b",
        r"\bwoman\b",
        r"\bgirl\b",
        r"young adult man",
        r"young man",
        r"\bmale\b",
        r"\bman\b",
        r"\bboy\b",
        r"silver[- ]gray high ponytail",
        r"silver[- ]gray ponytail",
        r"silver high ponytail",
        r"silver ponytail",
        r"\bvivid red eyes\b",
        r"\bred eyes\b",
        r"black and indigo floral kimono coat",
        r"black[- ]indigo floral kimono coat",
        r"floral kimono coat",
        r"kimono coat",
        r"katana at (?:her|his|their) waist",
        r"\bkatana\b",
    ]
    locks: list[str] = []
    for pattern in phrase_patterns:
        match = _find_unnegated_match(pattern, lower_prompt)
        if match:
            locks.append(prompt[match.start() : match.end()])
    return locks


def _extract_literal_identity_markers(prompt: str) -> list[str]:
    japanese_markers = [
        "若い成人女性",
        "女性",
        "少女",
        "若い成人男性",
        "男性",
        "少年",
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
    locks: list[str] = []
    for marker in japanese_markers:
        if _contains_unnegated_literal(prompt, marker):
            locks.append(marker)
    return locks


def _extract_derived_identity_locks(prompt: str, lower_prompt: str) -> list[str]:
    locks: list[str] = []
    if _has_silver_ponytail(prompt, lower_prompt):
        locks.append("silver-gray high ponytail")
    if _has_red_eyes(prompt, lower_prompt):
        locks.append("red eyes")
    if _has_female_subject(prompt, lower_prompt):
        locks.append("female character")
    if _has_male_subject(prompt, lower_prompt):
        locks.append("male character")
    if _has_katana(prompt, lower_prompt):
        locks.append("katana")

    return locks


def _has_silver_ponytail(prompt: str, lower_prompt: str) -> bool:
    has_silver = _contains_unnegated(r"\bsilver\b", lower_prompt) or _contains_unnegated_literal(
        prompt, "銀"
    )
    has_ponytail = _contains_unnegated(
        r"\bponytail\b", lower_prompt
    ) or _contains_unnegated_literal(
        prompt,
        "ポニーテール",
    )
    return has_silver and has_ponytail


def _has_red_eyes(prompt: str, lower_prompt: str) -> bool:
    return (
        _contains_unnegated(r"\bred eyes?\b", lower_prompt)
        or _contains_unnegated_literal(prompt, "赤い瞳")
        or _contains_unnegated_literal(prompt, "赤目")
    )


def _has_female_subject(prompt: str, lower_prompt: str) -> bool:
    return (
        _contains_unnegated(r"\bwoman\b", lower_prompt)
        or _contains_unnegated(r"\bfemale\b", lower_prompt)
        or _contains_unnegated_literal(prompt, "女性")
        or _contains_unnegated_literal(prompt, "少女")
    )


def _has_male_subject(prompt: str, lower_prompt: str) -> bool:
    return (
        _contains_unnegated(r"\bman\b", lower_prompt)
        or _contains_unnegated(r"\bmale\b", lower_prompt)
        or _contains_unnegated(r"\bboy\b", lower_prompt)
        or _contains_unnegated_literal(prompt, "男性")
        or _contains_unnegated_literal(prompt, "少年")
    )


def _has_katana(prompt: str, lower_prompt: str) -> bool:
    return (
        _contains_unnegated(r"\bkatana\b", lower_prompt)
        or _contains_unnegated_literal(prompt, "刀")
        or _contains_unnegated_literal(prompt, "日本刀")
    )
