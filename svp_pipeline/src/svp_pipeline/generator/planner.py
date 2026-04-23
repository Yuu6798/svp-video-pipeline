"""Planner that converts natural language prompts into validated SVPVideo objects."""

from __future__ import annotations

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
    ) -> None:
        if model not in self._MODEL_ALIASES:
            raise ValueError(f"unsupported planner model: {model}")
        self.requested_model = model
        self.model = self._MODEL_ALIASES[model]
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


def _load_system_prompt() -> str:
    return (
        resources.files("svp_pipeline.prompts")
        .joinpath("planner_system.md")
        .read_text(encoding="utf-8")
        .strip()
    )
