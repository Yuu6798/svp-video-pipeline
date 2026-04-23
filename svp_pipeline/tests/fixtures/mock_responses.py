"""Mock Anthropic responses for planner tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from anthropic.types import Message, ToolUseBlock, Usage

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _load_sample(name: str) -> dict[str, Any]:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def build_tool_use_response(
    svp_dict: dict[str, Any],
    *,
    tool_use_id: str = "toolu_mock",
    model: str = "claude-opus-4-7",
) -> Message:
    return Message(
        id=f"msg_{tool_use_id}",
        content=[
            ToolUseBlock(
                id=tool_use_id,
                name="generate_svp_video",
                type="tool_use",
                input=svp_dict,
            )
        ],
        model=model,
        role="assistant",
        type="message",
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def build_invalid_tool_use_response(
    invalid_dict: dict[str, Any],
    *,
    tool_use_id: str = "toolu_invalid",
    model: str = "claude-opus-4-7",
) -> Message:
    return build_tool_use_response(invalid_dict, tool_use_id=tool_use_id, model=model)


VALID_SHIBUYA_RESPONSE: Message = build_tool_use_response(
    _load_sample("shibuya_dusk.json"),
    tool_use_id="toolu_shibuya",
)
VALID_STILL_LIFE_RESPONSE: Message = build_tool_use_response(
    _load_sample("still_life_macro.json"),
    tool_use_id="toolu_still_life",
)
VALID_ACTION_RESPONSE: Message = build_tool_use_response(
    _load_sample("action_ninja.json"),
    tool_use_id="toolu_action",
)

_missing_forbidden = deepcopy(_load_sample("shibuya_dusk.json"))
_missing_forbidden["motion_layer"]["constraints"]["forbidden"] = []
MISSING_FORBIDDEN_RESPONSE: Message = build_tool_use_response(
    _missing_forbidden,
    tool_use_id="toolu_missing_forbidden",
)

_too_few_por_core = deepcopy(_load_sample("shibuya_dusk.json"))
_too_few_por_core["por_core"] = ["要素A", "要素B"]
TOO_FEW_POR_CORE_RESPONSE: Message = build_invalid_tool_use_response(
    _too_few_por_core,
    tool_use_id="toolu_too_few_por_core",
)
