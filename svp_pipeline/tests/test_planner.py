"""Tests for planner (M2)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from svp_pipeline.exceptions import PlannerAPIError, PlannerSchemaError
from svp_pipeline.generator import Planner
from svp_pipeline.schema import SVPVideo
from tests.fixtures.mock_responses import (
    MISSING_FORBIDDEN_RESPONSE,
    TOO_FEW_POR_CORE_RESPONSE,
    VALID_ACTION_RESPONSE,
    VALID_SHIBUYA_RESPONSE,
    VALID_STILL_LIFE_RESPONSE,
)

REQUIRED_MOTION_FORBIDDEN = (
    "PoR_core要素のフレームアウト",
    "grv_anchor主要要素の画面外移動",
)


class DummyMessages:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("no mock response left")
        return self._responses.pop(0)


class DummyClient:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = DummyMessages(responses=responses, error=error)


def test_plan_returns_valid_svp() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("夕暮れの渋谷スクランブル交差点")

    assert isinstance(svp, SVPVideo)
    assert svp.schema_version == "SVP.v4x-five-layer.video"


def test_plan_with_opus_model() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(model="claude-opus-4-7", client=client)

    planner.plan("opus model test")

    assert client.messages.calls[0]["model"] == "claude-opus-4-7"


def test_plan_with_haiku_model() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(model="claude-haiku-4-5", client=client)

    planner.plan("haiku model test")

    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_plan_uses_tool_choice_forced() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("tool choice test")

    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "generate_svp_video"}


def test_plan_injects_duration() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("duration test", duration=8)

    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "8 秒" in user_content
    assert svp.motion_layer.duration_seconds == 8


def test_plan_rejects_non_integer_duration() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    with pytest.raises(ValueError, match="integer"):
        planner.plan("duration type test", duration=8.5)

    assert len(client.messages.calls) == 0


def test_forbidden_injection_when_missing() -> None:
    client = DummyClient(responses=[MISSING_FORBIDDEN_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("forbidden injection test")
    forbidden = svp.motion_layer.constraints.forbidden

    for item in REQUIRED_MOTION_FORBIDDEN:
        assert item in forbidden


def test_forbidden_injection_idempotent() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("idempotent test")
    forbidden = svp.motion_layer.constraints.forbidden

    for item in REQUIRED_MOTION_FORBIDDEN:
        assert forbidden.count(item) == 1


def test_forbidden_injection_preserves_others() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("preserve existing forbidden")
    forbidden = svp.motion_layer.constraints.forbidden

    assert "過剰な手ブレ" in forbidden
    for item in REQUIRED_MOTION_FORBIDDEN:
        assert item in forbidden


def test_pydantic_validation_error_triggers_retry() -> None:
    client = DummyClient(responses=[TOO_FEW_POR_CORE_RESPONSE, VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("retry on validation error")

    assert isinstance(svp, SVPVideo)
    assert len(client.messages.calls) == 2
    retry_messages = client.messages.calls[1]["messages"]
    assert retry_messages[-1]["content"][0]["type"] == "tool_result"
    assert retry_messages[-1]["content"][0]["is_error"] is True


def test_retry_returns_tool_results_for_all_prior_tool_uses() -> None:
    invalid_input = copy.deepcopy(VALID_SHIBUYA_RESPONSE.content[0].input)
    invalid_input["por_core"] = ["one", "two"]
    extra_input = copy.deepcopy(VALID_STILL_LIFE_RESPONSE.content[0].input)
    first_response = Message(
        id="msg_multi_tool_use",
        content=[
            ToolUseBlock(
                id="toolu_invalid",
                name="generate_svp_video",
                type="tool_use",
                input=invalid_input,
            ),
            ToolUseBlock(
                id="toolu_extra",
                name="generate_svp_video",
                type="tool_use",
                input=extra_input,
            ),
        ],
        model="claude-opus-4-7",
        role="assistant",
        type="message",
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    client = DummyClient(responses=[first_response, VALID_ACTION_RESPONSE])
    planner = Planner(client=client)

    svp = planner.plan("retry should include all tool_result ids")

    assert isinstance(svp, SVPVideo)
    assert len(client.messages.calls) == 2
    retry_messages = client.messages.calls[1]["messages"]
    tool_results = retry_messages[-1]["content"]
    assert len(tool_results) == 2
    assert {block["tool_use_id"] for block in tool_results} == {"toolu_invalid", "toolu_extra"}
    assert all(block["type"] == "tool_result" for block in tool_results)
    assert all(block["is_error"] is True for block in tool_results)


def test_two_consecutive_failures_raise() -> None:
    client = DummyClient(responses=[TOO_FEW_POR_CORE_RESPONSE, TOO_FEW_POR_CORE_RESPONSE])
    planner = Planner(client=client)

    with pytest.raises(PlannerSchemaError):
        planner.plan("retry should fail twice")

    assert len(client.messages.calls) == 2


def test_api_error_is_wrapped() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    api_error = anthropic.APIError("api down", request=request, body=None)
    client = DummyClient(error=api_error)
    planner = Planner(client=client)

    with pytest.raises(PlannerAPIError):
        planner.plan("api error wrap")


def test_system_prompt_loaded_from_file() -> None:
    client = DummyClient(responses=[VALID_SHIBUYA_RESPONSE])
    planner = Planner(client=client)

    planner.plan("system prompt load test")

    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "svp_pipeline"
        / "prompts"
        / "planner_system.md"
    )
    expected = prompt_path.read_text(encoding="utf-8").strip()
    assert client.messages.calls[0]["system"] == expected


def test_mock_responses_are_valid_svp() -> None:
    for response in (VALID_SHIBUYA_RESPONSE, VALID_STILL_LIFE_RESPONSE, VALID_ACTION_RESPONSE):
        tool_use = next(block for block in response.content if block.type == "tool_use")
        svp = SVPVideo.model_validate(tool_use.input)
        assert isinstance(svp, SVPVideo)
