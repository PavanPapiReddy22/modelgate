"""Tests for unifai canonical types."""

import pytest
from pydantic import ValidationError

from unifai.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Response,
    Role,
    Tool,
    ToolParameter,
    Usage,
)


# ── Enum Tests ───────────────────────────────────────────────────────────────


class TestEnums:
    def test_role_values(self) -> None:
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.SYSTEM == "system"
        assert Role.TOOL == "tool"

    def test_content_type_values(self) -> None:
        assert ContentType.TEXT == "text"
        assert ContentType.TOOL_USE == "tool_use"
        assert ContentType.TOOL_RESULT == "tool_result"

    def test_finish_reason_values(self) -> None:
        assert FinishReason.STOP == "stop"
        assert FinishReason.TOOL_USE == "tool_use"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.ERROR == "error"


# ── ContentBlock Tests ───────────────────────────────────────────────────────


class TestContentBlock:
    def test_text_block(self) -> None:
        block = ContentBlock(type=ContentType.TEXT, text="Hello")
        assert block.type == ContentType.TEXT
        assert block.text == "Hello"
        assert block.tool_call_id is None

    def test_tool_use_block(self) -> None:
        block = ContentBlock(
            type=ContentType.TOOL_USE,
            tool_call_id="call_123",
            tool_name="get_weather",
            tool_input={"location": "NYC"},
        )
        assert block.type == ContentType.TOOL_USE
        assert block.tool_call_id == "call_123"
        assert block.tool_name == "get_weather"
        assert block.tool_input == {"location": "NYC"}

    def test_tool_result_block(self) -> None:
        block = ContentBlock(
            type=ContentType.TOOL_RESULT,
            tool_call_id="call_123",
            tool_result_content="72°F and sunny",
        )
        assert block.type == ContentType.TOOL_RESULT
        assert block.tool_result_content == "72°F and sunny"


# ── Message Tests ────────────────────────────────────────────────────────────


class TestMessage:
    def test_string_content(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_content_blocks(self) -> None:
        blocks = [ContentBlock(type=ContentType.TEXT, text="Hi")]
        msg = Message(role=Role.ASSISTANT, content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1

    def test_tool_role_with_valid_content(self) -> None:
        blocks = [
            ContentBlock(
                type=ContentType.TOOL_RESULT,
                tool_call_id="call_1",
                tool_result_content="result",
            )
        ]
        msg = Message(role=Role.TOOL, content=blocks)
        assert msg.role == Role.TOOL

    def test_tool_role_with_invalid_content_raises(self) -> None:
        blocks = [ContentBlock(type=ContentType.TEXT, text="bad")]
        with pytest.raises(ValidationError, match="TOOL_RESULT"):
            Message(role=Role.TOOL, content=blocks)

    def test_tool_role_with_string_content_allowed(self) -> None:
        # String content on TOOL role doesn't trigger block validation
        msg = Message(role=Role.TOOL, content="result text")
        assert msg.content == "result text"

    def test_model_validate_from_dict(self) -> None:
        msg = Message.model_validate({"role": "user", "content": "Hello"})
        assert msg.role == Role.USER
        assert msg.content == "Hello"


# ── Tool Tests ───────────────────────────────────────────────────────────────


class TestTool:
    def test_valid_tool(self) -> None:
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={
                "location": ToolParameter(type="string", description="City name"),
                "unit": ToolParameter(
                    type="string", enum=["celsius", "fahrenheit"]
                ),
            },
            required=["location"],
        )
        assert tool.name == "get_weather"
        assert "location" in tool.parameters
        assert tool.required == ["location"]

    def test_required_not_subset_raises(self) -> None:
        with pytest.raises(ValidationError, match="not in parameters"):
            Tool(
                name="bad_tool",
                description="Bad tool",
                parameters={
                    "location": ToolParameter(type="string"),
                },
                required=["location", "missing_param"],
            )

    def test_empty_required_is_valid(self) -> None:
        tool = Tool(
            name="simple",
            description="Simple tool",
            parameters={"x": ToolParameter(type="string")},
        )
        assert tool.required == []


# ── Usage Tests ──────────────────────────────────────────────────────────────


class TestUsage:
    def test_valid_usage(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150

    def test_invalid_total_raises(self) -> None:
        with pytest.raises(ValidationError, match="total_tokens"):
            Usage(input_tokens=100, output_tokens=50, total_tokens=999)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        assert usage.total_tokens == 0


# ── Response Tests ───────────────────────────────────────────────────────────


class TestResponse:
    def test_text_property(self) -> None:
        resp = Response(
            id="resp_1",
            model="test",
            content=[
                ContentBlock(type=ContentType.TEXT, text="Hello "),
                ContentBlock(type=ContentType.TEXT, text="world"),
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason=FinishReason.STOP,
        )
        assert resp.text == "Hello world"

    def test_text_property_no_text(self) -> None:
        resp = Response(
            id="resp_2",
            model="test",
            content=[
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call_id="call_1",
                    tool_name="fn",
                    tool_input={},
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason=FinishReason.TOOL_USE,
        )
        assert resp.text is None

    def test_tool_calls_property(self) -> None:
        resp = Response(
            id="resp_3",
            model="test",
            content=[
                ContentBlock(type=ContentType.TEXT, text="Let me check"),
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    tool_input={"location": "NYC"},
                ),
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason=FinishReason.TOOL_USE,
        )
        tool_calls = resp.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "get_weather"

    def test_mixed_content_preserves_order(self) -> None:
        resp = Response(
            id="resp_4",
            model="test",
            content=[
                ContentBlock(type=ContentType.TEXT, text="First"),
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call_id="c1",
                    tool_name="fn1",
                    tool_input={},
                ),
                ContentBlock(type=ContentType.TEXT, text="Second"),
            ],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason=FinishReason.STOP,
        )
        types = [b.type for b in resp.content]
        assert types == [ContentType.TEXT, ContentType.TOOL_USE, ContentType.TEXT]
