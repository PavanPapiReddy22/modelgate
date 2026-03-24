"""Tests for the Anthropic adapter."""

import json

import httpx
import pytest
import respx

from unifai.providers.anthropic import AnthropicAdapter
from unifai.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
    ToolParameter,
)
from unifai.errors import AuthenticationError, RateLimitError


# ── Mock Responses ───────────────────────────────────────────────────────────

MOCK_TEXT_RESPONSE = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I help?"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 8},
}

MOCK_TOOL_USE_RESPONSE = {
    "id": "msg_tool456",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check the weather for you."},
        {
            "type": "tool_use",
            "id": "toolu_01XFDUDYJgAACTvnkyLpI1",
            "name": "get_weather",
            "input": {"location": "NYC"},
        },
    ],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 15},
}

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
    },
    required=["location"],
)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAnthropicChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_text(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")

        assert resp.id == "msg_abc123"
        assert resp.text == "Hello! How can I help?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 12

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_use_with_mixed_content(self) -> None:
        """Anthropic can return text AND tool_use blocks in a single response."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TOOL_USE_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather in NYC?")
        ]
        resp = await adapter.chat(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
            tools=[WEATHER_TOOL],
        )

        # Verify mixed content preserved in order
        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.TEXT
        assert resp.content[0].text == "Let me check the weather for you."
        assert resp.content[1].type == ContentType.TOOL_USE
        assert resp.content[1].tool_call_id == "toolu_01XFDUDYJgAACTvnkyLpI1"
        assert resp.content[1].tool_name == "get_weather"
        assert resp.content[1].tool_input == {"location": "NYC"}
        assert isinstance(resp.content[1].tool_input, dict)

        # Verify both properties work
        assert resp.text == "Let me check the weather for you."
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == FinishReason.TOOL_USE


class TestAnthropicMessageFormat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_sent_as_user_message(self) -> None:
        """Tool results must be sent as user messages with tool_result blocks."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TEXT,
                        text="Let me check.",
                    ),
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="toolu_123",
                        tool_name="get_weather",
                        tool_input={"location": "NYC"},
                    ),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="toolu_123",
                        tool_result_content="72°F and sunny",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")

        sent_body = json.loads(route.calls[0].request.content)
        tool_result_msg = sent_body["messages"][-1]

        # Must be role=user with tool_result blocks
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_123"
        assert tool_result_msg["content"][0]["content"] == "72°F and sunny"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_sent_as_top_level(self) -> None:
        """Anthropic uses a top-level 'system' field, not a system message."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant",
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system"] == "You are a helpful assistant"
        # System should NOT appear in messages
        for msg in sent_body["messages"]:
            assert msg["role"] != "system"


class TestAnthropicErrors:
    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "Invalid API key"}}
            )
        )

        adapter = AnthropicAdapter(api_key="bad-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                429, json={"error": {"message": "Rate limited"}}
            )
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(RateLimitError):
            await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")
