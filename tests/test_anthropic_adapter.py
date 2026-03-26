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


# ── Extended Thinking Tests ──────────────────────────────────────────────────


MOCK_THINKING_RESPONSE = {
    "id": "msg_think_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me analyze this step by step...",
            "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8==",
        },
        {"type": "text", "text": "Based on my analysis, the answer is 42."},
    ],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 15, "output_tokens": 30, "thinking_input_tokens": 300},
}


class TestAnthropicThinkingResponse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_blocks_parsed(self) -> None:
        """Thinking blocks must be parsed into THINKING ContentBlocks."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="What is the meaning of life?")]
        resp = await adapter.chat(
            messages=messages,
            model="claude-sonnet-4-6",
            thinking_budget=10000,
            max_tokens=16000,
        )

        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.THINKING
        assert resp.content[0].thinking == "Let me analyze this step by step..."
        assert resp.content[0].thinking_signature == "WaUjzkypQ2mUEVM36O2TxuC06KN8=="
        assert resp.content[1].type == ContentType.TEXT
        assert resp.content[1].text == "Based on my analysis, the answer is 42."
        # Thinking token usage
        assert resp.usage.thinking_tokens == 300

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_property(self) -> None:
        """Response.thinking property returns concatenated thinking."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Think hard")]
        resp = await adapter.chat(
            messages=messages,
            model="claude-sonnet-4-6",
            thinking_budget=10000,
            max_tokens=16000,
        )

        assert resp.thinking == "Let me analyze this step by step..."
        assert resp.text == "Based on my analysis, the answer is 42."


class TestAnthropicThinkingPayload:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_budget_sent_and_temperature_omitted(self) -> None:
        """When thinking_budget is set, thinking config must be in payload
        and temperature must NOT be sent."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Think")]
        await adapter.chat(
            messages=messages,
            model="claude-sonnet-4-6",
            thinking_budget=10000,
            max_tokens=16000,
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert "temperature" not in sent_body

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_thinking_budget_sends_temperature(self) -> None:
        """Without thinking_budget, temperature must be sent normally."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")

        sent_body = json.loads(route.calls[0].request.content)
        assert "temperature" in sent_body
        assert "thinking" not in sent_body


class TestAnthropicThinkingRoundTrip:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_blocks_serialized_in_history(self) -> None:
        """Thinking blocks must be round-tripped in assistant message history."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What is 2+2?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.THINKING,
                        thinking="2+2=4",
                        thinking_signature="sig123==",
                    ),
                    ContentBlock(
                        type=ContentType.TEXT,
                        text="The answer is 4.",
                    ),
                ],
            ),
            Message(role=Role.USER, content="Are you sure?"),
        ]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")

        sent_body = json.loads(route.calls[0].request.content)
        assistant_msg = sent_body["messages"][1]
        assert assistant_msg["role"] == "assistant"

        thinking_block = assistant_msg["content"][0]
        assert thinking_block["type"] == "thinking"
        assert thinking_block["thinking"] == "2+2=4"
        assert thinking_block["signature"] == "sig123=="

        text_block = assistant_msg["content"][1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "The answer is 4."


# ── Redacted Thinking Tests ──────────────────────────────────────────────────


MOCK_REDACTED_THINKING_RESPONSE = {
    "id": "msg_redacted_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me analyze...",
            "signature": "sig_normal==",
        },
        {
            "type": "redacted_thinking",
            "data": "EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc=",
        },
        {"type": "text", "text": "Here is my answer."},
    ],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 20, "output_tokens": 50, "thinking_input_tokens": 100},
}


class TestAnthropicRedactedThinking:
    @respx.mock
    @pytest.mark.asyncio
    async def test_redacted_thinking_blocks_parsed(self) -> None:
        """Redacted thinking blocks must be parsed into REDACTED_THINKING."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_REDACTED_THINKING_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Help me")]
        resp = await adapter.chat(
            messages=messages,
            model="claude-sonnet-4-6",
            thinking_budget=10000,
            max_tokens=16000,
        )

        assert len(resp.content) == 3
        assert resp.content[0].type == ContentType.THINKING
        assert resp.content[1].type == ContentType.REDACTED_THINKING
        assert resp.content[1].redacted_thinking_data == "EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc="
        assert resp.content[2].type == ContentType.TEXT

    @respx.mock
    @pytest.mark.asyncio
    async def test_redacted_thinking_round_tripped(self) -> None:
        """Redacted thinking blocks must be round-tripped verbatim."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="Help"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.REDACTED_THINKING,
                        redacted_thinking_data="encrypted_data_abc==",
                    ),
                    ContentBlock(type=ContentType.TEXT, text="Answer."),
                ],
            ),
            Message(role=Role.USER, content="More"),
        ]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")

        sent_body = json.loads(route.calls[0].request.content)
        assistant_msg = sent_body["messages"][1]
        redacted_block = assistant_msg["content"][0]
        assert redacted_block["type"] == "redacted_thinking"
        assert redacted_block["data"] == "encrypted_data_abc=="


# ── Budget Validation Tests ──────────────────────────────────────────────────


class TestAnthropicBudgetValidation:
    @pytest.mark.asyncio
    async def test_budget_exceeds_max_tokens_raises(self) -> None:
        """thinking_budget >= max_tokens must raise ValueError."""
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Think")]

        with pytest.raises(ValueError, match="thinking_budget.*must be less than"):
            await adapter.chat(
                messages=messages,
                model="claude-sonnet-4-6",
                max_tokens=4096,
                thinking_budget=4096,
            )

    @pytest.mark.asyncio
    async def test_budget_greater_than_max_tokens_raises(self) -> None:
        """thinking_budget > max_tokens must also raise."""
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Think")]

        with pytest.raises(ValueError, match="thinking_budget.*must be less than"):
            await adapter.chat(
                messages=messages,
                model="claude-sonnet-4-6",
                max_tokens=4096,
                thinking_budget=8000,
            )


# ── Adaptive Thinking Tests ─────────────────────────────────────────────────


class TestAnthropicAdaptiveThinking:
    @respx.mock
    @pytest.mark.asyncio
    async def test_adaptive_sends_correct_payload(self) -> None:
        """thinking_budget='adaptive' must send {"type": "adaptive"}."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )

        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Think adaptively")]
        await adapter.chat(
            messages=messages,
            model="claude-opus-4-6",
            thinking_budget="adaptive",
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["thinking"] == {"type": "adaptive"}
        assert "temperature" not in sent_body
        assert "budget_tokens" not in sent_body.get("thinking", {})

