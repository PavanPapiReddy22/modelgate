"""Tests for the OpenAI / Generic OpenAI adapter."""

import json

import httpx
import pytest
import respx

from unifai.providers.generic_openai import GenericOpenAIAdapter
from unifai.providers.openai import OpenAIAdapter
from unifai.types import ContentType, FinishReason, Message, Role, Tool, ToolParameter
from unifai.errors import AuthenticationError, RateLimitError


# ── Fixtures ─────────────────────────────────────────────────────────────────

MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

MOCK_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
}

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
    },
    required=["location"],
)


# ── Non-Streaming Tests ─────────────────────────────────────────────────────


class TestGenericOpenAIChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_chat(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model="gpt-4o")

        assert resp.id == "chatcmpl-abc123"
        assert resp.text == "Hello! How can I help?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_TOOL_CALL_RESPONSE)
        )

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="What's the weather in NYC?")]
        resp = await adapter.chat(
            messages=messages, model="gpt-4o", tools=[WEATHER_TOOL]
        )

        assert resp.finish_reason == FinishReason.TOOL_USE
        assert len(resp.tool_calls) == 1

        tc = resp.tool_calls[0]
        assert tc.type == ContentType.TOOL_USE
        assert tc.tool_call_id == "call_abc"
        assert tc.tool_name == "get_weather"
        assert tc.tool_input == {"location": "NYC"}
        # Verify it's a dict, not a string
        assert isinstance(tc.tool_input, dict)

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error_maps_to_authentication_error(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        adapter = OpenAIAdapter(api_key="bad-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=messages, model="gpt-4o")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limit"}})
        )

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(RateLimitError):
            await adapter.chat(messages=messages, model="gpt-4o")


class TestGenericOpenAIAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_base_url(self) -> None:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        adapter = GenericOpenAIAdapter(
            base_url="http://localhost:11434/v1", api_key=None
        )
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model="mistral")

        assert resp.text == "Hello! How can I help?"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_message_prepended(self) -> None:
        route = respx.post("https://test.api.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        adapter = GenericOpenAIAdapter(base_url="https://test.api.com/v1")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(messages=messages, model="test", system="You are helpful")

        # Verify system message was sent
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["messages"][0]["role"] == "system"
        assert sent_body["messages"][0]["content"] == "You are helpful"


class TestToolResultConversion:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_message_format(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        from unifai.types import ContentBlock

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="call_1",
                        tool_name="get_weather",
                        tool_input={"location": "NYC"},
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="call_1",
                        tool_result_content="72°F and sunny",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="gpt-4o")

        sent_body = json.loads(route.calls[0].request.content)
        tool_msg = sent_body["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "72°F and sunny"


class TestMultiToolResult:
    @respx.mock
    @pytest.mark.asyncio
    async def test_multiple_tool_results_flattened(self) -> None:
        """Each tool result must become a separate message for OpenAI."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        from unifai.types import ContentBlock

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="Weather in NYC and SF?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="call_1",
                        tool_name="get_weather",
                        tool_input={"location": "NYC"},
                    ),
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="call_2",
                        tool_name="get_weather",
                        tool_input={"location": "SF"},
                    ),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="call_1",
                        tool_result_content="72°F sunny",
                    ),
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="call_2",
                        tool_result_content="65°F foggy",
                    ),
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="gpt-4o")

        sent_body = json.loads(route.calls[0].request.content)
        # Two separate tool messages instead of one combined
        tool_msgs = [m for m in sent_body["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "72°F sunny"
        assert tool_msgs[1]["tool_call_id"] == "call_2"
        assert tool_msgs[1]["content"] == "65°F foggy"


class TestMaxCompletionTokens:
    @respx.mock
    @pytest.mark.asyncio
    async def test_uses_max_completion_tokens(self) -> None:
        """OpenAI payload should use max_completion_tokens not max_tokens."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_CHAT_RESPONSE)
        )

        adapter = OpenAIAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(messages=messages, model="gpt-4o", max_tokens=1000)

        sent_body = json.loads(route.calls[0].request.content)
        assert "max_completion_tokens" in sent_body
        assert sent_body["max_completion_tokens"] == 1000
        assert "max_tokens" not in sent_body
