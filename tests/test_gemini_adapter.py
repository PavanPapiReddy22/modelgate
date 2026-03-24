"""Tests for the Gemini adapter."""

import json

import httpx
import pytest
import respx

from unifai.providers.gemini import GeminiAdapter
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
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello! How can I help you today?"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 10,
        "totalTokenCount": 18,
    },
    "responseId": "resp-gemini-123",
}

MOCK_TOOL_CALL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Let me check the weather for you."},
                    {
                        "functionCall": {
                            "id": "fc_001",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 15,
        "candidatesTokenCount": 12,
        "totalTokenCount": 27,
    },
    "responseId": "resp-gemini-tool456",
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


class TestGeminiChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_text(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model="gemini-2.0-flash")

        assert resp.id == "resp-gemini-123"
        assert resp.text == "Hello! How can I help you today?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 8
        assert resp.usage.output_tokens == 10

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        """Gemini can return text AND functionCall parts in a single response."""
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TOOL_CALL_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather in NYC?")
        ]
        resp = await adapter.chat(
            messages=messages,
            model="gemini-2.0-flash",
            tools=[WEATHER_TOOL],
        )

        # Verify mixed content preserved in order
        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.TEXT
        assert resp.content[0].text == "Let me check the weather for you."
        assert resp.content[1].type == ContentType.TOOL_USE
        assert resp.content[1].tool_call_id == "fc_001"
        assert resp.content[1].tool_name == "get_weather"
        assert resp.content[1].tool_input == {"location": "NYC"}
        assert isinstance(resp.content[1].tool_input, dict)

        # Finish reason should be TOOL_USE when tool calls present
        assert resp.finish_reason == FinishReason.TOOL_USE
        assert len(resp.tool_calls) == 1


class TestGeminiMessageFormat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_sent_as_function_response(self) -> None:
        """Tool results must be sent as user messages with functionResponse parts."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
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
                        tool_call_id="fc_001",
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
                        tool_call_id="fc_001",
                        tool_name="get_weather",
                        tool_result_content="72°F and sunny",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="gemini-2.0-flash")

        sent_body = json.loads(route.calls[0].request.content)

        # Assistant message → role=model with functionCall part
        assistant_msg = sent_body["contents"][1]
        assert assistant_msg["role"] == "model"
        assert "functionCall" in assistant_msg["parts"][1]

        # Tool result → role=user with functionResponse part including id
        tool_msg = sent_body["contents"][2]
        assert tool_msg["role"] == "user"
        assert "functionResponse" in tool_msg["parts"][0]
        fr = tool_msg["parts"][0]["functionResponse"]
        assert fr["name"] == "get_weather"
        assert fr["id"] == "fc_001"
        assert fr["response"]["result"] == "72°F and sunny"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_sent_as_system_instruction(self) -> None:
        """System prompt must be sent as top-level system_instruction."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(
            messages=messages,
            model="gemini-2.0-flash",
            system="You are a helpful assistant",
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system_instruction"]["parts"][0]["text"] == "You are a helpful assistant"
        # System should NOT appear in contents
        for content in sent_body["contents"]:
            for part in content.get("parts", []):
                if "text" in part:
                    assert part["text"] != "You are a helpful assistant"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tools_sent_as_function_declarations(self) -> None:
        """Tools should be sent as functionDeclarations."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Check weather")]
        await adapter.chat(
            messages=messages, model="gemini-2.0-flash", tools=[WEATHER_TOOL]
        )

        sent_body = json.loads(route.calls[0].request.content)
        tool_decl = sent_body["tools"][0]["functionDeclarations"][0]
        assert tool_decl["name"] == "get_weather"
        assert tool_decl["description"] == "Get current weather"
        assert "location" in tool_decl["parameters"]["properties"]


class TestGeminiErrors:
    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "API key not valid"}}
            )
        )

        adapter = GeminiAdapter(api_key="bad-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=messages, model="gemini-2.0-flash")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(
            return_value=httpx.Response(
                429, json={"error": {"message": "Rate limit exceeded"}}
            )
        )

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(RateLimitError):
            await adapter.chat(messages=messages, model="gemini-2.0-flash")
