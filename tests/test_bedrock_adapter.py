"""Tests for the Bedrock adapter — Converse API format and SigV4 signing."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from unifai.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
    ToolParameter,
)
from unifai.errors import BedrockError


# ── Mock Responses ───────────────────────────────────────────────────────────

MOCK_CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello! How can I assist you today?"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 8},
    "requestId": "req-123",
}

MOCK_CONVERSE_TOOL_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "Let me check the weather."},
                {
                    "toolUse": {
                        "toolUseId": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    }
                },
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 20, "outputTokens": 15},
    "requestId": "req-456",
}


# ── Helper to create adapter with mocked credentials ────────────────────────


def _make_adapter():
    """Create a BedrockAdapter with mocked boto3 credentials."""
    mock_session = MagicMock()
    mock_frozen = MagicMock()
    mock_frozen.access_key = "AKIATEST"
    mock_frozen.secret_key = "SECRET"
    mock_frozen.token = None
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = mock_frozen
    mock_session.get_credentials.return_value = mock_creds

    from unifai.providers.bedrock import BedrockAdapter

    adapter = BedrockAdapter(region="us-east-1", boto3_session=mock_session)

    # Replace signer with a no-op mock so signed headers pass through
    adapter._signer = MagicMock()
    adapter._signer.add_auth.side_effect = lambda req: None

    # Override _sign_request to return headers unchanged (skip SigV4 in tests)
    def _mock_sign(method, url, headers, body):
        return headers

    adapter._sign_request = _mock_sign  # type: ignore[assignment]

    return adapter


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
    },
    required=["location"],
)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBedrockChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_chat(self) -> None:
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        ).mock(return_value=httpx.Response(200, json=MOCK_CONVERSE_RESPONSE))

        adapter = _make_adapter()
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model=model_id)

        assert resp.text == "Hello! How can I assist you today?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_use_response(self) -> None:
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        ).mock(return_value=httpx.Response(200, json=MOCK_CONVERSE_TOOL_RESPONSE))

        adapter = _make_adapter()
        messages = [
            Message(role=Role.USER, content="What's the weather in NYC?")
        ]
        resp = await adapter.chat(
            messages=messages, model=model_id, tools=[WEATHER_TOOL]
        )

        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.TEXT
        assert resp.content[1].type == ContentType.TOOL_USE
        assert resp.content[1].tool_call_id == "tool_123"
        assert resp.content[1].tool_name == "get_weather"
        assert resp.content[1].tool_input == {"location": "NYC"}
        assert resp.finish_reason == FinishReason.TOOL_USE


class TestBedrockMessageFormat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_system_as_top_level(self) -> None:
        """Bedrock uses a top-level system field, as a list of text blocks."""
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        route = respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        ).mock(return_value=httpx.Response(200, json=MOCK_CONVERSE_RESPONSE))

        adapter = _make_adapter()
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(
            messages=messages,
            model=model_id,
            system="You are a helpful assistant",
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system"] == [{"text": "You are a helpful assistant"}]

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_format(self) -> None:
        """Tool results must use toolResult blocks in user messages."""
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        route = respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        ).mock(return_value=httpx.Response(200, json=MOCK_CONVERSE_RESPONSE))

        adapter = _make_adapter()
        messages = [
            Message(role=Role.USER, content="Weather?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="tool_1",
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
                        tool_call_id="tool_1",
                        tool_result_content="72°F",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model=model_id)

        sent_body = json.loads(route.calls[0].request.content)
        tool_result_msg = sent_body["messages"][-1]

        # Must be role=user with toolResult block
        assert tool_result_msg["role"] == "user"
        tr = tool_result_msg["content"][0]["toolResult"]
        assert tr["toolUseId"] == "tool_1"
        assert tr["content"] == [{"text": "72°F"}]


class TestBedrockToolConfig:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_spec_format(self) -> None:
        """Tools must be sent as toolConfig.tools[].toolSpec."""
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        route = respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        ).mock(return_value=httpx.Response(200, json=MOCK_CONVERSE_RESPONSE))

        adapter = _make_adapter()
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(
            messages=messages, model=model_id, tools=[WEATHER_TOOL]
        )

        sent_body = json.loads(route.calls[0].request.content)
        tool_config = sent_body["toolConfig"]
        assert "tools" in tool_config
        spec = tool_config["tools"][0]["toolSpec"]
        assert spec["name"] == "get_weather"
        assert spec["description"] == "Get current weather"
        assert "inputSchema" in spec
