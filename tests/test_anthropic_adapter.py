"""Tests for the Anthropic adapter."""

import json

import httpx
import pytest
import respx

from modelgate.providers.anthropic import AnthropicAdapter
from modelgate.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
    ToolParameter,
    Usage,
)
from modelgate.errors import AuthenticationError, ProviderError, RateLimitError, StreamingError


# ── Mock Responses ───────────────────────────────────────────────────────────

MOCK_TEXT_RESPONSE = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I help?"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
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

MOCK_THINKING_RESPONSE = {
    "id": "msg_think_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Let me analyze this step by step...", "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8=="},
        {"type": "text", "text": "Based on my analysis, the answer is 42."},
    ],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 15, "output_tokens": 30, "thinking_input_tokens": 300},
}

MOCK_REDACTED_THINKING_RESPONSE = {
    "id": "msg_redacted_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Let me analyze...", "signature": "sig_normal=="},
        {"type": "redacted_thinking", "data": "EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc="},
        {"type": "text", "text": "Here is my answer."},
    ],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 20, "output_tokens": 50, "thinking_input_tokens": 100},
}

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={"location": ToolParameter(type="string", description="City name")},
    required=["location"],
)


# ── Helper ──────────────────────────────────────────────────────────────────

def _make_sse(*events: dict) -> str:
    """Build an SSE text body from a list of event dicts."""
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}\n\n")
    return "".join(lines)


# ── Chat Tests ──────────────────────────────────────────────────────────────


class TestAnthropicChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_text(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022")
        assert resp.id == "msg_abc123"
        assert resp.text == "Hello! How can I help?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 12

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_use_with_mixed_content(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TOOL_USE_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(
            messages=[Message(role=Role.USER, content="What's the weather in NYC?")],
            model="claude-3-5-sonnet-20241022",
            tools=[WEATHER_TOOL],
        )
        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.TEXT
        assert resp.content[1].type == ContentType.TOOL_USE
        assert resp.content[1].tool_call_id == "toolu_01XFDUDYJgAACTvnkyLpI1"
        assert resp.content[1].tool_name == "get_weather"
        assert resp.content[1].tool_input == {"location": "NYC"}
        assert resp.finish_reason == FinishReason.TOOL_USE


# ── Message Format Tests ───────────────────────────────────────────────────


class TestAnthropicMessageFormat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_sent_as_user_message(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather?"),
            Message(role=Role.ASSISTANT, content=[
                ContentBlock(type=ContentType.TEXT, text="Let me check."),
                ContentBlock(type=ContentType.TOOL_USE, tool_call_id="toolu_123", tool_name="get_weather", tool_input={"location": "NYC"}),
            ]),
            Message(role=Role.TOOL, content=[
                ContentBlock(type=ContentType.TOOL_RESULT, tool_call_id="toolu_123", tool_result_content="72°F and sunny"),
            ]),
        ]
        await adapter.chat(messages=messages, model="claude-3-5-sonnet-20241022")
        sent_body = json.loads(route.calls[0].request.content)
        tool_result_msg = sent_body["messages"][-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_sent_as_top_level(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022", system="You are a helpful assistant")
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system"] == "You are a helpful assistant"

    @respx.mock
    @pytest.mark.asyncio
    async def test_array_system_prompt(self) -> None:
        """Array system prompts (for cache_control) must be forwarded verbatim."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        sys_blocks = [
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
        ]
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", system=sys_blocks)
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["system"] == sys_blocks

    @respx.mock
    @pytest.mark.asyncio
    async def test_image_block_base64_sent(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content=[
            ContentBlock(type=ContentType.IMAGE, image_source_type="base64", image_media_type="image/png", image_data="iVBOR..."),
            ContentBlock(type=ContentType.TEXT, text="What is this?"),
        ])]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")
        sent_body = json.loads(route.calls[0].request.content)
        img_block = sent_body["messages"][0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"

    @respx.mock
    @pytest.mark.asyncio
    async def test_document_block_sent(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content=[
            ContentBlock(type=ContentType.DOCUMENT, document_source_type="base64", document_media_type="application/pdf", document_data="JVBE...", document_filename="doc.pdf"),
            ContentBlock(type=ContentType.TEXT, text="Summarize this."),
        ])]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")
        sent_body = json.loads(route.calls[0].request.content)
        doc_block = sent_body["messages"][0]["content"][0]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "base64"
        assert doc_block["source"]["filename"] == "doc.pdf"


# ── Error Tests ─────────────────────────────────────────────────────────────


class TestAnthropicErrors:
    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )
        adapter = AnthropicAdapter(api_key="bad-key")
        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(RateLimitError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022")

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, json={"error": {"message": "Internal error"}})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(ProviderError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022")


# ── Kwargs Passthrough Tests ────────────────────────────────────────────────


class TestAnthropicKwargsPassthrough:
    @respx.mock
    @pytest.mark.asyncio
    async def test_stop_sequences_forwarded(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", stop_sequences=["END"])
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["stop_sequences"] == ["END"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_top_p_top_k_forwarded(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", top_p=0.9, top_k=40)
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["top_p"] == 0.9
        assert sent_body["top_k"] == 40

    @respx.mock
    @pytest.mark.asyncio
    async def test_metadata_forwarded(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", metadata={"user_id": "u123"})
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["metadata"] == {"user_id": "u123"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_service_tier_forwarded(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", service_tier="auto")
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["service_tier"] == "auto"


# ── Tool Choice Tests ───────────────────────────────────────────────────────


class TestAnthropicToolChoice:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_auto(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6", tools=[WEATHER_TOOL], tool_choice="auto")
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["tool_choice"] == {"type": "auto"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_specific_tool(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TOOL_USE_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6",
            tools=[WEATHER_TOOL], tool_choice={"type": "tool", "name": "get_weather"},
        )
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["tool_choice"] == {"type": "tool", "name": "get_weather"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_none_omitted(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6")
        sent_body = json.loads(route.calls[0].request.content)
        assert "tool_choice" not in sent_body


# ── Stop Reason Tests ───────────────────────────────────────────────────────


class TestAnthropicStopReasons:
    @respx.mock
    @pytest.mark.asyncio
    async def test_pause_turn(self) -> None:
        resp_data = {**MOCK_TEXT_RESPONSE, "stop_reason": "pause_turn"}
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=resp_data)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6")
        assert resp.finish_reason == FinishReason.PAUSE_TURN

    @respx.mock
    @pytest.mark.asyncio
    async def test_refusal(self) -> None:
        resp_data = {**MOCK_TEXT_RESPONSE, "stop_reason": "refusal"}
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=resp_data)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6")
        assert resp.finish_reason == FinishReason.REFUSAL

    @respx.mock
    @pytest.mark.asyncio
    async def test_stop_sequence_captured(self) -> None:
        resp_data = {**MOCK_TEXT_RESPONSE, "stop_reason": "stop_sequence", "stop_sequence": "END"}
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=resp_data)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6")
        assert resp.finish_reason == FinishReason.STOP
        assert resp.stop_sequence == "END"


# ── Server Tool Use Tests ───────────────────────────────────────────────────


class TestAnthropicServerToolUse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_server_tool_use_parsed(self) -> None:
        resp_data = {
            **MOCK_TEXT_RESPONSE,
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "latest news"}},
                {"type": "text", "text": "Here are results."},
            ],
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=resp_data)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Search")], model="claude-sonnet-4-6")
        assert resp.content[0].type == ContentType.SERVER_TOOL_USE
        assert resp.content[0].tool_name == "web_search"
        assert resp.content[0].tool_input == {"query": "latest news"}


# ── Cache Usage Tests ───────────────────────────────────────────────────────


class TestAnthropicCacheUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_tokens_parsed(self) -> None:
        resp_data = {
            **MOCK_TEXT_RESPONSE,
            "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=resp_data)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6")
        assert resp.usage.cache_read_input_tokens == 80
        assert resp.usage.cache_creation_input_tokens == 20


# ── Extended Thinking Tests ──────────────────────────────────────────────────


class TestAnthropicThinkingResponse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_blocks_parsed(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", thinking_budget=10000, max_tokens=16000)
        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.THINKING
        assert resp.content[0].thinking_signature == "WaUjzkypQ2mUEVM36O2TxuC06KN8=="
        assert resp.usage.thinking_tokens == 300

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_property(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", thinking_budget=10000, max_tokens=16000)
        assert resp.thinking == "Let me analyze this step by step..."
        assert resp.text == "Based on my analysis, the answer is 42."


class TestAnthropicThinkingPayload:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_budget_sent_and_temperature_omitted(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", thinking_budget=10000, max_tokens=16000)
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert "temperature" not in sent_body

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_thinking_budget_sends_temperature(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Hello")], model="claude-3-5-sonnet-20241022")
        sent_body = json.loads(route.calls[0].request.content)
        assert "temperature" in sent_body
        assert "thinking" not in sent_body


class TestAnthropicThinkingRoundTrip:
    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_blocks_serialized_in_history(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content=[
                ContentBlock(type=ContentType.THINKING, thinking="2+2=4", thinking_signature="sig123=="),
                ContentBlock(type=ContentType.TEXT, text="The answer is 4."),
            ]),
            Message(role=Role.USER, content="Are you sure?"),
        ]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")
        sent_body = json.loads(route.calls[0].request.content)
        assistant_msg = sent_body["messages"][1]
        assert assistant_msg["content"][0]["type"] == "thinking"
        assert assistant_msg["content"][0]["signature"] == "sig123=="


class TestAnthropicRedactedThinking:
    @respx.mock
    @pytest.mark.asyncio
    async def test_redacted_thinking_blocks_parsed(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_REDACTED_THINKING_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Help")], model="claude-sonnet-4-6", thinking_budget=10000, max_tokens=16000)
        assert len(resp.content) == 3
        assert resp.content[1].type == ContentType.REDACTED_THINKING
        assert resp.content[1].redacted_thinking_data == "EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc="

    @respx.mock
    @pytest.mark.asyncio
    async def test_redacted_thinking_round_tripped(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="Help"),
            Message(role=Role.ASSISTANT, content=[
                ContentBlock(type=ContentType.REDACTED_THINKING, redacted_thinking_data="encrypted_data_abc=="),
                ContentBlock(type=ContentType.TEXT, text="Answer."),
            ]),
            Message(role=Role.USER, content="More"),
        ]
        await adapter.chat(messages=messages, model="claude-sonnet-4-6")
        sent_body = json.loads(route.calls[0].request.content)
        redacted_block = sent_body["messages"][1]["content"][0]
        assert redacted_block["type"] == "redacted_thinking"
        assert redacted_block["data"] == "encrypted_data_abc=="


# ── Budget Validation Tests ──────────────────────────────────────────────────


class TestAnthropicBudgetValidation:
    @pytest.mark.asyncio
    async def test_budget_exceeds_max_tokens_raises(self) -> None:
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(ValueError, match="thinking_budget.*must be less than"):
            await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", max_tokens=4096, thinking_budget=4096)

    @pytest.mark.asyncio
    async def test_budget_greater_than_max_tokens_raises(self) -> None:
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(ValueError, match="thinking_budget.*must be less than"):
            await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", max_tokens=4096, thinking_budget=8000)


class TestAnthropicAdaptiveThinking:
    @respx.mock
    @pytest.mark.asyncio
    async def test_adaptive_sends_correct_payload(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_THINKING_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.chat(messages=[Message(role=Role.USER, content="Think")], model="claude-opus-4-6", thinking_budget="adaptive")
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["thinking"] == {"type": "adaptive"}
        assert "temperature" not in sent_body


# ── Output Config Tests ─────────────────────────────────────────────────────


class TestAnthropicOutputConfig:
    @respx.mock
    @pytest.mark.asyncio
    async def test_output_config_forwarded(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE)
        )
        adapter = AnthropicAdapter(api_key="test-key")
        schema = {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}}
        await adapter.chat(messages=[Message(role=Role.USER, content="JSON")], model="claude-sonnet-4-6", output_config=schema)
        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["output_config"] == schema


# ── Streaming Tests ──────────────────────────────────────────────────────────


class TestAnthropicStream:
    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_text(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6"):
            chunks.append(chunk)
        text_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.TEXT]
        assert len(text_chunks) == 2
        assert text_chunks[0].text == "Hello"
        assert text_chunks[1].text == " world"
        usage = chunks[-1]
        assert isinstance(usage, Usage)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_tool_use(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 20}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"loc'}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": 'ation": "NYC"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 15}},
            {"type": "message_stop"},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(messages=[Message(role=Role.USER, content="Weather")], model="claude-sonnet-4-6"):
            chunks.append(chunk)
        tool_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.TOOL_USE]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_name == "get_weather"
        assert tool_chunks[0].tool_input == {"location": "NYC"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_thinking(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Step 1..."}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig=="}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Answer"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 30, "thinking_input_tokens": 200}},
            {"type": "message_stop"},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(messages=[Message(role=Role.USER, content="Think")], model="claude-sonnet-4-6", thinking_budget=5000, max_tokens=16000):
            chunks.append(chunk)
        thinking_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.THINKING]
        assert len(thinking_chunks) == 1
        assert thinking_chunks[0].thinking == "Step 1..."
        assert thinking_chunks[0].thinking_signature == "sig=="
        usage = chunks[-1]
        assert isinstance(usage, Usage)
        assert usage.thinking_tokens == 200

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_error_event_raises(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(StreamingError, match="Overloaded"):
            async for _ in adapter.stream(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6"):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_cache_usage(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(messages=[Message(role=Role.USER, content="Hi")], model="claude-sonnet-4-6"):
            chunks.append(chunk)
        usage = chunks[-1]
        assert isinstance(usage, Usage)
        assert usage.cache_read_input_tokens == 80
        assert usage.cache_creation_input_tokens == 20

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_server_tool_use(self) -> None:
        sse_body = _make_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query": "news"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
        )
        adapter = AnthropicAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(messages=[Message(role=Role.USER, content="Search")], model="claude-sonnet-4-6"):
            chunks.append(chunk)
        srv_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.SERVER_TOOL_USE]
        assert len(srv_chunks) == 1
        assert srv_chunks[0].tool_name == "web_search"
