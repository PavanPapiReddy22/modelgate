"""
Mocked unit tests for GenericOpenAIAdapter.

Uses `respx` to mock HTTP requests — no live API key needed.

Run:
    pytest tests/test_openai_adapter_unit.py -v
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from modelgate.providers.generic_openai import GenericOpenAIAdapter
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
from modelgate.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    StreamingError,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

BASE_URL = "https://api.openai.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def make_adapter() -> GenericOpenAIAdapter:
    return GenericOpenAIAdapter(base_url=BASE_URL, api_key="sk-test-key")


def openai_response(
    text: str | None = "Hello!",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int = 0,
    stop_sequence: str | None = None,
) -> dict[str, Any]:
    """Build a realistic OpenAI Chat Completions response."""
    message: dict[str, Any] = {"role": "assistant"}
    if text is not None:
        message["content"] = text
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls

    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
    }
    if stop_sequence:
        choice["stop_sequence"] = stop_sequence

    resp: dict[str, Any] = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if reasoning_tokens:
        resp["usage"]["completion_tokens_details"] = {
            "reasoning_tokens": reasoning_tokens,
        }
    return resp


def sse_lines(*events: str) -> str:
    """Build SSE-formatted response body from event data strings."""
    lines = []
    for event in events:
        lines.append(f"data: {event}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def make_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get weather",
        parameters={"city": ToolParameter(type="string", description="City name")},
        required=["city"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. SIMPLE TEXT
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIChat:

    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_text(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(text="Hello world!")
        ))
        adapter = make_adapter()
        resp = await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
        )
        assert resp.text == "Hello world!"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.id == "chatcmpl-test123"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_parsing(self):
        """Tool call arguments (JSON string) must be parsed into a dict."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=openai_response(
            text=None,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
            }],
        )))
        adapter = make_adapter()
        resp = await adapter.chat(
            messages=[Message(role=Role.USER, content="Weather in NYC?")],
            model="gpt-4o-mini",
            tools=[make_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.tool_name == "get_weather"
        assert tc.tool_input == {"city": "NYC"}
        assert tc.tool_call_id == "call_abc123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_tool_calls(self):
        """Multiple parallel tool calls in one response."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=openai_response(
            text=None,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                },
                {
                    "id": "call_2", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                },
            ],
        )))
        adapter = make_adapter()
        resp = await adapter.chat(
            messages=[Message(role=Role.USER, content="Weather?")],
            model="gpt-4o-mini",
            tools=[make_tool()],
        )
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].tool_input == {"city": "NYC"}
        assert resp.tool_calls[1].tool_input == {"city": "London"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_json_args_returns_empty_dict(self):
        """If tool arguments are invalid JSON, return {} instead of crashing."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=openai_response(
            text=None,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_bad", "type": "function",
                "function": {"name": "get_weather", "arguments": "{invalid json"},
            }],
        )))
        adapter = make_adapter()
        resp = await adapter.chat(
            messages=[Message(role=Role.USER, content="Weather?")],
            model="gpt-4o-mini",
        )
        assert resp.tool_calls[0].tool_input == {}


# ──────────────────────────────────────────────────────────────────────────────
# 2. MESSAGE FORMAT
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIMessageFormat:

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_prompt_sent_as_first_message(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            system="You are helpful.",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert body["messages"][1] == {"role": "user", "content": "Hi"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_flattened(self):
        """Multiple tool results in one TOOL message must be flattened into separate messages."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[
                Message(role=Role.USER, content="Weather?"),
                Message(role=Role.TOOL, content=[
                    ContentBlock(type=ContentType.TOOL_RESULT, tool_call_id="call_1", tool_result_content="sunny"),
                    ContentBlock(type=ContentType.TOOL_RESULT, tool_call_id="call_2", tool_result_content="rainy"),
                ]),
            ],
            model="gpt-4o-mini",
        )
        body = json.loads(route.calls[0].request.content)
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "sunny"
        assert tool_msgs[1]["tool_call_id"] == "call_2"
        assert tool_msgs[1]["content"] == "rainy"

    @respx.mock
    @pytest.mark.asyncio
    async def test_image_url_sent(self):
        """Image URL content blocks should be serialized as image_url parts."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content=[
                ContentBlock(type=ContentType.TEXT, text="What's this?"),
                ContentBlock(type=ContentType.IMAGE, image_source_type="url",
                             image_data="https://example.com/cat.jpg"),
            ])],
            model="gpt-4o-mini",
        )
        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What's this?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/cat.jpg"

    @respx.mock
    @pytest.mark.asyncio
    async def test_image_base64_sent_as_data_uri(self):
        """Base64 images should be converted to data URIs."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content=[
                ContentBlock(type=ContentType.IMAGE, image_source_type="base64",
                             image_media_type="image/png", image_data="iVBOR"),
            ])],
            model="gpt-4o-mini",
        )
        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "data:image/png;base64,iVBOR"

    @respx.mock
    @pytest.mark.asyncio
    async def test_assistant_tool_use_serialized(self):
        """TOOL_USE blocks in assistant messages must serialize as tool_calls with JSON string arguments."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[
                Message(role=Role.USER, content="Weather?"),
                Message(role=Role.ASSISTANT, content=[
                    ContentBlock(type=ContentType.TOOL_USE, tool_call_id="call_1",
                                 tool_name="get_weather", tool_input={"city": "NYC"}),
                ]),
                Message(role=Role.TOOL, content=[
                    ContentBlock(type=ContentType.TOOL_RESULT, tool_call_id="call_1",
                                 tool_result_content="sunny"),
                ]),
            ],
            model="gpt-4o-mini",
        )
        body = json.loads(route.calls[0].request.content)
        asst_msg = body["messages"][1]
        assert asst_msg["role"] == "assistant"
        assert asst_msg["content"] is None
        assert asst_msg["tool_calls"][0]["id"] == "call_1"
        assert asst_msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(asst_msg["tool_calls"][0]["function"]["arguments"]) == {"city": "NYC"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. KWARGS
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIKwargs:

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_auto(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tools=[make_tool()],
            tool_choice="auto",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == "auto"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_required(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tools=[make_tool()],
            tool_choice="required",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == "required"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_specific_function(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tools=[make_tool()],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_none_omitted(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
        )
        body = json.loads(route.calls[0].request.content)
        assert "tool_choice" not in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_response_format_forwarded(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(text='{"answer":"42"}')
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
        )
        body = json.loads(route.calls[0].request.content)
        assert body["response_format"] == {"type": "json_object"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_passthrough_kwargs_forwarded(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop=["END"],
            seed=42,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["top_p"] == 0.9
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.3
        assert body["stop"] == ["END"]
        assert body["seed"] == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_reasoning_effort_sent_temperature_omitted(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="o4-mini",
            reasoning_effort="medium",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["reasoning_effort"] == "medium"
        assert "temperature" not in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_reasoning_effort_sends_temperature(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            temperature=0.7,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["temperature"] == 0.7
        assert "reasoning_effort" not in body


# ──────────────────────────────────────────────────────────────────────────────
# 4. ERROR HANDLING
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIErrors:

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            401, json={"error": {"message": "Invalid API key"}}
        ))
        adapter = make_adapter()
        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            429, json={"error": {"message": "Rate limited"}}
        ))
        adapter = make_adapter()
        with pytest.raises(RateLimitError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_request_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            400, json={"error": {"message": "Invalid request"}}
        ))
        adapter = make_adapter()
        with pytest.raises(InvalidRequestError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            500, json={"error": {"message": "Server error"}}
        ))
        adapter = make_adapter()
        with pytest.raises(ProviderError):
            await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")


# ──────────────────────────────────────────────────────────────────────────────
# 5. FINISH REASONS
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIFinishReasons:

    @respx.mock
    @pytest.mark.asyncio
    async def test_stop(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(finish_reason="stop")
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.finish_reason == FinishReason.STOP

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_calls(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=openai_response(
            text=None, finish_reason="tool_calls",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        )))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.finish_reason == FinishReason.TOOL_USE

    @respx.mock
    @pytest.mark.asyncio
    async def test_length(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(finish_reason="length")
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.finish_reason == FinishReason.LENGTH

    @respx.mock
    @pytest.mark.asyncio
    async def test_content_filter(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(finish_reason="content_filter")
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.finish_reason == FinishReason.STOP

    @respx.mock
    @pytest.mark.asyncio
    async def test_stop_sequence_captured(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(finish_reason="stop", stop_sequence="END")
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.stop_sequence == "END"


# ──────────────────────────────────────────────────────────────────────────────
# 6. USAGE
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIUsage:

    @respx.mock
    @pytest.mark.asyncio
    async def test_usage_parsed(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(prompt_tokens=100, completion_tokens=50)
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini")
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.total_tokens == 150

    @respx.mock
    @pytest.mark.asyncio
    async def test_reasoning_tokens_parsed(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response(reasoning_tokens=200)
        ))
        adapter = make_adapter()
        resp = await adapter.chat(messages=[Message(role=Role.USER, content="Hi")], model="o4-mini")
        assert resp.usage.thinking_tokens == 200


# ──────────────────────────────────────────────────────────────────────────────
# 7. TOOL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIToolDefs:

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_schema_sent(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tools=[make_tool()],
        )
        body = json.loads(route.calls[0].request.content)
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["properties"]["city"]["type"] == "string"
        assert tool["function"]["parameters"]["required"] == ["city"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_raw_schema_used(self):
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, json=openai_response()
        ))
        adapter = make_adapter()
        raw = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tools=[Tool(name="search", description="Search", raw_schema=raw)],
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["function"]["parameters"] == raw


# ──────────────────────────────────────────────────────────────────────────────
# 8. STREAMING
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIStream:

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_text(self):
        sse = sse_lines(
            json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
        )
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
        adapter = make_adapter()
        chunks = []
        async for chunk in adapter.stream(
            messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini"
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.TEXT]
        assert len(text_chunks) == 2
        assert text_chunks[0].text == "Hello"
        assert text_chunks[1].text == " world"

        usage_chunks = [c for c in chunks if isinstance(c, Usage)]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].input_tokens == 5
        assert usage_chunks[0].output_tokens == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_tool_use(self):
        sse = sse_lines(
            json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "NYC"}'}}]}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 8}}),
        )
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
        adapter = make_adapter()
        tool_chunks = []
        async for chunk in adapter.stream(
            messages=[Message(role=Role.USER, content="Weather?")], model="gpt-4o-mini"
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TOOL_USE:
                tool_chunks.append(chunk)

        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_name == "get_weather"
        assert tool_chunks[0].tool_input == {"city": "NYC"}
        assert tool_chunks[0].tool_call_id == "call_1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_error_raises(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(
            401, json={"error": {"message": "Bad key"}}
        ))
        adapter = make_adapter()
        with pytest.raises((AuthenticationError, StreamingError)):
            async for _ in adapter.stream(
                messages=[Message(role=Role.USER, content="Hi")], model="gpt-4o-mini"
            ):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_kwargs_forwarded(self):
        """Verify kwargs are forwarded in stream() too."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(
            200, text=sse_lines(
                json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
                json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            ), headers={"content-type": "text/event-stream"}
        ))
        adapter = make_adapter()
        async for _ in adapter.stream(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gpt-4o-mini",
            tool_choice="auto",
            response_format={"type": "json_object"},
            top_p=0.8,
            seed=99,
        ):
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == "auto"
        assert body["response_format"] == {"type": "json_object"}
        assert body["top_p"] == 0.8
        assert body["seed"] == 99
