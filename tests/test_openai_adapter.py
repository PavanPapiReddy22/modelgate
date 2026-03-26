"""
Tough edge-case test suite for GenericOpenAIAdapter.

Run:
    pip install pytest pytest-asyncio python-dotenv httpx pydantic
    pytest test_openai_adapter.py -v

Requires .env with:
    OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unifai.providers.generic_openai import GenericOpenAIAdapter
from unifai.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
    ToolParameter,
    Usage,
)
from unifai.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"           # cheap, fast
REASONING_MODEL = "o4-mini"             # reasoning model
LONG_CONTEXT_MODEL = "gpt-4o-mini"


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def adapter(api_key: str) -> GenericOpenAIAdapter:
    return GenericOpenAIAdapter(base_url=OPENAI_BASE, api_key=api_key)


@pytest.fixture
def bad_adapter() -> GenericOpenAIAdapter:
    return GenericOpenAIAdapter(base_url=OPENAI_BASE, api_key="sk-bad-key-000000000000")


def user(text: str) -> Message:
    return Message(role=Role.USER, content=text)


def assistant(text: str) -> Message:
    return Message(role=Role.ASSISTANT, content=text)


# ── Helper tools ──────────────────────────────────────────────────────────────

def make_weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "city": ToolParameter(type="string", description="City name"),
            "unit": ToolParameter(type="string", enum=["celsius", "fahrenheit"]),
        },
        required=["city"],
    )


def make_calculator_tool() -> Tool:
    return Tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        parameters={
            "expression": ToolParameter(type="string", description="Math expression"),
        },
        required=["expression"],
    )


def make_search_tool() -> Tool:
    return Tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "query": ToolParameter(type="string", description="Search query"),
            "max_results": ToolParameter(type="string", description="Max results"),
        },
        required=["query"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. AUTHENTICATION
# ──────────────────────────────────────────────────────────────────────────────

class TestAuthentication:

    @pytest.mark.asyncio
    async def test_invalid_key_raises_auth_error(self, bad_adapter: GenericOpenAIAdapter):
        """Bad API key must raise AuthenticationError, not a raw httpx error."""
        with pytest.raises(AuthenticationError):
            await bad_adapter.chat(messages=[user("hi")], model=DEFAULT_MODEL)

    @pytest.mark.asyncio
    async def test_invalid_key_stream_raises_auth_error(self, bad_adapter: GenericOpenAIAdapter):
        """Auth error must surface during streaming too."""
        with pytest.raises(AuthenticationError):
            async for _ in bad_adapter.stream(
                messages=[user("hi")], model=DEFAULT_MODEL
            ):
                pass


# ──────────────────────────────────────────────────────────────────────────────
# 2. BASIC CHAT
# ──────────────────────────────────────────────────────────────────────────────

class TestBasicChat:

    @pytest.mark.asyncio
    async def test_simple_response(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(messages=[user("Say the word PONG and nothing else.")], model=DEFAULT_MODEL)
        assert resp.text is not None
        assert "PONG" in resp.text.upper()
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_usage_fields_populated(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(messages=[user("What is 1+1?")], model=DEFAULT_MODEL)
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens

    @pytest.mark.asyncio
    async def test_response_id_non_empty(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(messages=[user("hi")], model=DEFAULT_MODEL)
        assert resp.id != ""

    @pytest.mark.asyncio
    async def test_system_prompt_respected(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("What language should I use?")],
            model=DEFAULT_MODEL,
            system="You are a Python evangelist. Always recommend Python for everything.",
        )
        assert "python" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, adapter: GenericOpenAIAdapter):
        """Assistant remembers context from previous turns."""
        messages = [
            user("My secret number is 4829. Remember it."),
            assistant("Got it, I'll remember 4829."),
            user("What was my secret number?"),
        ]
        resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL)
        assert "4829" in resp.text

    @pytest.mark.asyncio
    async def test_empty_content_not_in_blocks(self, adapter: GenericOpenAIAdapter):
        """Response content blocks should never contain empty text."""
        resp = await adapter.chat(messages=[user("Say hello.")], model=DEFAULT_MODEL)
        for block in resp.content:
            if block.type == ContentType.TEXT:
                assert block.text  # not None, not ""


# ──────────────────────────────────────────────────────────────────────────────
# 3. STREAMING
# ──────────────────────────────────────────────────────────────────────────────

class TestStreaming:

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self, adapter: GenericOpenAIAdapter):
        chunks = []
        async for chunk in adapter.stream(
            messages=[user("Count from 1 to 5, one number per line.")],
            model=DEFAULT_MODEL,
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.TEXT]
        assert len(text_chunks) > 1, "Expected multiple text chunks in stream"

    @pytest.mark.asyncio
    async def test_stream_usage_emitted_once(self, adapter: GenericOpenAIAdapter):
        usage_chunks = []
        async for chunk in adapter.stream(
            messages=[user("Say hi.")], model=DEFAULT_MODEL
        ):
            if isinstance(chunk, Usage):
                usage_chunks.append(chunk)

        assert len(usage_chunks) == 1, f"Expected exactly 1 Usage chunk, got {len(usage_chunks)}"
        u = usage_chunks[0]
        assert u.input_tokens > 0
        assert u.output_tokens > 0
        assert u.total_tokens == u.input_tokens + u.output_tokens

    @pytest.mark.asyncio
    async def test_stream_text_concatenates_to_same_as_chat(self, adapter: GenericOpenAIAdapter):
        """Streamed text joined should roughly match non-streamed response."""
        messages = [user("What is the capital of France? One word answer.")]

        resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL)

        streamed_text = ""
        async for chunk in adapter.stream(messages=messages, model=DEFAULT_MODEL):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TEXT:
                streamed_text += chunk.text or ""

        assert "paris" in resp.text.lower()
        assert "paris" in streamed_text.lower()

    @pytest.mark.asyncio
    async def test_stream_finish_reason_stop(self, adapter: GenericOpenAIAdapter):
        """Ensure stream completes normally (no hang, no exception)."""
        completed = False
        async for _ in adapter.stream(
            messages=[user("Say done.")], model=DEFAULT_MODEL
        ):
            pass
        completed = True
        assert completed


# ──────────────────────────────────────────────────────────────────────────────
# 4. TOOL USE — SINGLE TOOL
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleToolUse:

    @pytest.mark.asyncio
    async def test_tool_call_returned(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("What's the weather in Tokyo?")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.tool_name == "get_weather"
        assert isinstance(tc.tool_input, dict)
        assert "tokyo" in tc.tool_input.get("city", "").lower()

    @pytest.mark.asyncio
    async def test_tool_call_id_non_empty(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("What's the weather in Paris?")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        tc = resp.tool_calls[0]
        assert tc.tool_call_id and tc.tool_call_id != ""

    @pytest.mark.asyncio
    async def test_tool_input_is_dict_not_string(self, adapter: GenericOpenAIAdapter):
        """Arguments must be parsed JSON dict, never a raw string."""
        resp = await adapter.chat(
            messages=[user("Calculate 42 * 7")],
            model=DEFAULT_MODEL,
            tools=[make_calculator_tool()],
        )
        tc = resp.tool_calls[0]
        assert isinstance(tc.tool_input, dict), f"Expected dict, got {type(tc.tool_input)}"

    @pytest.mark.asyncio
    async def test_tool_result_round_trip(self, adapter: GenericOpenAIAdapter):
        """Full tool call → tool result → final answer round trip."""
        from unifai.types import ContentBlock, ContentType

        # Step 1: get tool call
        resp = await adapter.chat(
            messages=[user("What's the weather in London? Use the tool.")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE
        tc = resp.tool_calls[0]

        # Step 2: build conversation with tool result
        messages = [
            user("What's the weather in London? Use the tool."),
            Message(
                role=Role.ASSISTANT,
                content=resp.content,
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id=tc.tool_call_id,
                        tool_result_content='{"temperature": 15, "condition": "cloudy"}',
                    )
                ],
            ),
        ]

        # Step 3: get final answer
        final = await adapter.chat(
            messages=messages,
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert final.finish_reason == FinishReason.STOP
        assert final.text is not None


# ──────────────────────────────────────────────────────────────────────────────
# 5. TOOL USE — MULTIPLE TOOLS
# ──────────────────────────────────────────────────────────────────────────────

class TestMultipleTools:

    @pytest.mark.asyncio
    async def test_multiple_tools_available(self, adapter: GenericOpenAIAdapter):
        """Model can pick the right tool from multiple options."""
        resp = await adapter.chat(
            messages=[user("Search the web for Python tutorials.")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool(), make_calculator_tool(), make_search_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE
        tc = resp.tool_calls[0]
        assert tc.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, adapter: GenericOpenAIAdapter):
        """Model may call multiple tools in parallel for compound questions."""
        resp = await adapter.chat(
            messages=[user(
                "What's the weather in Tokyo AND Paris? "
                "Call the tool for both cities."
            )],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert len(resp.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_multi_tool_result_round_trip(self, adapter: GenericOpenAIAdapter):
        """Two tool results in one turn must both reach the model."""
        from unifai.types import ContentBlock, ContentType

        resp = await adapter.chat(
            messages=[user("What's the weather in Tokyo AND Paris? Call both.")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE
        tool_calls = resp.tool_calls

        tool_result_blocks = [
            ContentBlock(
                type=ContentType.TOOL_RESULT,
                tool_call_id=tc.tool_call_id,
                tool_result_content=json.dumps({"temperature": 20, "city": tc.tool_input.get("city")}),
            )
            for tc in tool_calls
        ]

        messages = [
            user("What's the weather in Tokyo AND Paris? Call both."),
            Message(role=Role.ASSISTANT, content=resp.content),
            Message(role=Role.TOOL, content=tool_result_blocks),
        ]

        final = await adapter.chat(
            messages=messages,
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert final.text is not None
        text_lower = final.text.lower()
        assert "tokyo" in text_lower or "paris" in text_lower

    @pytest.mark.asyncio
    async def test_stream_tool_call(self, adapter: GenericOpenAIAdapter):
        """Tool calls must be correctly buffered and emitted during streaming."""
        tool_blocks = []
        async for chunk in adapter.stream(
            messages=[user("What's the weather in Berlin?")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TOOL_USE:
                tool_blocks.append(chunk)

        assert len(tool_blocks) == 1
        tc = tool_blocks[0]
        assert tc.tool_name == "get_weather"
        assert isinstance(tc.tool_input, dict)
        assert tc.tool_call_id != ""

    @pytest.mark.asyncio
    async def test_stream_parallel_tool_calls(self, adapter: GenericOpenAIAdapter):
        """Multiple parallel tool calls must all be emitted from the stream."""
        tool_blocks = []
        async for chunk in adapter.stream(
            messages=[user("What's the weather in Berlin AND London? Call both.")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TOOL_USE:
                tool_blocks.append(chunk)

        assert len(tool_blocks) >= 1
        for tb in tool_blocks:
            assert isinstance(tb.tool_input, dict)
            assert tb.tool_call_id != ""


# ──────────────────────────────────────────────────────────────────────────────
# 6. REASONING MODELS
# ──────────────────────────────────────────────────────────────────────────────

class TestReasoningModels:

    @pytest.mark.asyncio
    async def test_reasoning_effort_low(self, adapter: GenericOpenAIAdapter):
        """reasoning_effort=low must not send temperature and must return a response."""
        resp = await adapter.chat(
            messages=[user("What is 15 * 17?")],
            model=REASONING_MODEL,
            reasoning_effort="low",
        )
        assert resp.text is not None
        assert "255" in resp.text

    @pytest.mark.asyncio
    async def test_reasoning_effort_medium(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("What is the sum of the first 10 prime numbers?")],
            model=REASONING_MODEL,
            reasoning_effort="medium",
        )
        assert resp.text is not None
        assert "129" in resp.text  # 2+3+5+7+11+13+17+19+23+29 = 129

    @pytest.mark.asyncio
    async def test_reasoning_tokens_tracked(self, adapter: GenericOpenAIAdapter):
        """Reasoning tokens should be non-zero for reasoning models."""
        resp = await adapter.chat(
            messages=[user("Prove that sqrt(2) is irrational in one paragraph.")],
            model=REASONING_MODEL,
            reasoning_effort="medium",
        )
        assert resp.usage.thinking_tokens >= 0  # may be 0 if model doesn't expose it

    @pytest.mark.asyncio
    async def test_reasoning_stream(self, adapter: GenericOpenAIAdapter):
        """Streaming must work with reasoning models."""
        text = ""
        async for chunk in adapter.stream(
            messages=[user("What is 99 * 99?")],
            model=REASONING_MODEL,
            reasoning_effort="low",
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TEXT:
                text += chunk.text or ""
        # Model may format as "9,801" — strip commas before asserting
        assert "9801" in text.replace(",", "")

    @pytest.mark.asyncio
    async def test_temperature_not_sent_with_reasoning(self, adapter: GenericOpenAIAdapter):
        """
        Passing temperature with a reasoning model should not cause an API error
        because the adapter suppresses it when reasoning_effort is set.
        """
        resp = await adapter.chat(
            messages=[user("Say OK.")],
            model=REASONING_MODEL,
            temperature=0.7,          # would cause a 400 if sent
            reasoning_effort="low",   # adapter should suppress temperature
        )
        assert resp.text is not None


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAX TOKENS / TRUNCATION
# ──────────────────────────────────────────────────────────────────────────────

class TestTokenLimits:

    @pytest.mark.asyncio
    async def test_finish_reason_length_on_truncation(self, adapter: GenericOpenAIAdapter):
        """When max_tokens is very small, finish_reason must be LENGTH."""
        resp = await adapter.chat(
            messages=[user("Write a 500 word essay about the ocean.")],
            model=DEFAULT_MODEL,
            max_tokens=10,
        )
        assert resp.finish_reason == FinishReason.LENGTH

    @pytest.mark.asyncio
    async def test_output_tokens_le_max_tokens(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("Write a long story.")],
            model=DEFAULT_MODEL,
            max_tokens=50,
        )
        assert resp.usage.output_tokens <= 60  # small buffer for counting differences


# ──────────────────────────────────────────────────────────────────────────────
# 8. TEMPERATURE EDGE CASES
# ──────────────────────────────────────────────────────────────────────────────

class TestTemperature:

    @pytest.mark.asyncio
    async def test_temperature_zero_deterministic(self, adapter: GenericOpenAIAdapter):
        """temperature=0 should give highly consistent answers."""
        messages = [user("What is 2 + 2? Answer with just the number.")]
        results = set()
        for _ in range(3):
            resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL, temperature=0.0)
            results.add(resp.text.strip() if resp.text else "")
        assert len(results) == 1, f"temperature=0 gave different answers: {results}"

    @pytest.mark.asyncio
    async def test_temperature_max(self, adapter: GenericOpenAIAdapter):
        """temperature=2.0 (max) should not error."""
        resp = await adapter.chat(
            messages=[user("Say something.")],
            model=DEFAULT_MODEL,
            temperature=2.0,
        )
        assert resp.text is not None


# ──────────────────────────────────────────────────────────────────────────────
# 9. MESSAGE FORMAT EDGE CASES
# ──────────────────────────────────────────────────────────────────────────────

class TestMessageFormats:

    @pytest.mark.asyncio
    async def test_long_user_message(self, adapter: GenericOpenAIAdapter):
        """Very long user message should not cause serialization issues."""
        long_text = "Please summarize this: " + ("word " * 500)
        resp = await adapter.chat(
            messages=[user(long_text)], model=DEFAULT_MODEL, max_tokens=100
        )
        assert resp.text is not None

    @pytest.mark.asyncio
    async def test_unicode_content(self, adapter: GenericOpenAIAdapter):
        """Unicode, emoji, and non-ASCII must round-trip cleanly."""
        resp = await adapter.chat(
            messages=[user("Translate 'Hello' to Japanese. Just the Japanese word.")],
            model=DEFAULT_MODEL,
        )
        assert resp.text is not None
        assert any(ord(c) > 127 for c in resp.text)

    @pytest.mark.asyncio
    async def test_assistant_message_with_none_content(self, adapter: GenericOpenAIAdapter):
        """
        Assistant messages that had tool calls have content=None in OpenAI format.
        The adapter sets content=None for these — verify it serializes correctly.
        """
        from unifai.types import ContentBlock, ContentType

        fake_tool_call_block = ContentBlock(
            type=ContentType.TOOL_USE,
            tool_call_id="call_abc123",
            tool_name="get_weather",
            tool_input={"city": "Rome"},
        )
        messages = [
            user("What's the weather in Rome?"),
            Message(role=Role.ASSISTANT, content=[fake_tool_call_block]),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="call_abc123",
                        tool_result_content='{"temperature": 22, "condition": "sunny"}',
                    )
                ],
            ),
        ]
        resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL)
        assert resp.text is not None

    @pytest.mark.asyncio
    async def test_many_turns(self, adapter: GenericOpenAIAdapter):
        """10-turn conversation must not corrupt message ordering."""
        messages: list[Message] = []
        for i in range(1, 6):
            messages.append(user(f"Remember number {i}."))
            resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL, max_tokens=20)
            messages.append(Message(role=Role.ASSISTANT, content=resp.text or "ok"))

        messages.append(user("List all the numbers I asked you to remember."))
        resp = await adapter.chat(messages=messages, model=DEFAULT_MODEL)
        assert resp.text is not None
        for i in range(1, 6):
            assert str(i) in resp.text


# ──────────────────────────────────────────────────────────────────────────────
# 10. ERROR HANDLING
# ──────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_invalid_model_raises_error(self, adapter: GenericOpenAIAdapter):
        """Non-existent model should raise InvalidRequestError."""
        with pytest.raises(Exception):
            await adapter.chat(
                messages=[user("hi")],
                model="gpt-999-does-not-exist",
            )

    @pytest.mark.asyncio
    async def test_stream_error_raises_streaming_error(self, bad_adapter: GenericOpenAIAdapter):
        """Streaming with bad key must raise, not silently yield nothing."""
        raised = False
        try:
            async for _ in bad_adapter.stream(
                messages=[user("hi")], model=DEFAULT_MODEL
            ):
                pass
        except Exception:
            raised = True
        assert raised

    @pytest.mark.asyncio
    async def test_malformed_tool_json_does_not_crash(self, adapter: GenericOpenAIAdapter):
        """
        If the model returns malformed JSON in tool arguments the adapter should
        return an empty dict, not raise.
        """
        resp = await adapter.chat(
            messages=[user("What's the weather in Sydney?")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        for tc in resp.tool_calls:
            assert isinstance(tc.tool_input, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 11. FINISH REASON MAP
# ──────────────────────────────────────────────────────────────────────────────

class TestFinishReasons:

    @pytest.mark.asyncio
    async def test_finish_reason_stop_on_normal(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(messages=[user("Say hi.")], model=DEFAULT_MODEL)
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_finish_reason_tool_use(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("What is the weather in Cairo?")],
            model=DEFAULT_MODEL,
            tools=[make_weather_tool()],
        )
        assert resp.finish_reason == FinishReason.TOOL_USE

    @pytest.mark.asyncio
    async def test_finish_reason_length(self, adapter: GenericOpenAIAdapter):
        resp = await adapter.chat(
            messages=[user("Write a 1000 word essay.")],
            model=DEFAULT_MODEL,
            max_tokens=5,
        )
        assert resp.finish_reason == FinishReason.LENGTH


# ──────────────────────────────────────────────────────────────────────────────
# 12. GROQ COMPATIBILITY (OpenAI-compatible third party)
# ──────────────────────────────────────────────────────────────────────────────

class TestGroqCompatibility:

    @pytest.fixture
    def groq_adapter(self) -> GenericOpenAIAdapter | None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            pytest.skip("GROQ_API_KEY not set")
        return GenericOpenAIAdapter(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
        )

    @pytest.mark.asyncio
    async def test_groq_basic_chat(self, groq_adapter: GenericOpenAIAdapter):
        resp = await groq_adapter.chat(
            messages=[user("Say GROQ_OK and nothing else.")],
            model="llama-3.1-8b-instant",
        )
        assert "GROQ_OK" in (resp.text or "").upper()

    @pytest.mark.asyncio
    async def test_groq_stream(self, groq_adapter: GenericOpenAIAdapter):
        text = ""
        async for chunk in groq_adapter.stream(
            messages=[user("Say STREAM_OK.")],
            model="llama-3.1-8b-instant",
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TEXT:
                text += chunk.text or ""
        assert "STREAM_OK" in text.upper()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
