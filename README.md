# unifai

A minimalist, model-agnostic adapter layer for LLMs. No massive SDKs, strict type-safe normalization, zero-overhead abstraction — just `pydantic`, `httpx`, and `boto3`. Unlike LiteLLM, unifai calls provider APIs directly rather than wrapping heavyweight SDKs, giving you a predictable canonical schema with nothing hidden underneath.

## Install

```bash
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from unifai import UnifAI, UnifAIConfig

async def main():
    client = UnifAI(UnifAIConfig(
        openai_api_key="sk-...",
        anthropic_api_key="sk-ant-...",
    ))

    # Non-streaming
    response = await client.chat(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.text)       # "4"
    print(response.tool_calls) # [] — same shape for ALL providers

    # Streaming (yields ContentBlock chunks, then a final Usage)
    async for chunk in client.stream(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
    ):
        if chunk.type == "text":
            print(chunk.text, end="", flush=True)

asyncio.run(main())
```

## Supported Providers

| Example Model string | Adapter | Base URL override | Status |
|---|---|---|---|
| `openai/gpt-4o` | `OpenAIAdapter` | — | ✅ Full |
| `anthropic/claude-3-5-sonnet-20241022` | `AnthropicAdapter` | — | ✅ Full |
| `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | `BedrockAdapter` | — | ✅ Full |
| `groq/llama-3.1-70b-versatile` | `GenericOpenAIAdapter` | `https://api.groq.com/openai/v1` | ✅ Full |
| `ollama/mistral` | `GenericOpenAIAdapter` | `http://localhost:11434/v1` | ✅ Full |
| `gemini/gemini-2.0-flash` | `GeminiAdapter` | — | ✅ Full |
| `vertex/gemini-2.0-flash` | `VertexAdapter` | — | ✅ Full |

Any OpenAI-compatible API can be added by pointing `GenericOpenAIAdapter` at a new `base_url` — no new adapter code required.

## Tool Use

Tools produce the exact same `ContentBlock` shape regardless of provider:

```python
from unifai import Tool, ToolParameter

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
    },
    required=["location"],
)

response = await client.chat(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=[weather_tool],
)

for tc in response.tool_calls:
    print(tc.tool_name)   # "get_weather"
    print(tc.tool_input)  # {"location": "NYC"} — always a dict, never a string
```

## Error Handling

Every adapter catches raw `httpx.HTTPStatusError` and re-raises as a typed `UnifAIError` — provider-specific error formats never leak to your code:

```
UnifAIError
├── AuthenticationError   # 401 — invalid or missing API key
├── RateLimitError        # 429 — provider rate limit exceeded
├── InvalidRequestError   # 400 — malformed input
├── ProviderError         # 5xx — unexpected provider failure
│   ├── BedrockError
│   └── VertexError
└── StreamingError        # error mid-stream
```

```python
from unifai import RateLimitError, AuthenticationError

try:
    response = await client.chat(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
except RateLimitError:
    # retry with backoff
except AuthenticationError:
    # bad key
```

## Architecture

```
src/unifai/
├── __init__.py          # Public API surface
├── types.py             # Pydantic v2 canonical schemas
├── errors.py            # UnifAIError hierarchy
├── client.py            # UnifAI entry point + provider routing
└── providers/
    ├── base.py          # BaseProvider ABC
    ├── openai.py        # OpenAI adapter
    ├── anthropic.py     # Anthropic adapter
    ├── bedrock.py       # AWS Bedrock Converse API
    ├── gemini.py        # Gemini (stub)
    ├── vertex.py        # Vertex AI (stub)
    └── generic_openai.py  # OpenAI-compatible fallback
```

## Testing

```bash
pytest tests/ -v
```

## Dependencies

- `pydantic` — type-safe models
- `httpx` — async HTTP (no provider SDKs)
- `boto3` — AWS credential signing only
