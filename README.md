# modelgate

![PyPI version](https://img.shields.io/pypi/v/modelgate)
![PyPI downloads](https://img.shields.io/pypi/dm/modelgate)
![License](https://img.shields.io/github/license/PavanPapiReddy22/modelgate)

A lightweight, type-safe adapter layer that gives you one consistent interface across every major LLM provider. No bloated SDKs — just `httpx` and `pydantic` under the hood.

## Install

```bash
pip install modelgate

# AWS Bedrock support
pip install "modelgate[aws]"

# Vertex AI support
pip install "modelgate[vertex]"
```

## Quick Start

```python
import asyncio
from modelgate import ModelGate, ModelGateConfig

async def main():
    client = ModelGate(ModelGateConfig(
        openai_api_key="sk-...",
        anthropic_api_key="sk-ant-...",
    ))

    response = await client.chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.text)  # "4"

asyncio.run(main())
```

## Supported Providers

| Provider    | Model String Prefix | Config Key             |
|-------------|---------------------|------------------------|
| OpenAI      | `openai/`           | `openai_api_key`       |
| Anthropic   | `anthropic/`        | `anthropic_api_key`    |
| AWS Bedrock | `bedrock/`          | `aws_region`, `boto3_session` |
| Gemini      | `gemini/`           | `gemini_api_key`       |
| Vertex AI   | `vertex/`           | `vertex_credentials`   |
| Groq        | `groq/`             | `groq_api_key`         |
| Ollama      | `ollama/`           | `ollama_base_url`      |

Any OpenAI-compatible API works via `GenericOpenAIAdapter` — no new adapter code needed.
> ⚠️ **AWS Bedrock and Vertex AI adapters are implemented but not yet verified against live APIs. Community testing and feedback welcome — please open an issue if you run into problems.**

---

## Input

### `chat()` / `stream()` Parameters

Both methods share the same signature:

```python
response = await client.chat(
    model="anthropic/claude-sonnet-4-6",     # required — "provider/model-id"
    messages=[...],                            # required — conversation history
    tools=[...],                               # optional — tool definitions
    system="You are helpful.",                 # optional — system prompt
    max_tokens=4096,                           # optional — default 4096
    temperature=1.0,                           # optional — default 1.0
    **kwargs,                                  # optional — provider-specific extras
)
```

| Parameter     | Type                  | Default  | Description                            |
|---------------|-----------------------|----------|----------------------------------------|
| `model`       | `str`                 | required | `"provider/model-id"` format           |
| `messages`    | `list[dict\|Message]` | required | Conversation history                   |
| `tools`       | `list[Tool]\|None`    | `None`   | Tools available to the model           |
| `system`      | `str\|None`           | `None`   | System prompt                          |
| `max_tokens`  | `int`                 | `4096`   | Maximum tokens to generate             |
| `temperature` | `float`               | `1.0`    | Sampling temperature                   |

---

### Messages

Messages can be raw dicts (auto-coerced) or `Message` objects:

```python
# Raw dicts — simplest way
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is 2+2?"},
]

# Message objects — explicit typing
from modelgate import Message, Role
messages = [
    Message(role=Role.USER, content="Hello"),
    Message(role=Role.ASSISTANT, content="Hi there!"),
    Message(role=Role.USER, content="What is 2+2?"),
]
```

Content can be a **string** or a **list of ContentBlocks** for rich content:

```python
from modelgate import Message, ContentBlock, ContentType, Role

# String content
Message(role=Role.USER, content="Describe this image")

# Rich content — mix text, images, documents in one message
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.IMAGE,
                 image_source_type="base64",
                 image_media_type="image/png",
                 image_data="iVBOR..."),
    ContentBlock(type=ContentType.TEXT, text="What's in this image?"),
])
```

### Content Block Types (Input)

| Type | Purpose | Key Fields |
|------|---------|------------|
| `TEXT` | Plain text | `text` |
| `IMAGE` | Image (base64, URL, or file) | `image_source_type`, `image_media_type`, `image_data` |
| `DOCUMENT` | PDF/text document | `document_source_type`, `document_media_type`, `document_data`, `document_filename` |
| `TOOL_RESULT` | Send tool output back | `tool_call_id`, `tool_result_content` |
| `TOOL_USE` | Round-trip previous tool call | `tool_call_id`, `tool_name`, `tool_input` |
| `THINKING` | Round-trip thinking block | `thinking`, `thinking_signature` |
| `REDACTED_THINKING` | Round-trip redacted thinking | `redacted_thinking_data` |

```python
# Text
ContentBlock(type=ContentType.TEXT, text="Hello world")

# Image (base64)
ContentBlock(type=ContentType.IMAGE,
             image_source_type="base64",
             image_media_type="image/png",
             image_data="iVBOR...")

# Image (URL)
ContentBlock(type=ContentType.IMAGE,
             image_source_type="url",
             image_data="https://example.com/photo.jpg")

# Document (PDF)
ContentBlock(type=ContentType.DOCUMENT,
             document_source_type="base64",
             document_media_type="application/pdf",
             document_data="JVBE...",
             document_filename="report.pdf")

# Tool result
ContentBlock(type=ContentType.TOOL_RESULT,
             tool_call_id="toolu_123",
             tool_result_content="72°F and sunny")
```

---

### Tools

Define tools with `ToolParameter` (simple) or `raw_schema` (complex):

```python
from modelgate import Tool, ToolParameter

# Simple tool
weather = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
        "unit": ToolParameter(type="string", enum=["celsius", "fahrenheit"]),
    },
    required=["location"],
)

# Complex tool — raw JSON schema for nested objects, arrays, etc.
search = Tool(
    name="search",
    description="Search documents",
    raw_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "array",
                "items": {"type": "object",
                          "properties": {"field": {"type": "string"},
                                         "value": {"type": "string"}}},
            },
        },
        "required": ["query"],
    },
)
```

When `raw_schema` is set, it's sent directly to the provider and `parameters` is ignored.

---

### Provider-Specific Extras (`**kwargs`)

Extra keyword arguments are forwarded to the specific provider. Example for Anthropic:

```python
await client.chat(
    model="anthropic/claude-sonnet-4-6",
    messages=[...],
    thinking_budget=10000,              # extended thinking budget (int or "adaptive")
    thinking_display="summarized",      # "summarized" | "omitted"
    tool_choice="auto",                 # "auto" | "any" | "none" | {"type":"tool","name":"..."}
    output_config={"format": {...}},    # structured JSON output
    top_p=0.9,                          # nucleus sampling
    top_k=40,                           # top-k sampling
    stop_sequences=["END", "STOP"],     # custom stop strings
    metadata={"user_id": "u123"},       # request metadata
    service_tier="auto",                # service tier
    interleaved_thinking=True,          # beta header
)
```

---

## Output

### Non-Streaming — `chat()`

Returns a `Response` object:

```python
response = await client.chat(model="anthropic/claude-sonnet-4-6", messages=[...])
```

| Field | Type | Description |
|-------|------|-------------|
| `response.id` | `str` | Message ID |
| `response.model` | `str` | Model used |
| `response.content` | `list[ContentBlock]` | All content blocks |
| `response.usage` | `Usage` | Token counts |
| `response.finish_reason` | `FinishReason` | Why the model stopped |
| `response.stop_sequence` | `str \| None` | Stop string that triggered |

**Convenience properties:**

```python
response.text        # str | None — concatenated text from all TEXT blocks
response.tool_calls  # list[ContentBlock] — only TOOL_USE blocks
response.thinking    # str | None — concatenated thinking text
```

### Content Block Types (Output)

```python
for block in response.content:
    match block.type:
        case ContentType.TEXT:
            print(block.text)                    # "Hello! How can I help?"

        case ContentType.TOOL_USE:
            print(block.tool_call_id)            # "toolu_01XF..."
            print(block.tool_name)               # "get_weather"
            print(block.tool_input)              # {"location": "NYC"}  — always a dict

        case ContentType.THINKING:
            print(block.thinking)                # "Let me analyze..."
            print(block.thinking_signature)      # "WaUjzkypQ2m..."

        case ContentType.REDACTED_THINKING:
            print(block.redacted_thinking_data)  # opaque encrypted data

        case ContentType.SERVER_TOOL_USE:
            print(block.tool_name)               # "web_search"
            print(block.tool_input)              # {"query": "latest news"}
```

### Usage

```python
response.usage.input_tokens                 # tokens in the prompt
response.usage.output_tokens                # tokens generated
response.usage.total_tokens                 # always input + output
response.usage.thinking_tokens              # extended thinking tokens (Anthropic)
response.usage.cache_read_input_tokens      # tokens read from cache
response.usage.cache_creation_input_tokens  # tokens written to cache
```

### Finish Reasons

| Value | Meaning |
|-------|---------|
| `FinishReason.STOP` | Normal completion |
| `FinishReason.TOOL_USE` | Model wants to call a tool |
| `FinishReason.LENGTH` | Hit `max_tokens` limit |
| `FinishReason.ERROR` | Error occurred |
| `FinishReason.PAUSE_TURN` | Long turn paused — can resume |
| `FinishReason.REFUSAL` | Safety refusal |

---

### Streaming — `stream()`

Yields `ContentBlock` chunks followed by a final `Usage`:

```python
async for chunk in client.stream(model="anthropic/claude-sonnet-4-6", messages=[...]):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)       # streamed word by word
    elif chunk.type == "tool_use":
        print(f"Tool: {chunk.tool_name}({chunk.tool_input})")  # complete tool call
    elif chunk.type == "thinking":
        print(f"Thinking: {chunk.thinking[:50]}...")  # complete thinking block
    elif chunk.type == "usage":
        print(f"Tokens: {chunk.input_tokens} in, {chunk.output_tokens} out")
```

| Chunk Type | When | Count |
|------------|------|-------|
| `ContentBlock(TEXT)` | Each text fragment as it arrives | Many |
| `ContentBlock(TOOL_USE)` | When a tool call is complete | One per tool |
| `ContentBlock(THINKING)` | When thinking block is complete | One per block |
| `ContentBlock(REDACTED_THINKING)` | When block is complete | One per block |
| `ContentBlock(SERVER_TOOL_USE)` | When server tool call is complete | One per call |
| `Usage` | End of stream | Always last |

---

## Tool Use — Full Loop

```python
from modelgate import ModelGate, ModelGateConfig, Message, ContentBlock, ContentType, Role, Tool, ToolParameter

client = ModelGate(ModelGateConfig(anthropic_api_key="sk-ant-..."))

weather = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={"location": ToolParameter(type="string", description="City")},
    required=["location"],
)

messages = [Message(role=Role.USER, content="What's the weather in NYC?")]

# 1. Send request with tools
response = await client.chat(
    model="anthropic/claude-sonnet-4-6",
    messages=messages,
    tools=[weather],
)

# 2. Model returns a tool call
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = get_weather(tool_call.tool_input["location"])  # your function

    # 3. Send tool result back
    messages.append(Message(role=Role.ASSISTANT, content=response.content))
    messages.append(Message(role=Role.TOOL, content=[
        ContentBlock(
            type=ContentType.TOOL_RESULT,
            tool_call_id=tool_call.tool_call_id,
            tool_result_content=result,
        ),
    ]))

    # 4. Get final response
    final = await client.chat(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        tools=[weather],
    )
    print(final.text)  # "It's 72°F and sunny in NYC!"
```

---

## Error Handling

All provider errors are normalized into typed exceptions:

```python
from modelgate import AuthenticationError, RateLimitError, InvalidRequestError, StreamingError

try:
    response = await client.chat(...)
except AuthenticationError:
    pass  # 401 — bad or missing API key
except RateLimitError:
    pass  # 429 — rate limited, retry with backoff
except InvalidRequestError:
    pass  # 400 — malformed request
except StreamingError:
    pass  # error mid-stream
```

Error hierarchy:

```
ModelGateError
├── AuthenticationError   # 401
├── RateLimitError        # 429
├── InvalidRequestError   # 400
├── ProviderError         # 5xx
│   ├── BedrockError
│   └── VertexError
└── StreamingError        # error mid-stream
```

---

## Serialization

Both `ContentBlock` and `Response` exclude `None` fields by default for clean output:

```python
response.model_dump()
# {"type": "text", "text": "Hello"}
# NOT: {"type": "text", "text": "Hello", "tool_call_id": null, "tool_name": null, ...}
```

---

## `ModelGateConfig` Reference

| Field               | Type        | Default                            | Description                     |
|---------------------|-------------|------------------------------------|---------------------------------|
| `openai_api_key`    | `str\|None` | `None`                             | OpenAI API key                  |
| `anthropic_api_key` | `str\|None` | `None`                             | Anthropic API key               |
| `gemini_api_key`    | `str\|None` | `None`                             | Gemini API key                  |
| `groq_api_key`      | `str\|None` | `None`                             | Groq API key                    |
| `aws_region`        | `str`       | `"us-east-1"`                      | AWS region for Bedrock          |
| `boto3_session`     | `Any\|None` | `None`                             | Custom boto3 session            |
| `vertex_credentials`| `Any\|None` | `None`                             | Google auth credentials         |
| `ollama_base_url`   | `str`       | `"http://localhost:11434/v1"`      | Ollama server URL               |
| `groq_base_url`     | `str`       | `"https://api.groq.com/openai/v1"` | Groq API base URL              |

API keys can also be set via environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
