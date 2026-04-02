# OpenAI Adapter

Complete reference for the ModelGate OpenAI adapter. Covers every supported feature, how inputs are translated to the OpenAI Chat Completions API, and how responses are normalized back.

**API Endpoint:** `https://api.openai.com/v1/chat/completions`  
**Also works with:** Any OpenAI-compatible API (Groq, Ollama, vLLM, etc.)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Supported Models](#supported-models)
- [Chat (Non-Streaming)](#chat-non-streaming)
- [Streaming](#streaming)
- [Content Types](#content-types)
  - [Text](#text)
  - [Tool Use](#tool-use)
  - [Tool Results](#tool-results)
  - [Images (Vision)](#images-vision)
- [System Prompts](#system-prompts)
- [Tool Definitions](#tool-definitions)
- [Tool Choice](#tool-choice)
- [Structured Output](#structured-output)
- [Sampling Parameters](#sampling-parameters)
- [Reasoning Models](#reasoning-models)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
- [Translation Reference](#translation-reference)
- [Test Coverage](#test-coverage)

---

## Quick Start

```python
from modelgate import ModelGate, ModelGateConfig, Message, Role

client = ModelGate(ModelGateConfig(openai_api_key="sk-..."))

# Non-streaming
response = await client.chat(
    model="openai/gpt-4o-mini",
    messages=[Message(role=Role.USER, content="Hello!")],
)
print(response.text)  # "Hello! How can I help you today?"

# Streaming
async for chunk in client.stream(
    model="openai/gpt-4o-mini",
    messages=[Message(role=Role.USER, content="Tell me a story")],
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
```

---

## Authentication

The adapter resolves the API key in this order:

1. `api_key` passed to `GenericOpenAIAdapter(api_key="...")` directly
2. `openai_api_key` on `ModelGateConfig`
3. `OPENAI_API_KEY` environment variable

All requests include:
```
Authorization: Bearer <api_key>
Content-Type: application/json
```

---

## Supported Models

Any model ID works. Pass via the `openai/` prefix:

| ModelGate String | Model ID Sent to API |
|---|---|
| `openai/gpt-4o` | `gpt-4o` |
| `openai/gpt-4o-mini` | `gpt-4o-mini` |
| `openai/o4-mini` | `o4-mini` |
| `openai/o3` | `o3` |
| `openai/gpt-4.1` | `gpt-4.1` |
| `openai/gpt-4.1-mini` | `gpt-4.1-mini` |

OpenAI-compatible APIs use their own provider prefix:

| Provider | Example |
|---|---|
| Groq | `groq/llama-3.1-70b-versatile` |
| Ollama | `ollama/llama3` |

---

## Chat (Non-Streaming)

```python
response = await client.chat(
    model="openai/gpt-4o-mini",
    messages=[...],          # required
    tools=[...],             # optional
    system="...",            # optional
    max_tokens=4096,         # optional, default 4096
    temperature=1.0,         # optional, default 1.0
    **kwargs,                # optional, see below
)
```

### Response Object

| Field | Type | Description |
|---|---|---|
| `response.id` | `str` | Completion ID (`"chatcmpl-abc123"`) |
| `response.model` | `str` | Model ID used |
| `response.content` | `list[ContentBlock]` | All content blocks |
| `response.usage` | `Usage` | Token counts |
| `response.finish_reason` | `FinishReason` | Why the model stopped |
| `response.stop_sequence` | `str \| None` | Which stop string triggered |
| `response.text` | `str \| None` | Concatenated text (convenience) |
| `response.tool_calls` | `list[ContentBlock]` | TOOL_USE blocks only (convenience) |

---

## Streaming

```python
async for chunk in client.stream(
    model="openai/gpt-4o-mini",
    messages=[...],
    # same parameters as chat()
):
    # chunk is ContentBlock or Usage
```

### Stream Protocol

```
┌─────────────────────────────────────────────────────┐
│ OpenAI SSE Events           ModelGate Yields         │
├─────────────────────────────────────────────────────┤
│ data: {delta.content}    → ContentBlock(TEXT)    ──►  │
│   (streamed immediately)                              │
│ data: {delta.tool_calls} → (buffers JSON args)        │
│ data: {finish_reason:                                 │
│          "tool_calls"}   → ContentBlock(TOOL_USE) ──► │
│ data: {usage: {...}}     → Usage               ──►   │
│ data: [DONE]             → (stream ends)              │
└─────────────────────────────────────────────────────┘
```

**Key behaviors:**
- **Text:** Yielded immediately on each delta — one `ContentBlock(TEXT)` per fragment
- **Tool use:** JSON argument fragments buffered by index, emitted as complete `ContentBlock(TOOL_USE)` at `finish_reason: "tool_calls"`
- **Parallel tool calls:** Multiple tools are tracked independently by `index` and all emitted together
- **Usage:** The adapter sends `stream_options: {include_usage: true}` — usage arrives as the final chunk

---

## Content Types

### Text

**Sending:**
```python
Message(role=Role.USER, content="Hello")
```

**Receiving:**
```python
response.text  # "Hello! How can I help?"
```

**API translation:**
```
ModelGate                              OpenAI API
─────────                              ──────────
"Hello"                          ──►   {"role": "user", "content": "Hello"}
```

---

### Tool Use

**Receiving:**
```python
for tc in response.tool_calls:
    tc.tool_call_id  # "call_abc123"
    tc.tool_name     # "get_weather"
    tc.tool_input    # {"city": "NYC"} — always a parsed dict, never a JSON string
```

**API translation (response → canonical):**
```
OpenAI API                                 ModelGate
──────────                                 ─────────
{"tool_calls": [{                    ──►   ContentBlock(
   "id": "call_abc123",                      type=TOOL_USE,
   "type": "function",                       tool_call_id="call_abc123",
   "function": {                             tool_name="get_weather",
     "name": "get_weather",                  tool_input={"city": "NYC"})
     "arguments": "{\"city\":\"NYC\"}"
   }}]}
```

> **Note:** OpenAI sends `arguments` as a JSON *string*. The adapter always parses this into a dict. If parsing fails, returns `{}`.

---

### Tool Results

**Sending tool results back:**
```python
Message(role=Role.TOOL, content=[
    ContentBlock(type=ContentType.TOOL_RESULT,
                 tool_call_id="call_abc123",
                 tool_result_content="72°F and sunny"),
])
```

**API translation (canonical → API):**
```
ModelGate                              OpenAI API
─────────                              ──────────
Message(role=TOOL, content=[     ──►   {"role": "tool",
  ContentBlock(TOOL_RESULT,              "tool_call_id": "call_abc123",
    tool_call_id="call_abc123",          "content": "72°F and sunny"}
    tool_result_content="72°F")])
```

> **Note:** OpenAI requires **one message per tool result**. If a TOOL message has multiple blocks, they're automatically flattened into separate messages.

---

### Images (Vision)

Two source types: `url` and `base64`.

**Sending:**
```python
# URL
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.TEXT, text="What's in this image?"),
    ContentBlock(type=ContentType.IMAGE, image_source_type="url",
                 image_data="https://example.com/cat.jpg"),
])

# Base64
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.IMAGE, image_source_type="base64",
                 image_media_type="image/png", image_data="iVBOR..."),
    ContentBlock(type=ContentType.TEXT, text="Describe this image."),
])
```

**API translation:**
```
ModelGate                              OpenAI API
─────────                              ──────────
ContentBlock(IMAGE,              ──►   {"type": "image_url",
  image_source_type="url",               "image_url": {
  image_data="https://...")                 "url": "https://..."}}

ContentBlock(IMAGE,              ──►   {"type": "image_url",
  image_source_type="base64",            "image_url": {
  image_media_type="image/png",            "url": "data:image/png;base64,iVBOR..."}}
  image_data="iVBOR...")
```

> **Note:** Base64 images are converted to `data:` URIs, which is the OpenAI-required format.

---

## System Prompts

Sent as the first message with `role: "system"`:

```python
await client.chat(..., system="You are a helpful assistant.")
# Becomes: [{"role": "system", "content": "You are..."}, {"role": "user", ...}]
```

---

## Tool Definitions

```python
from modelgate import Tool, ToolParameter

tool = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={"city": ToolParameter(type="string", description="City name")},
    required=["city"],
)
```

**API translation:**
```
ModelGate                              OpenAI API
─────────                              ──────────
Tool(name="get_weather",         ──►   {"type": "function",
  parameters={"city":                    "function": {
    ToolParameter(type="string")},         "name": "get_weather",
  required=["city"])                       "description": "Get weather",
                                           "parameters": {
                                             "type": "object",
                                             "properties": {
                                               "city": {"type": "string"}},
                                             "required": ["city"]}}}
```

`raw_schema` is also supported — sent directly as the `parameters` field.

---

## Tool Choice

```python
tool_choice=None                                     # omitted (API default)
tool_choice="none"                                   # don't call tools
tool_choice="auto"                                   # model decides
tool_choice="required"                               # must call a tool
tool_choice={"type": "function",
             "function": {"name": "get_weather"}}    # force specific tool
```

---

## Structured Output

```python
# JSON mode
response_format={"type": "json_object"}

# Strict schema-validated output
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        },
    },
}
```

---

## Sampling Parameters

```python
await client.chat(...,
    temperature=0.7,           # 0.0 to 2.0 (omitted for reasoning models)
    top_p=0.9,                 # nucleus sampling
    frequency_penalty=0.5,     # penalize repetition (-2.0 to 2.0)
    presence_penalty=0.3,      # penalize token presence (-2.0 to 2.0)
    stop=["END", "---"],       # custom stop sequences (up to 4)
    seed=42,                   # deterministic sampling
    max_tokens=4096,           # max output tokens
)
```

---

## Reasoning Models

For o-series models (o3, o4-mini, etc.):

```python
await client.chat(
    model="openai/o4-mini",
    messages=[...],
    reasoning_effort="medium",   # "low" | "medium" | "high"
)
```

**Behavior:**
- `reasoning_effort` is sent as a top-level field
- `temperature` is **automatically omitted** when `reasoning_effort` is set (OpenAI rejects it)
- Exception: `reasoning_effort="none"` re-enables temperature
- `reasoning_tokens` are tracked in `response.usage.thinking_tokens`

---

## Stop Reasons

| OpenAI `finish_reason` | ModelGate `FinishReason` | Meaning |
|---|---|---|
| `stop` | `STOP` | Normal completion |
| `tool_calls` | `TOOL_USE` | Model wants to call tools |
| `function_call` | `TOOL_USE` | Deprecated, same as tool_calls |
| `length` | `LENGTH` | Hit max_tokens |
| `content_filter` | `STOP` | Content filter triggered |
| `null` | `STOP` | API returned null |

---

## Error Handling

| HTTP Status | Exception |
|---|---|
| 401 | `AuthenticationError` |
| 429 | `RateLimitError` |
| 400 | `InvalidRequestError` |
| 5xx | `ProviderError` |
| Stream error | `StreamingError` |

---

## Translation Reference

### Kwargs

| Kwarg | Type | OpenAI Field | Notes |
|---|---|---|---|
| `reasoning_effort` | `str` | `reasoning_effort` | o-series only |
| `tool_choice` | `str \| dict` | `tool_choice` | `"none"`, `"auto"`, `"required"`, or dict |
| `response_format` | `dict` | `response_format` | `json_object` or `json_schema` |
| `top_p` | `float` | `top_p` | Nucleus sampling |
| `frequency_penalty` | `float` | `frequency_penalty` | -2.0 to 2.0 |
| `presence_penalty` | `float` | `presence_penalty` | -2.0 to 2.0 |
| `stop` | `list[str]` | `stop` | Up to 4 sequences |
| `seed` | `int` | `seed` | Best-effort determinism |

### Usage Fields

| Field | Source in OpenAI Response |
|---|---|
| `usage.input_tokens` | `usage.prompt_tokens` |
| `usage.output_tokens` | `usage.completion_tokens` |
| `usage.total_tokens` | Computed: `input + output` |
| `usage.thinking_tokens` | `usage.completion_tokens_details.reasoning_tokens` |

### Streaming Usage

The adapter sends `stream_options: {"include_usage": true}` — usage arrives in the final SSE event before `[DONE]`.

---

## Test Coverage

### Mocked Tests (34 tests, no API key needed)

| Test Class | Tests | What's Covered |
|---|---|---|
| `TestOpenAIChat` | 4 | Text, tool calls, parallel tools, malformed JSON |
| `TestOpenAIMessageFormat` | 5 | System prompt, tool result flattening, image URL, image base64, assistant tool_use |
| `TestOpenAIKwargs` | 8 | tool_choice (auto/required/specific/none), response_format, passthrough kwargs, reasoning_effort, temperature |
| `TestOpenAIErrors` | 4 | 401, 429, 400, 500 error mapping |
| `TestOpenAIFinishReasons` | 5 | stop, tool_calls, length, content_filter, stop_sequence |
| `TestOpenAIUsage` | 2 | Token parsing, reasoning tokens |
| `TestOpenAIToolDefs` | 2 | Tool schema, raw_schema |
| `TestOpenAIStream` | 4 | Text streaming, tool call buffering, stream error, kwargs forwarded |

### Live API Tests (38 tests, requires OPENAI_API_KEY)

Comprehensive integration tests covering real API behavior.

Run tests:
```bash
# Mocked tests (fast, no API key)
.venv/bin/python -m pytest tests/test_openai_adapter_unit.py -v

# Live API tests
.venv/bin/python -m pytest tests/test_openai_adapter.py -v
```
