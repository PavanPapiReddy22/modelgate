# Anthropic Adapter

Complete reference for the ModelGate Anthropic adapter. Covers every supported feature, how inputs are translated to the Anthropic Messages API, and how responses are normalized back.

**API Endpoint:** `https://api.anthropic.com/v1/messages`
**API Version:** `2023-06-01`

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
  - [Images](#images)
  - [Documents](#documents)
  - [Extended Thinking](#extended-thinking)
  - [Redacted Thinking](#redacted-thinking)
  - [Server Tool Use](#server-tool-use)
- [System Prompts](#system-prompts)
- [Tool Definitions](#tool-definitions)
- [Tool Choice](#tool-choice)
- [Structured Output](#structured-output)
- [Sampling Parameters](#sampling-parameters)
- [Extended Thinking](#extended-thinking-configuration)
- [Prompt Caching](#prompt-caching)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
- [Translation Reference](#translation-reference)
- [Test Coverage](#test-coverage)

---

## Quick Start

```python
from modelgate import ModelGate, ModelGateConfig, Message, Role

client = ModelGate(ModelGateConfig(anthropic_api_key="sk-ant-..."))

# Non-streaming
response = await client.chat(
    model="anthropic/claude-sonnet-4-6",
    messages=[Message(role=Role.USER, content="Hello!")],
)
print(response.text)  # "Hello! How can I help you today?"

# Streaming
async for chunk in client.stream(
    model="anthropic/claude-sonnet-4-6",
    messages=[Message(role=Role.USER, content="Tell me a story")],
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
```

---

## Authentication

The adapter resolves the API key in this order:

1. `api_key` passed to `AnthropicAdapter(api_key="...")` directly
2. `anthropic_api_key` on `ModelGateConfig`
3. `ANTHROPIC_API_KEY` environment variable

All requests include:
```
x-api-key: <api_key>
anthropic-version: 2023-06-01
Content-Type: application/json
```

---

## Supported Models

Any Anthropic model ID works. Pass via the `anthropic/` prefix:

| ModelGate String | Model ID Sent to API |
|---|---|
| `anthropic/claude-opus-4-6` | `claude-opus-4-6` |
| `anthropic/claude-sonnet-4-6` | `claude-sonnet-4-6` |
| `anthropic/claude-haiku-4` | `claude-haiku-4` |
| `anthropic/claude-3-5-sonnet-20241022` | `claude-3-5-sonnet-20241022` |

---

## Chat (Non-Streaming)

```python
response = await client.chat(
    model="anthropic/claude-sonnet-4-6",
    messages=[...],          # required
    tools=[...],             # optional
    system="...",            # optional
    max_tokens=4096,         # optional, default 4096
    temperature=1.0,         # optional, default 1.0
    **kwargs,                # optional, see below
)
```

**Returns:** `Response` object with:

| Field | Type | Description |
|---|---|---|
| `response.id` | `str` | Message ID (`"msg_abc123"`) |
| `response.model` | `str` | Model ID used |
| `response.content` | `list[ContentBlock]` | All content blocks |
| `response.usage` | `Usage` | Token counts |
| `response.finish_reason` | `FinishReason` | Why the model stopped |
| `response.stop_sequence` | `str \| None` | Which stop string triggered |
| `response.text` | `str \| None` | Concatenated text (convenience) |
| `response.tool_calls` | `list[ContentBlock]` | TOOL_USE blocks only (convenience) |
| `response.thinking` | `str \| None` | Concatenated thinking (convenience) |

---

## Streaming

```python
async for chunk in client.stream(
    model="anthropic/claude-sonnet-4-6",
    messages=[...],
    # same parameters as chat()
):
    # chunk is ContentBlock or Usage
```

### Stream Protocol

The adapter translates Anthropic's SSE events into a flat stream:

```
┌─────────────────────────────────────────────────────┐
│ Anthropic SSE Events        ModelGate Yields         │
├─────────────────────────────────────────────────────┤
│ message_start            → (captures input_tokens)   │
│ content_block_start      → (starts buffering)        │
│ content_block_delta      → ContentBlock(TEXT)    ──►  │
│   (text_delta)              streamed immediately      │
│ content_block_delta      → (buffers JSON fragments)   │
│   (input_json_delta)                                  │
│ content_block_delta      → (buffers thinking text)    │
│   (thinking_delta)                                    │
│ content_block_delta      → (buffers signature)        │
│   (signature_delta)                                   │
│ content_block_stop       → ContentBlock(TOOL_USE) ──► │
│                            ContentBlock(THINKING) ──► │
│                            ContentBlock(REDACTED) ──► │
│                            ContentBlock(SRV_TOOL) ──► │
│ message_delta            → (captures output_tokens)   │
│ message_stop             → (no-op)                    │
│ ping                     → (ignored)                  │
│ error                    → raises StreamingError  ──► │
│ (end of stream)          → Usage               ──►   │
└─────────────────────────────────────────────────────┘
```

**Key behaviors:**
- **Text:** Yielded immediately on each `text_delta` — one `ContentBlock(TEXT)` per fragment
- **Tool use:** JSON fragments buffered, single `ContentBlock(TOOL_USE)` emitted at `content_block_stop`
- **Thinking:** Buffered, single `ContentBlock(THINKING)` emitted at `content_block_stop` with signature
- **Usage:** Always the last item yielded, combining `message_start` + `message_delta` token counts

---

## Content Types

### Text

**Sending:**
```python
Message(role=Role.USER, content="Hello")
# or
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.TEXT, text="Hello")
])
```

**Receiving:**
```python
response.content[0].type == ContentType.TEXT
response.content[0].text  # "Hello! How can I help?"
response.text              # convenience — concatenates all TEXT blocks
```

**API translation:**
```
ModelGate                          Anthropic API
─────────                          ─────────────
ContentBlock(TEXT, text="Hi") ──►  {"type": "text", "text": "Hi"}
```

---

### Tool Use

What the model returns when it wants to call a tool.

**Receiving:**
```python
for block in response.content:
    if block.type == ContentType.TOOL_USE:
        block.tool_call_id  # "toolu_01XFDUDYJgAACTvnkyLpI1"
        block.tool_name     # "get_weather"
        block.tool_input    # {"location": "NYC"} — always a dict
```

**API translation (response → canonical):**
```
Anthropic API                              ModelGate
─────────────                              ─────────
{"type": "tool_use",                 ──►   ContentBlock(
 "id": "toolu_01XF...",                      type=TOOL_USE,
 "name": "get_weather",                      tool_call_id="toolu_01XF...",
 "input": {"location": "NYC"}}               tool_name="get_weather",
                                             tool_input={"location": "NYC"})
```

**Round-tripping in multi-turn** (sending previous tool_use back as assistant history):
```
ModelGate                              Anthropic API
─────────                              ─────────────
ContentBlock(TOOL_USE,           ──►   {"type": "tool_use",
  tool_call_id="toolu_123",              "id": "toolu_123",
  tool_name="get_weather",               "name": "get_weather",
  tool_input={"location":"NYC"})         "input": {"location":"NYC"}}
```

---

### Tool Results

Sending tool execution results back to the model.

**Sending:**
```python
Message(role=Role.TOOL, content=[
    ContentBlock(
        type=ContentType.TOOL_RESULT,
        tool_call_id="toolu_123",
        tool_result_content="72°F and sunny",
    ),
])
```

**API translation (canonical → API):**
```
ModelGate                              Anthropic API
─────────                              ─────────────
Message(role=TOOL, content=[     ──►   {"role": "user",
  ContentBlock(TOOL_RESULT,              "content": [{
    tool_call_id="toolu_123",              "type": "tool_result",
    tool_result_content="72°F")])          "tool_use_id": "toolu_123",
                                           "content": "72°F"}]}
```

> **Note:** Anthropic requires tool results as `role: "user"` messages. The adapter handles this conversion automatically.

---

### Images

Three source types: `base64`, `url`, `file`.

**Sending:**
```python
# Base64
ContentBlock(type=ContentType.IMAGE,
             image_source_type="base64",
             image_media_type="image/png",
             image_data="iVBOR...")

# URL
ContentBlock(type=ContentType.IMAGE,
             image_source_type="url",
             image_data="https://example.com/photo.jpg")

# File (Anthropic Files API)
ContentBlock(type=ContentType.IMAGE,
             image_source_type="file",
             image_data="file-abc123")
```

**API translation:**
```
ModelGate                              Anthropic API
─────────                              ─────────────
ContentBlock(IMAGE,              ──►   {"type": "image",
  image_source_type="base64",            "source": {
  image_media_type="image/png",            "type": "base64",
  image_data="iVBOR...")                   "media_type": "image/png",
                                           "data": "iVBOR..."}}

ContentBlock(IMAGE,              ──►   {"type": "image",
  image_source_type="url",               "source": {
  image_data="https://...")                "type": "url",
                                           "url": "https://..."}}
```

---

### Documents

PDF and text document support. Three source types: `base64`, `url`, `file`.

**Sending:**
```python
ContentBlock(type=ContentType.DOCUMENT,
             document_source_type="base64",
             document_media_type="application/pdf",
             document_data="JVBE...",
             document_filename="report.pdf")  # optional
```

**API translation:**
```
ModelGate                              Anthropic API
─────────                              ─────────────
ContentBlock(DOCUMENT,           ──►   {"type": "document",
  document_source_type="base64",         "source": {
  document_media_type="...",               "type": "base64",
  document_data="JVBE...",                 "media_type": "...",
  document_filename="doc.pdf")             "data": "JVBE...",
                                           "filename": "doc.pdf"}}
```

---

### Extended Thinking

See [Extended Thinking Configuration](#extended-thinking-configuration) for how to enable it. Here's what comes back.

**Receiving:**
```python
response.content[0].type == ContentType.THINKING
response.content[0].thinking             # "Let me analyze step by step..."
response.content[0].thinking_signature   # "WaUjzkypQ2mUEVM36O2TxuC06KN8=="
response.thinking                         # convenience — concatenates all thinking
```

**Round-tripping** (must include thinking blocks when sending history back):
```python
# Previous assistant response had thinking + text
Message(role=Role.ASSISTANT, content=[
    ContentBlock(type=ContentType.THINKING,
                 thinking="Step 1: ...",
                 thinking_signature="sig123=="),
    ContentBlock(type=ContentType.TEXT, text="The answer is 42."),
])
```

**API translation:**
```
ModelGate                              Anthropic API
─────────                              ─────────────
ContentBlock(THINKING,           ──►   {"type": "thinking",
  thinking="Step 1...",                   "thinking": "Step 1...",
  thinking_signature="sig==")             "signature": "sig=="}
```

---

### Redacted Thinking

Opaque encrypted thinking blocks returned when safety filters activate. Must be round-tripped exactly.

**Receiving:**
```python
block.type == ContentType.REDACTED_THINKING
block.redacted_thinking_data  # "EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc="
```

**Round-tripping:**
```python
ContentBlock(type=ContentType.REDACTED_THINKING,
             redacted_thinking_data="EjVGbmxlcmQgb2YgcmVkYWN0ZWQgdGhpbmtpbmc=")
```

**API translation:**
```
ModelGate                              Anthropic API
─────────                              ─────────────
ContentBlock(REDACTED_THINKING,  ──►   {"type": "redacted_thinking",
  redacted_thinking_data="abc==")         "data": "abc=="}
```

---

### Server Tool Use

Returned when Anthropic server-side tools (web_search, web_fetch, code_execution) are invoked.

**Receiving:**
```python
block.type == ContentType.SERVER_TOOL_USE
block.tool_call_id  # "srvtoolu_014hJH..."
block.tool_name     # "web_search"
block.tool_input    # {"query": "latest news"}
```

> **Note:** Server tool results (web_search_tool_result, etc.) are handled internally by Anthropic and are not exposed in the canonical schema. They are silently skipped during response parsing.

---

## System Prompts

Two formats supported:

```python
# Simple string
await client.chat(..., system="You are a helpful assistant.")

# Array with cache control (forwarded verbatim to the API)
await client.chat(..., system=[
    {
        "type": "text",
        "text": "You are a helpful assistant with deep knowledge.",
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }
])
```

> **Note:** Anthropic uses a top-level `system` field, not a system role message. The adapter handles this automatically.

---

## Tool Definitions

```python
from modelgate import Tool, ToolParameter

# Simple tool with ToolParameter
tool = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
        "unit": ToolParameter(type="string", enum=["celsius", "fahrenheit"]),
    },
    required=["location"],
)

# Complex tool with raw JSON schema
tool = Tool(
    name="search",
    description="Search documents",
    raw_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["query"],
    },
)
```

**API translation:**
```
ModelGate                              Anthropic API
─────────                              ─────────────
Tool(name="get_weather",         ──►   {"name": "get_weather",
  description="Get weather",             "description": "Get weather",
  parameters={"location":                "input_schema": {
    ToolParameter(type="string")},          "type": "object",
  required=["location"])                    "properties": {
                                              "location": {"type": "string"}},
                                            "required": ["location"]}}

Tool(name="search",              ──►   {"name": "search",
  raw_schema={...})                      "description": "...",
                                         "input_schema": {...}}  ← raw_schema used directly
```

---

## Tool Choice

Control how the model uses tools:

```python
# Let the model decide (default)
tool_choice=None       # omitted from payload

# Model can use tools or not
tool_choice="auto"     # → {"type": "auto"}

# Model must use at least one tool
tool_choice="any"      # → {"type": "any"}

# Model must not use tools
tool_choice="none"     # → {"type": "none"}

# Force a specific tool
tool_choice={"type": "tool", "name": "get_weather"}
```

---

## Structured Output

Force JSON schema-validated output:

```python
await client.chat(...,
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
    },
)
```

---

## Sampling Parameters

```python
await client.chat(...,
    temperature=0.7,              # 0.0 to 1.0 (omitted when thinking is enabled)
    top_p=0.9,                    # nucleus sampling
    top_k=40,                     # top-k sampling
    stop_sequences=["END", "---"],  # custom stop strings
    max_tokens=4096,              # max output tokens
)
```

> **Note:** `temperature` is automatically omitted when extended thinking is enabled — Anthropic rejects it.

---

## Extended Thinking Configuration

Enable Claude's chain-of-thought reasoning:

```python
# Fixed budget
await client.chat(...,
    thinking_budget=10000,   # int, must be ≥ 1024 and < max_tokens
    max_tokens=16000,        # must be > thinking_budget
)
# Sends: {"thinking": {"type": "enabled", "budget_tokens": 10000}}

# Adaptive (model decides how much to think)
await client.chat(...,
    thinking_budget="adaptive",
)
# Sends: {"thinking": {"type": "adaptive"}}

# With display control (Claude 4+ only)
await client.chat(...,
    thinking_budget=10000,
    thinking_display="summarized",  # or "omitted"
    max_tokens=16000,
)
# Sends: {"thinking": {"type": "enabled", "budget_tokens": 10000, "display": "summarized"}}

# Interleaved thinking (beta)
await client.chat(...,
    thinking_budget=10000,
    interleaved_thinking=True,   # adds anthropic-beta header
    max_tokens=16000,
)
```

**Validation rules:**
- `thinking_budget` must be `int ≥ 1024` or `"adaptive"`
- `thinking_budget` must be `< max_tokens` (when int)
- `temperature` is automatically omitted when thinking is enabled

---

## Prompt Caching

Cache usage is tracked in the `Usage` object:

```python
response.usage.cache_read_input_tokens     # tokens read from cache
response.usage.cache_creation_input_tokens  # tokens written to cache
```

To enable caching, use array system prompts with `cache_control`:

```python
system = [
    {
        "type": "text",
        "text": "Very long system prompt...",
        "cache_control": {"type": "ephemeral"}       # 5 min TTL (default)
    }
]
# or
system = [
    {
        "type": "text",
        "text": "Very long system prompt...",
        "cache_control": {"type": "ephemeral", "ttl": "1h"}  # 1 hour TTL
    }
]
```

---

## Stop Reasons

The adapter normalizes all Anthropic stop reasons:

| Anthropic `stop_reason` | ModelGate `FinishReason` | Meaning |
|---|---|---|
| `end_turn` | `STOP` | Normal completion |
| `stop_sequence` | `STOP` | Custom stop string hit |
| `tool_use` | `TOOL_USE` | Model wants to call a tool |
| `max_tokens` | `LENGTH` | Hit `max_tokens` limit |
| `pause_turn` | `PAUSE_TURN` | Long turn paused (can resume) |
| `refusal` | `REFUSAL` | Safety refusal |

When `stop_reason` is `stop_sequence`, the actual string is available:
```python
response.stop_sequence  # "END" — which stop string triggered
```

---

## Error Handling

All HTTP errors are mapped to typed exceptions:

| HTTP Status | Exception | When |
|---|---|---|
| 401 | `AuthenticationError` | Invalid or missing API key |
| 429 | `RateLimitError` | Rate limit exceeded |
| 400 | `InvalidRequestError` | Malformed request |
| 400 + `API_KEY_INVALID` | `AuthenticationError` | Bad key in 400 response |
| 5xx | `ProviderError` | Server error |
| Stream error event | `StreamingError` | Error mid-stream |

```python
from modelgate import AuthenticationError, RateLimitError, StreamingError

try:
    response = await client.chat(model="anthropic/claude-sonnet-4-6", ...)
except AuthenticationError:
    print("Check your API key")
except RateLimitError:
    print("Rate limited — back off")
except StreamingError as e:
    print(f"Stream error: {e}")
```

---

## Translation Reference

### Full Kwargs Reference

All extra `**kwargs` accepted by `chat()` and `stream()`:

| Kwarg | Type | Anthropic Field | Notes |
|---|---|---|---|
| `thinking_budget` | `int \| "adaptive"` | `thinking.type` + `thinking.budget_tokens` | ≥ 1024, < max_tokens |
| `thinking_display` | `str` | `thinking.display` | `"summarized"` or `"omitted"` |
| `tool_choice` | `str \| dict` | `tool_choice` | `"auto"`, `"any"`, `"none"`, or dict |
| `output_config` | `dict` | `output_config` | Structured JSON output schema |
| `top_p` | `float` | `top_p` | Nucleus sampling |
| `top_k` | `int` | `top_k` | Top-k sampling |
| `stop_sequences` | `list[str]` | `stop_sequences` | Custom stop strings |
| `metadata` | `dict` | `metadata` | Request metadata (e.g. `user_id`) |
| `service_tier` | `str` | `service_tier` | `"auto"`, etc. |
| `interleaved_thinking` | `bool` | Beta header | Adds `interleaved-thinking-2025-05-14` |

### Usage Fields

| Field | Source in Anthropic Response |
|---|---|
| `usage.input_tokens` | `usage.input_tokens` |
| `usage.output_tokens` | `usage.output_tokens` |
| `usage.total_tokens` | Computed: `input + output` |
| `usage.thinking_tokens` | `usage.thinking_input_tokens` |
| `usage.cache_read_input_tokens` | `usage.cache_read_input_tokens` |
| `usage.cache_creation_input_tokens` | `usage.cache_creation_input_tokens` |

### Streaming Usage Assembly

Anthropic splits token counts across two events:

| Event | Tokens Captured |
|---|---|
| `message_start` → `message.usage` | `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` |
| `message_delta` → `usage` | `output_tokens`, `thinking_input_tokens` |

The adapter combines these into a single `Usage` emitted at end of stream.

---

## Test Coverage

39 tests covering every feature:

| Test Class | Tests | What's Covered |
|---|---|---|
| `TestAnthropicChat` | 2 | Simple text, tool use with mixed content |
| `TestAnthropicMessageFormat` | 5 | Tool result → user msg, system prompt, array system, image base64, document |
| `TestAnthropicErrors` | 3 | 401 → AuthError, 429 → RateLimit, 500 → ProviderError |
| `TestAnthropicKwargsPassthrough` | 4 | stop_sequences, top_p/top_k, metadata, service_tier |
| `TestAnthropicToolChoice` | 3 | auto, specific tool, none omitted |
| `TestAnthropicStopReasons` | 3 | pause_turn, refusal, stop_sequence captured |
| `TestAnthropicServerToolUse` | 1 | server_tool_use parsed |
| `TestAnthropicCacheUsage` | 1 | cache tokens parsed |
| `TestAnthropicThinkingResponse` | 2 | Thinking blocks parsed, .thinking property |
| `TestAnthropicThinkingPayload` | 2 | Budget sent + temperature omitted, no budget → temperature sent |
| `TestAnthropicThinkingRoundTrip` | 1 | Thinking blocks serialized in history |
| `TestAnthropicRedactedThinking` | 2 | Parsed, round-tripped |
| `TestAnthropicBudgetValidation` | 2 | Budget exceeds max_tokens, budget > max_tokens |
| `TestAnthropicAdaptiveThinking` | 1 | Adaptive payload correct |
| `TestAnthropicOutputConfig` | 1 | output_config forwarded |
| `TestAnthropicStream` | 6 | Text, tool use, thinking, error event, cache usage, server tool use |

Run tests:
```bash
.venv/bin/python -m pytest tests/test_anthropic_adapter.py -v
```
