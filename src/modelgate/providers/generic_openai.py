"""Generic OpenAI-compatible adapter — works with any OpenAI Chat Completions API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from modelgate.errors import StreamingError, map_http_status
from modelgate.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Response,
    Role,
    Tool,
    Usage,
)

from .base import BaseProvider

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_USE,
    "function_call": FinishReason.TOOL_USE,  # deprecated but still emitted
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.STOP,
    None: FinishReason.STOP,
}


class GenericOpenAIAdapter(BaseProvider):
    """Adapter for any OpenAI Chat Completions-compatible API."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ── Helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, object]]:
        """Convert canonical Messages to OpenAI message dicts.

        OpenAI requires one message per tool result, so TOOL messages
        with multiple blocks are flattened into separate messages.
        """
        out: list[dict[str, object]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == Role.TOOL and isinstance(msg.content, list):
                # Flatten: one message per tool result
                for block in msg.content:
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.tool_call_id or "",
                        "content": block.tool_result_content or "",
                    })
            else:
                out.append(self._convert_message(msg))
        return out

    def _convert_message(self, msg: Message) -> dict[str, object]:
        """Convert a single canonical Message to the OpenAI format.

        Note: Role.TOOL messages are handled by _build_messages (which
        flattens multiple tool results into separate messages) and never
        reach this method.
        """
        if isinstance(msg.content, str):
            return {"role": msg.role.value, "content": msg.content}

        # ASSISTANT with tool calls
        tool_use_blocks = [b for b in msg.content if b.type == ContentType.TOOL_USE]
        text_blocks = [b for b in msg.content if b.type == ContentType.TEXT]

        if tool_use_blocks:
            result: dict[str, object] = {"role": msg.role.value}
            if text_blocks:
                result["content"] = "".join(b.text or "" for b in text_blocks)
            else:
                result["content"] = None
            result["tool_calls"] = [
                {
                    "id": b.tool_call_id or "",
                    "type": "function",
                    "function": {
                        "name": b.tool_name or "",
                        "arguments": json.dumps(b.tool_input or {}),
                    },
                }
                for b in tool_use_blocks
            ]
            return result

        # Plain content blocks
        text = "".join(b.text or "" for b in msg.content if b.type == ContentType.TEXT)
        return {"role": msg.role.value, "content": text}

    def _build_tools(self, tools: list[Tool]) -> list[dict[str, object]]:
        """Convert canonical Tool list to OpenAI function tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {
                                k: v
                                for k, v in param.model_dump().items()
                                if v is not None
                            }
                            for name, param in tool.parameters.items()
                        },
                        "required": tool.required,
                    },
                },
            }
            for tool in tools
        ]

    def _parse_response(self, data: dict[str, object], model: str) -> Response:
        """Parse an OpenAI Chat Completions response into a canonical Response."""
        choice = data["choices"][0]  # type: ignore[index]
        message = choice["message"]  # type: ignore[index]

        content_blocks: list[ContentBlock] = []

        # Text content
        text = message.get("content")  # type: ignore[union-attr]
        if text:
            content_blocks.append(ContentBlock(type=ContentType.TEXT, text=text))

        # Tool calls
        tool_calls = message.get("tool_calls", [])  # type: ignore[union-attr]
        for tc in tool_calls:  # type: ignore[union-attr]
            func = tc["function"]  # type: ignore[index]
            arguments = func.get("arguments", "{}")  # type: ignore[union-attr]
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_args = {}
            else:
                parsed_args = arguments

            content_blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call_id=tc.get("id"),  # type: ignore[union-attr]
                    tool_name=func.get("name"),  # type: ignore[union-attr]
                    tool_input=parsed_args,
                )
            )

        finish_reason_str = choice.get("finish_reason")  # type: ignore[union-attr]
        finish_reason = _FINISH_REASON_MAP.get(finish_reason_str, FinishReason.STOP)

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)  # type: ignore[union-attr]
        output_tokens = usage_data.get("completion_tokens", 0)  # type: ignore[union-attr]
        completion_details = usage_data.get("completion_tokens_details") or {}  # type: ignore[union-attr]
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)  # type: ignore[union-attr]

        return Response(
            id=str(data.get("id", "")),
            model=model,
            content=content_blocks,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                thinking_tokens=reasoning_tokens,
            ),
            finish_reason=finish_reason,
        )

    # ── Public API ───────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> Response:
        payload: dict[str, object] = {
            "model": model,
            "messages": self._build_messages(messages, system),
            "max_completion_tokens": max_tokens,
        }
        # o-series reasoning models (o3, o4-mini, etc.) reject temperature — omit it.
        # Valid reasoning_effort values: none, minimal, low, medium, high, xhigh
        # (none/xhigh added with gpt-5.1; ignored by non-OpenAI compatible APIs)
        # "none" means reasoning disabled — temperature is allowed in that case.
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if not reasoning_effort or reasoning_effort == "none":
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = self._build_tools(tools)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text
                raise map_http_status(exc.response.status_code, body) from exc

        return self._parse_response(resp.json(), model)

    async def stream(
        self,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> AsyncIterator[ContentBlock | Usage]:
        payload: dict[str, object] = {
            "model": model,
            "messages": self._build_messages(messages, system),
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Same as chat: temperature suppressed for reasoning models, allowed for "none".
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if not reasoning_effort or reasoning_effort == "none":
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = self._build_tools(tools)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=120.0,
                ) as resp:
                    resp.raise_for_status()

                    # Buffer for accumulating tool call arguments
                    tool_call_buffers: dict[int, dict[str, object]] = {}

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Usage in final chunk
                        if "usage" in data and data["usage"]:
                            u = data["usage"]
                            input_t = u.get("prompt_tokens", 0)
                            output_t = u.get("completion_tokens", 0)
                            details = u.get("completion_tokens_details") or {}
                            reasoning_t = details.get("reasoning_tokens", 0)
                            yield Usage(
                                input_tokens=input_t,
                                output_tokens=output_t,
                                total_tokens=input_t + output_t,
                                thinking_tokens=reasoning_t,
                            )
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Text delta
                        if delta.get("content"):
                            yield ContentBlock(
                                type=ContentType.TEXT,
                                text=delta["content"],
                            )

                        # Tool call deltas
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_buffers:
                                tool_call_buffers[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            buf = tool_call_buffers[idx]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                buf["name"] = func["name"]
                            if func.get("arguments"):
                                buf["arguments"] = str(buf["arguments"]) + func["arguments"]

                        # Check finish reason — emit buffered tool calls
                        finish = choices[0].get("finish_reason")
                        if finish == "tool_calls":
                            for buf in tool_call_buffers.values():
                                args_str = str(buf.get("arguments", "{}"))
                                try:
                                    parsed = json.loads(args_str)
                                except json.JSONDecodeError:
                                    parsed = {}
                                yield ContentBlock(
                                    type=ContentType.TOOL_USE,
                                    tool_call_id=str(buf["id"]),
                                    tool_name=str(buf["name"]),
                                    tool_input=parsed,
                                )
                            tool_call_buffers.clear()

            except httpx.HTTPStatusError as exc:
                await exc.response.aread()
                raise map_http_status(exc.response.status_code, exc.response.text) from exc
            except Exception as exc:
                if isinstance(exc, StreamingError):
                    raise
                raise StreamingError(str(exc)) from exc
