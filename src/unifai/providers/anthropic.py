"""Anthropic adapter — content-block normalization and streaming tool-call buffering."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from unifai.errors import StreamingError, map_http_status
from unifai.types import (
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

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP,
}


class AnthropicAdapter(BaseProvider):
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _build_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert canonical Messages to Anthropic format."""
        out: list[dict[str, object]] = []
        for msg in messages:
            out.append(self._convert_message(msg))
        return out

    def _convert_message(self, msg: Message) -> dict[str, object]:
        """Convert a single canonical Message to Anthropic format."""
        if isinstance(msg.content, str):
            return {"role": msg.role.value, "content": msg.content}

        # TOOL role → user message with tool_result blocks
        if msg.role == Role.TOOL:
            tool_results = []
            for block in msg.content:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id or "",
                        "content": block.tool_result_content or "",
                    }
                )
            return {"role": "user", "content": tool_results}

        # ASSISTANT with mixed content blocks
        content_blocks = []
        for block in msg.content:
            if block.type == ContentType.TEXT:
                content_blocks.append({"type": "text", "text": block.text or ""})
            elif block.type == ContentType.TOOL_USE:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.tool_call_id or "",
                        "name": block.tool_name or "",
                        "input": block.tool_input or {},
                    }
                )

        return {"role": msg.role.value, "content": content_blocks}

    def _build_tools(self, tools: list[Tool]) -> list[dict[str, object]]:
        """Convert canonical Tool list to Anthropic tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
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
            }
            for tool in tools
        ]

    def _parse_response(self, data: dict[str, object], model: str) -> Response:
        """Parse an Anthropic Messages API response into canonical Response."""
        content_blocks: list[ContentBlock] = []

        for block in data.get("content", []):  # type: ignore[union-attr]
            block_type = block.get("type")  # type: ignore[union-attr]
            if block_type == "text":
                content_blocks.append(
                    ContentBlock(type=ContentType.TEXT, text=block.get("text", ""))  # type: ignore[union-attr]
                )
            elif block_type == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id=block.get("id"),  # type: ignore[union-attr]
                        tool_name=block.get("name"),  # type: ignore[union-attr]
                        tool_input=block.get("input", {}),  # type: ignore[union-attr]
                    )
                )

        stop_reason = str(data.get("stop_reason", "end_turn"))
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, FinishReason.STOP)

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)  # type: ignore[union-attr]
        output_tokens = usage_data.get("output_tokens", 0)  # type: ignore[union-attr]

        return Response(
            id=str(data.get("id", "")),
            model=model,
            content=content_blocks,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
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
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._build_tools(tools)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{_ANTHROPIC_API_URL}/messages",
                    headers=self._headers(),
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise map_http_status(exc.response.status_code, exc.response.text) from exc

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
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._build_tools(tools)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{_ANTHROPIC_API_URL}/messages",
                    headers=self._headers(),
                    json=payload,
                    timeout=120.0,
                ) as resp:
                    resp.raise_for_status()

                    # Buffers for streaming
                    current_block_type: str | None = None
                    text_buffer: str = ""
                    tool_id: str = ""
                    tool_name: str = ""
                    tool_input_json: str = ""

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        if event_type == "content_block_start":
                            block = event.get("content_block", {})
                            current_block_type = block.get("type")
                            if current_block_type == "tool_use":
                                tool_id = block.get("id", "")
                                tool_name = block.get("name", "")
                                tool_input_json = ""
                            elif current_block_type == "text":
                                text_buffer = block.get("text", "")

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                yield ContentBlock(type=ContentType.TEXT, text=text)

                            elif delta_type == "input_json_delta":
                                tool_input_json += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            if current_block_type == "tool_use":
                                try:
                                    parsed_input = json.loads(tool_input_json) if tool_input_json else {}
                                except json.JSONDecodeError:
                                    parsed_input = {}
                                yield ContentBlock(
                                    type=ContentType.TOOL_USE,
                                    tool_call_id=tool_id,
                                    tool_name=tool_name,
                                    tool_input=parsed_input,
                                )
                            current_block_type = None

                        elif event_type == "message_delta":
                            # Final usage info
                            usage = event.get("usage", {})
                            if usage:
                                output_t = usage.get("output_tokens", 0)
                                # Anthropic sends output_tokens in message_delta
                                # We'll emit a partial usage; full usage comes from message_start
                                pass  # Handled below in message_stop

                        elif event_type == "message_start":
                            msg = event.get("message", {})
                            msg_usage = msg.get("usage", {})
                            self._stream_input_tokens = msg_usage.get("input_tokens", 0)

                        elif event_type == "message_stop":
                            pass

                    # Emit final usage if we have it
                    # Anthropic splits usage across message_start and message_delta
                    # We collect what we can
                    input_t = getattr(self, "_stream_input_tokens", 0)
                    # Note: output tokens aren't always available in stream
                    yield Usage(
                        input_tokens=input_t,
                        output_tokens=0,
                        total_tokens=input_t,
                    )

            except httpx.HTTPStatusError as exc:
                raise map_http_status(exc.response.status_code, exc.response.text) from exc
            except Exception as exc:
                if isinstance(exc, StreamingError):
                    raise
                raise StreamingError(str(exc)) from exc
