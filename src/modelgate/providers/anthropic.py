"""Anthropic adapter — content-block normalization and streaming tool-call buffering."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

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

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP,
    "pause_turn": FinishReason.PAUSE_TURN,
    "refusal": FinishReason.REFUSAL,
}

# Optional kwargs that are forwarded verbatim into the API payload.
_PASSTHROUGH_KWARGS = frozenset({
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
    "service_tier",
})


class AnthropicAdapter(BaseProvider):
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _headers(self, beta: list[str] | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if beta:
            headers["anthropic-beta"] = ",".join(beta)
        return headers

    def _build_thinking_config(
        self,
        thinking_budget: object,
        max_tokens: int,
        display: str | None,
    ) -> dict[str, object] | None:
        """Return the thinking payload dict, or None if thinking is disabled."""
        if thinking_budget is None:
            return None
        if thinking_budget == "adaptive":
            cfg: dict[str, object] = {"type": "adaptive"}
        else:
            if not isinstance(thinking_budget, int):
                raise ValueError(f"thinking_budget must be an int or 'adaptive', got {thinking_budget!r}")
            if thinking_budget < 1024:
                raise ValueError(f"thinking_budget ({thinking_budget}) must be at least 1024")
            if thinking_budget >= max_tokens:
                raise ValueError(
                    f"thinking_budget ({thinking_budget}) must be less than max_tokens ({max_tokens})"
                )
            cfg = {"type": "enabled", "budget_tokens": thinking_budget}
        if display is not None:
            # "display" controls thinking summarization ("summarized" | "omitted").
            # Only honoured on Claude 4+ models — Sonnet 3.7 always returns full
            # (non-summarized) thinking regardless of this field.
            cfg["display"] = display
        return cfg

    def _build_tool_choice(self, tool_choice: object) -> dict[str, object] | None:
        """Convert tool_choice kwarg to Anthropic format.

        Accepts:
            - str: "auto", "any", "none"
            - dict: {"type": "tool", "name": "my_tool"} — force a specific tool
            - None: omit from payload (API default)
        """
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return {"type": tool_choice}
        if isinstance(tool_choice, dict):
            return tool_choice  # type: ignore[return-value]
        raise ValueError(f"tool_choice must be a str or dict, got {type(tool_choice)}")

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

        # ASSISTANT / USER with mixed content blocks
        content_blocks: list[dict[str, Any]] = []
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
            elif block.type == ContentType.THINKING:
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking or "",
                        "signature": block.thinking_signature or "",
                    }
                )
            elif block.type == ContentType.REDACTED_THINKING:
                content_blocks.append(
                    {
                        "type": "redacted_thinking",
                        "data": block.redacted_thinking_data or "",
                    }
                )
            elif block.type == ContentType.IMAGE:
                img_block: dict[str, Any] = {"type": "image"}
                source_type = block.image_source_type or "base64"
                if source_type == "base64":
                    img_block["source"] = {
                        "type": "base64",
                        "media_type": block.image_media_type or "image/png",
                        "data": block.image_data or "",
                    }
                elif source_type == "url":
                    img_block["source"] = {
                        "type": "url",
                        "url": block.image_data or "",
                    }
                elif source_type == "file":
                    img_block["source"] = {
                        "type": "file",
                        "file_id": block.image_data or "",
                    }
                else:
                    img_block["source"] = {
                        "type": "base64",
                        "media_type": block.image_media_type or "image/png",
                        "data": block.image_data or "",
                    }
                content_blocks.append(img_block)
            elif block.type == ContentType.DOCUMENT:
                doc_block: dict[str, Any] = {"type": "document"}
                source_type = block.document_source_type or "base64"
                if source_type == "base64":
                    src: dict[str, Any] = {
                        "type": "base64",
                        "media_type": block.document_media_type or "application/pdf",
                        "data": block.document_data or "",
                    }
                elif source_type == "url":
                    src = {
                        "type": "url",
                        "url": block.document_data or "",
                    }
                else:  # file
                    src = {
                        "type": "file",
                        "file_id": block.document_data or "",
                    }
                if block.document_filename:
                    src["filename"] = block.document_filename
                doc_block["source"] = src
                content_blocks.append(doc_block)

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
            elif block_type == "thinking":
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.THINKING,
                        thinking=block.get("thinking", ""),  # type: ignore[union-attr]
                        thinking_signature=block.get("signature", ""),  # type: ignore[union-attr]
                    )
                )
            elif block_type == "redacted_thinking":
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.REDACTED_THINKING,
                        redacted_thinking_data=block.get("data", ""),  # type: ignore[union-attr]
                    )
                )
            elif block_type == "server_tool_use":
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.SERVER_TOOL_USE,
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
        thinking_tokens = usage_data.get("thinking_input_tokens", 0)  # type: ignore[union-attr]
        cache_read = usage_data.get("cache_read_input_tokens", 0)  # type: ignore[union-attr]
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)  # type: ignore[union-attr]

        return Response(
            id=str(data.get("id", "")),
            model=model,
            content=content_blocks,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                thinking_tokens=thinking_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
            ),
            finish_reason=finish_reason,
            stop_sequence=data.get("stop_sequence"),  # type: ignore[arg-type]
        )

    def _build_payload(
        self,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None,
        system: str | list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        """Build the full API payload from chat/stream parameters."""
        thinking_budget = kwargs.get("thinking_budget")
        thinking_display = kwargs.get("thinking_display")
        tool_choice = kwargs.get("tool_choice")
        output_config = kwargs.get("output_config")

        thinking_cfg = self._build_thinking_config(thinking_budget, max_tokens, thinking_display)  # type: ignore[arg-type]

        payload: dict[str, object] = {
            "model": model,
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
        }

        if stream:
            payload["stream"] = True

        # Anthropic rejects temperature when thinking is enabled
        if thinking_cfg is None:
            payload["temperature"] = temperature
        else:
            payload["thinking"] = thinking_cfg

        # System prompt — string or array-of-blocks (for cache_control)
        if system is not None:
            payload["system"] = system

        if tools:
            payload["tools"] = self._build_tools(tools)

        # Tool choice
        tc = self._build_tool_choice(tool_choice)
        if tc is not None:
            payload["tool_choice"] = tc

        # Structured JSON output
        if output_config is not None:
            payload["output_config"] = output_config

        # Pass-through optional kwargs
        for key in _PASSTHROUGH_KWARGS:
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        return payload

    def _build_beta_headers(self, **kwargs: object) -> list[str]:
        """Collect beta feature headers from kwargs."""
        beta: list[str] = []
        if bool(kwargs.get("interleaved_thinking", False)):
            beta.append("interleaved-thinking-2025-05-14")
        return beta

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
        payload = self._build_payload(
            messages, model, tools, system, max_tokens, temperature,
            stream=False, **kwargs,
        )
        beta = self._build_beta_headers(**kwargs)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{_ANTHROPIC_API_URL}/messages",
                    headers=self._headers(beta or None),
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
        payload = self._build_payload(
            messages, model, tools, system, max_tokens, temperature,
            stream=True, **kwargs,
        )
        beta = self._build_beta_headers(**kwargs)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{_ANTHROPIC_API_URL}/messages",
                    headers=self._headers(beta or None),
                    json=payload,
                    timeout=120.0,
                ) as resp:
                    resp.raise_for_status()

                    # Buffers for streaming
                    current_block_type: str | None = None
                    tool_id: str = ""
                    tool_name: str = ""
                    tool_input_json: str = ""
                    thinking_buffer: str = ""
                    thinking_signature: str = ""
                    redacted_thinking_data: str = ""
                    stream_input_tokens: int = 0
                    stream_output_tokens: int = 0
                    stream_thinking_tokens: int = 0
                    stream_cache_read: int = 0
                    stream_cache_creation: int = 0

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        # Handle error events from the stream
                        if event_type == "error":
                            error_data = event.get("error", {})
                            error_msg = error_data.get("message", "Unknown streaming error")
                            raise StreamingError(f"Anthropic stream error: {error_msg}")

                        if event_type == "content_block_start":
                            block = event.get("content_block", {})
                            current_block_type = block.get("type")
                            if current_block_type == "tool_use":
                                tool_id = block.get("id", "")
                                tool_name = block.get("name", "")
                                tool_input_json = ""
                            elif current_block_type == "thinking":
                                thinking_buffer = ""
                                thinking_signature = ""
                            elif current_block_type == "redacted_thinking":
                                redacted_thinking_data = block.get("data", "")  # reset + capture in one step
                            elif current_block_type == "server_tool_use":
                                tool_id = block.get("id", "")
                                tool_name = block.get("name", "")
                                tool_input_json = ""

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                yield ContentBlock(type=ContentType.TEXT, text=text)

                            elif delta_type == "input_json_delta":
                                tool_input_json += delta.get("partial_json", "")

                            elif delta_type == "thinking_delta":
                                thinking_buffer += delta.get("thinking", "")

                            elif delta_type == "signature_delta":
                                thinking_signature += delta.get("signature", "")

                        elif event_type == "content_block_stop":
                            if current_block_type == "thinking" and (thinking_buffer or thinking_signature):
                                yield ContentBlock(
                                    type=ContentType.THINKING,
                                    thinking=thinking_buffer,
                                    thinking_signature=thinking_signature,
                                )
                            elif current_block_type == "redacted_thinking" and redacted_thinking_data:
                                yield ContentBlock(
                                    type=ContentType.REDACTED_THINKING,
                                    redacted_thinking_data=redacted_thinking_data,
                                )
                            elif current_block_type == "tool_use":
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
                            elif current_block_type == "server_tool_use":
                                try:
                                    parsed_input = json.loads(tool_input_json) if tool_input_json else {}
                                except json.JSONDecodeError:
                                    parsed_input = {}
                                yield ContentBlock(
                                    type=ContentType.SERVER_TOOL_USE,
                                    tool_call_id=tool_id,
                                    tool_name=tool_name,
                                    tool_input=parsed_input,
                                )
                            current_block_type = None

                        elif event_type == "message_delta":
                            # Anthropic sends output_tokens in message_delta
                            delta_usage = event.get("usage", {})
                            if delta_usage:
                                stream_output_tokens = delta_usage.get("output_tokens", 0)
                                stream_thinking_tokens = delta_usage.get("thinking_input_tokens", 0)

                        elif event_type == "message_start":
                            msg = event.get("message", {})
                            msg_usage = msg.get("usage", {})
                            stream_input_tokens = msg_usage.get("input_tokens", 0)
                            stream_cache_read = msg_usage.get("cache_read_input_tokens", 0)
                            stream_cache_creation = msg_usage.get("cache_creation_input_tokens", 0)

                        elif event_type == "message_stop":
                            pass

                    # Emit final usage — Anthropic splits across
                    # message_start (input_tokens) and message_delta (output_tokens)
                    yield Usage(
                        input_tokens=stream_input_tokens,
                        output_tokens=stream_output_tokens,
                        total_tokens=stream_input_tokens + stream_output_tokens,
                        thinking_tokens=stream_thinking_tokens,
                        cache_read_input_tokens=stream_cache_read,
                        cache_creation_input_tokens=stream_cache_creation,
                    )

            except httpx.HTTPStatusError as exc:
                await exc.response.aread()
                raise map_http_status(exc.response.status_code, exc.response.text) from exc
            except Exception as exc:
                if isinstance(exc, StreamingError):
                    raise
                raise StreamingError(str(exc)) from exc
