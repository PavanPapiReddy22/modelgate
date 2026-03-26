"""Gemini adapter — Google Gemini REST API with tool-call normalization and SSE streaming."""

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

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.STOP,
    "RECITATION": FinishReason.STOP,
    "LANGUAGE": FinishReason.STOP,
    "OTHER": FinishReason.ERROR,
    "BLOCKLIST": FinishReason.STOP,
    "PROHIBITED_CONTENT": FinishReason.STOP,
    "SPII": FinishReason.STOP,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "UNEXPECTED_TOOL_CALL": FinishReason.ERROR,
    "TOO_MANY_TOOL_CALLS": FinishReason.ERROR,
    "MISSING_THOUGHT_SIGNATURE": FinishReason.ERROR,
    "MALFORMED_RESPONSE": FinishReason.ERROR,
    "IMAGE_SAFETY": FinishReason.STOP,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.STOP,
    "IMAGE_OTHER": FinishReason.ERROR,
    "NO_IMAGE": FinishReason.ERROR,
    "IMAGE_RECITATION": FinishReason.STOP,
    "FINISH_REASON_UNSPECIFIED": FinishReason.STOP,
}


class GeminiAdapter(BaseProvider):
    """Adapter for the Google Gemini REST API (generativelanguage.googleapis.com)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    # ── URL helpers (overridden by VertexAdapter) ────────────────────────

    def _chat_url(self, model: str) -> str:
        return f"{_GEMINI_API_URL}/models/{model}:generateContent?key={self._api_key}"

    def _stream_url(self, model: str) -> str:
        return f"{_GEMINI_API_URL}/models/{model}:streamGenerateContent?key={self._api_key}&alt=sse"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    # ── Message conversion ───────────────────────────────────────────────

    def _build_contents(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert canonical Messages to Gemini contents array."""
        contents: list[dict[str, object]] = []
        for msg in messages:
            contents.append(self._convert_message(msg))
        return contents

    def _convert_message(self, msg: Message) -> dict[str, object]:
        """Convert a single canonical Message to Gemini Content format."""
        role = "model" if msg.role == Role.ASSISTANT else "user"

        if isinstance(msg.content, str):
            return {"role": role, "parts": [{"text": msg.content}]}

        parts: list[dict[str, object]] = []

        # TOOL role → functionResponse parts
        if msg.role == Role.TOOL:
            for block in msg.content:
                fr: dict[str, object] = {
                    "name": block.tool_name or block.tool_call_id or "",
                    "response": {"result": block.tool_result_content or ""},
                }
                # Include the matching id so Gemini can map result → call
                if block.tool_call_id:
                    fr["id"] = block.tool_call_id
                parts.append({"functionResponse": fr})
            return {"role": "user", "parts": parts}

        # Regular content blocks
        for block in msg.content:
            if block.type == ContentType.TEXT:
                parts.append({"text": block.text or ""})
            elif block.type == ContentType.TOOL_USE:
                fc_part: dict[str, object] = {
                    "functionCall": {
                        "name": block.tool_name or "",
                        "args": block.tool_input or {},
                    }
                }
                # Gemini 3: pass thought_signature back on the same part
                if block.thought_signature:
                    fc_part["thoughtSignature"] = block.thought_signature
                parts.append(fc_part)

        return {"role": role, "parts": parts}

    def _build_tools(self, tools: list[Tool]) -> list[dict[str, object]]:
        """Convert canonical Tool list to Gemini functionDeclarations format."""
        declarations = []
        for tool in tools:
            declarations.append({
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
            })
        return [{"functionDeclarations": declarations}]

    # ── Response parsing ─────────────────────────────────────────────────

    def _parse_response(self, data: dict[str, object], model: str) -> Response:
        """Parse a Gemini generateContent response into canonical Response."""
        candidates = data.get("candidates", [])
        candidate = candidates[0] if candidates else {}  # type: ignore[index]

        content_blocks = self._parse_parts(candidate)

        finish_reason_str = str(candidate.get("finishReason", "STOP"))  # type: ignore[union-attr]
        finish_reason = _FINISH_REASON_MAP.get(finish_reason_str, FinishReason.STOP)

        # Check if any content block is a tool call → override finish reason
        if any(b.type == ContentType.TOOL_USE for b in content_blocks):
            finish_reason = FinishReason.TOOL_USE

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)  # type: ignore[union-attr]
        candidates_tokens = usage_meta.get("candidatesTokenCount", 0)  # type: ignore[union-attr]

        return Response(
            id=str(data.get("responseId", "")),
            model=model,
            content=content_blocks,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=candidates_tokens,
                total_tokens=prompt_tokens + candidates_tokens,
            ),
            finish_reason=finish_reason,
        )

    def _parse_parts(self, candidate: dict[str, object]) -> list[ContentBlock]:
        """Extract ContentBlocks from a Gemini candidate's parts."""
        content_blocks: list[ContentBlock] = []
        content = candidate.get("content", {})  # type: ignore[union-attr]
        parts = content.get("parts", []) if isinstance(content, dict) else []  # type: ignore[union-attr]

        for part in parts:
            if "text" in part:
                content_blocks.append(
                    ContentBlock(type=ContentType.TEXT, text=part["text"])
                )
            elif "functionCall" in part:
                fc = part["functionCall"]
                # Gemini 3+ returns a unique id; older models may not
                call_id = fc.get("id") or fc.get("name", "")
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id=call_id,
                        tool_name=fc.get("name", ""),
                        tool_input=fc.get("args", {}),
                        # Gemini 3 thought signature — opaque pass-through
                        thought_signature=part.get("thoughtSignature"),
                    )
                )

        return content_blocks

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
            "contents": self._build_contents(messages),
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._build_tools(tools)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    self._chat_url(model),
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
            "contents": self._build_contents(messages),
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._build_tools(tools)

        # Buffers for accumulating function calls across stream chunks
        tool_call_buffers: list[dict[str, object]] = []
        usage_prompt = 0
        usage_candidates = 0

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    self._stream_url(model),
                    headers=self._headers(),
                    json=payload,
                    timeout=120.0,
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract usage metadata
                        usage_meta = data.get("usageMetadata", {})
                        if usage_meta:
                            usage_prompt = usage_meta.get("promptTokenCount", usage_prompt)
                            usage_candidates = usage_meta.get("candidatesTokenCount", usage_candidates)

                        # Process candidates
                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue

                        candidate = candidates[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", []) if isinstance(content, dict) else []

                        for part in parts:
                            if "text" in part:
                                yield ContentBlock(
                                    type=ContentType.TEXT,
                                    text=part["text"],
                                )
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                call_id = fc.get("id") or fc.get("name", "")
                                tool_call_buffers.append({
                                    "id": call_id,
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args", {}),
                                })

                    # Emit buffered tool calls at end of stream
                    for buf in tool_call_buffers:
                        yield ContentBlock(
                            type=ContentType.TOOL_USE,
                            tool_call_id=str(buf["id"]),
                            tool_name=str(buf["name"]),
                            tool_input=buf["args"],  # type: ignore[arg-type]
                        )

                    # Emit final usage
                    yield Usage(
                        input_tokens=usage_prompt,
                        output_tokens=usage_candidates,
                        total_tokens=usage_prompt + usage_candidates,
                    )

            except httpx.HTTPStatusError as exc:
                raise map_http_status(exc.response.status_code, exc.response.text) from exc
            except Exception as exc:
                if isinstance(exc, StreamingError):
                    raise
                raise StreamingError(str(exc)) from exc
