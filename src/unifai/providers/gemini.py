"""Gemini adapter — stub implementation (lower priority)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from unifai.types import ContentBlock, Message, Response, Tool, Usage

from .base import BaseProvider


class GeminiAdapter(BaseProvider):
    """Adapter for the Google Gemini API. Currently a stub."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

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
        raise NotImplementedError("GeminiAdapter.chat() is not yet implemented")

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
        raise NotImplementedError("GeminiAdapter.stream() is not yet implemented")
        if False:  # pragma: no cover
            yield  # type: ignore[misc]
