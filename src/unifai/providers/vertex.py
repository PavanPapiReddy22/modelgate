"""Vertex AI adapter — stub implementation (lower priority)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from unifai.types import ContentBlock, Message, Response, Tool, Usage

from .base import BaseProvider


class VertexAdapter(BaseProvider):
    """Adapter for Google Vertex AI. Currently a stub."""

    def __init__(self, credentials: Any | None = None) -> None:
        self._credentials = credentials

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
        raise NotImplementedError("VertexAdapter.chat() is not yet implemented")

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
        raise NotImplementedError("VertexAdapter.stream() is not yet implemented")
        if False:  # pragma: no cover
            yield  # type: ignore[misc]
