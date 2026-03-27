"""Base provider ABC — contract all adapters must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from modelgate.types import ContentBlock, Message, Response, Tool, Usage


class BaseProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    @abstractmethod
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
        """Single-turn, non-streaming chat completion."""
        ...

    @abstractmethod
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
        """Streaming chat. Yields ContentBlock chunks, then a final Usage."""
        ...
        # Make this an async generator for type-checking
        if False:  # pragma: no cover  # noqa: SIM108
            yield  # type: ignore[misc]
