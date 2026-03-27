"""unifai canonical types — Pydantic v2 models for the unified LLM schema."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
    IMAGE = "image"
    DOCUMENT = "document"
    SERVER_TOOL_USE = "server_tool_use"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_USE = "tool_use"
    LENGTH = "length"
    ERROR = "error"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


# ── Content & Messages ──────────────────────────────────────────────────────


class ContentBlock(BaseModel):
    """A single block of content — text, tool call, tool result, thinking, image, or document."""

    type: ContentType
    # TEXT
    text: str | None = None
    # TOOL_USE / SERVER_TOOL_USE
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    # TOOL_RESULT
    tool_result_content: str | None = None
    # THINKING (Anthropic extended thinking)
    thinking: str | None = None
    thinking_signature: str | None = None
    # REDACTED_THINKING (Anthropic safety — opaque encrypted data, must round-trip)
    redacted_thinking_data: str | None = None
    # Gemini 3 thought signature — opaque pass-through, other providers ignore
    thought_signature: str | None = None
    # IMAGE — base64 or URL source
    image_source_type: str | None = None  # "base64" | "url" | "file"
    image_media_type: str | None = None  # e.g. "image/png", "image/jpeg"
    image_data: str | None = None  # base64-encoded data or URL string or file_id
    # DOCUMENT — PDF / text documents
    document_source_type: str | None = None  # "base64" | "url" | "file"
    document_media_type: str | None = None  # e.g. "application/pdf"
    document_data: str | None = None  # base64-encoded data or URL string or file_id
    document_filename: str | None = None


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | list[ContentBlock]

    @model_validator(mode="after")
    def _validate_tool_role(self) -> Message:
        """Messages with role=TOOL must have all content blocks of type TOOL_RESULT."""
        if self.role == Role.TOOL and isinstance(self.content, list):
            for block in self.content:
                if block.type != ContentType.TOOL_RESULT:
                    raise ValueError(
                        f"Message with role=TOOL must only contain TOOL_RESULT blocks, "
                        f"got {block.type.value}"
                    )
        return self


# ── Tools ────────────────────────────────────────────────────────────────────


class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class Tool(BaseModel):
    """A tool definition that can be passed to any provider."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = []

    @model_validator(mode="after")
    def _validate_required_subset(self) -> Tool:
        """required list must be a subset of parameters keys."""
        param_keys = set(self.parameters.keys())
        for key in self.required:
            if key not in param_keys:
                raise ValueError(
                    f"Required key '{key}' is not in parameters: {sorted(param_keys)}"
                )
        return self


# ── Usage & Response ─────────────────────────────────────────────────────────


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    thinking_tokens: int = 0  # informational only — already included in output_tokens
    cache_read_input_tokens: int = 0  # tokens read from prompt cache
    cache_creation_input_tokens: int = 0  # tokens written to prompt cache

    @property
    def type(self) -> str:
        """Sentinel so stream consumers can safely check chunk.type without isinstance."""
        return "usage"

    @model_validator(mode="after")
    def _validate_total(self) -> Usage:
        """total_tokens must equal input_tokens + output_tokens."""
        expected = self.input_tokens + self.output_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) != "
                f"input_tokens ({self.input_tokens}) + output_tokens ({self.output_tokens})"
            )
        return self


class Response(BaseModel):
    """Unified response from any LLM provider."""

    id: str
    model: str
    content: list[ContentBlock]
    usage: Usage
    finish_reason: FinishReason
    stop_sequence: str | None = None  # the actual stop string that triggered stop_reason

    @property
    def text(self) -> str | None:
        """Return concatenated text from all TEXT blocks, or None if no text."""
        texts = [b.text for b in self.content if b.type == ContentType.TEXT and b.text]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ContentBlock]:
        """Return only TOOL_USE blocks."""
        return [b for b in self.content if b.type == ContentType.TOOL_USE]

    @property
    def thinking(self) -> str | None:
        """Return concatenated thinking from all THINKING blocks, or None."""
        parts = [b.thinking for b in self.content if b.type == ContentType.THINKING and b.thinking]
        return "".join(parts) if parts else None
