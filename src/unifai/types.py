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


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_USE = "tool_use"
    LENGTH = "length"
    ERROR = "error"


# ── Content & Messages ──────────────────────────────────────────────────────


class ContentBlock(BaseModel):
    """A single block of content — text, tool call, or tool result."""

    type: ContentType
    # TEXT
    text: str | None = None
    # TOOL_USE
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    # TOOL_RESULT
    tool_result_content: str | None = None


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

    @property
    def text(self) -> str | None:
        """Return concatenated text from all TEXT blocks, or None if no text."""
        texts = [b.text for b in self.content if b.type == ContentType.TEXT and b.text]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ContentBlock]:
        """Return only TOOL_USE blocks."""
        return [b for b in self.content if b.type == ContentType.TOOL_USE]
