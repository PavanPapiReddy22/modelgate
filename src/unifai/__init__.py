"""unifai — model-agnostic adapter layer for LLMs."""

from unifai.client import UnifAI, UnifAIConfig
from unifai.errors import (
    AuthenticationError,
    BedrockError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    StreamingError,
    UnifAIError,
    VertexError,
)
from unifai.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Response,
    Role,
    Tool,
    ToolParameter,
    Usage,
)

__all__ = [
    # Client
    "UnifAI",
    "UnifAIConfig",
    # Types
    "ContentBlock",
    "ContentType",
    "FinishReason",
    "Message",
    "Response",
    "Role",
    "Tool",
    "ToolParameter",
    "Usage",
    # Errors
    "UnifAIError",
    "AuthenticationError",
    "BedrockError",
    "InvalidRequestError",
    "ProviderError",
    "RateLimitError",
    "StreamingError",
    "VertexError",
]
