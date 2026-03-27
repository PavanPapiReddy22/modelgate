"""modelgate — model-agnostic adapter layer for LLMs."""

__version__ = "0.1.0"

from modelgate.client import UnifAI, UnifAIConfig
from modelgate.errors import (
    AuthenticationError,
    BedrockError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    StreamingError,
    UnifAIError,
    VertexError,
)
from modelgate.types import (
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
    "__version__",
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
