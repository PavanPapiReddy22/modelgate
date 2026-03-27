"""modelgate — model-agnostic adapter layer for LLMs."""

__version__ = "0.1.0"

from modelgate.client import ModelGate, ModelGateConfig
from modelgate.errors import (
    AuthenticationError,
    BedrockError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    StreamingError,
    ModelGateError,
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
    "ModelGate",
    "ModelGateConfig",
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
    "ModelGateError",
    "AuthenticationError",
    "BedrockError",
    "InvalidRequestError",
    "ProviderError",
    "RateLimitError",
    "StreamingError",
    "VertexError",
]
