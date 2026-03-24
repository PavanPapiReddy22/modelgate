"""unifai error hierarchy — typed exceptions for all provider errors."""

from __future__ import annotations


class UnifAIError(Exception):
    """Base exception for all unifai errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(UnifAIError):
    """401 — invalid or missing API key."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class RateLimitError(UnifAIError):
    """429 — provider rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)


class InvalidRequestError(UnifAIError):
    """400 — malformed input or invalid parameters."""

    def __init__(self, message: str = "Invalid request") -> None:
        super().__init__(message, status_code=400)


class ProviderError(UnifAIError):
    """5xx — unexpected provider-side failure."""

    def __init__(self, message: str = "Provider error", status_code: int | None = 500) -> None:
        super().__init__(message, status_code=status_code)


class BedrockError(ProviderError):
    """AWS Bedrock-specific provider error."""

    def __init__(self, message: str = "Bedrock error", status_code: int | None = 500) -> None:
        super().__init__(message, status_code=status_code)


class VertexError(ProviderError):
    """Google Vertex AI-specific provider error."""

    def __init__(self, message: str = "Vertex AI error", status_code: int | None = 500) -> None:
        super().__init__(message, status_code=status_code)


class StreamingError(UnifAIError):
    """Error occurred mid-stream."""

    def __init__(self, message: str = "Streaming error") -> None:
        super().__init__(message)


# ── Utility ──────────────────────────────────────────────────────────────────


def map_http_status(status_code: int, message: str) -> UnifAIError:
    """Map an HTTP status code to the appropriate UnifAIError subclass."""
    if status_code == 401:
        return AuthenticationError(message)
    if status_code == 429:
        return RateLimitError(message)
    if status_code == 400:
        if "API_KEY_INVALID" in message:
            return AuthenticationError(message)
        return InvalidRequestError(message)
    if status_code >= 500:
        return ProviderError(message, status_code=status_code)
    return UnifAIError(message, status_code=status_code)
