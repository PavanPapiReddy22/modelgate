"""UnifAI client — model string routing and provider management."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from unifai.types import ContentBlock, Message, Response, Tool, Usage

from collections.abc import AsyncIterator


class UnifAIConfig(BaseModel):
    """Configuration for UnifAI client."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    aws_region: str = "us-east-1"
    boto3_session: Any | None = None
    vertex_credentials: Any | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434/v1"

    model_config = {"arbitrary_types_allowed": True}


# Provider name → (adapter_class_import_path, requires_key_name)
_PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    "gemini": "gemini",
    "vertex": "vertex",
    "groq": "generic_openai",
    "ollama": "generic_openai",
}


class UnifAI:
    """Model-agnostic LLM client with provider routing."""

    def __init__(self, config: UnifAIConfig | None = None) -> None:
        self._config = config or UnifAIConfig()
        self._providers: dict[str, Any] = {}

    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """Parse 'provider/model-id' into (provider_name, model_id)."""
        if "/" not in model:
            raise ValueError(
                f"Model string must be in 'provider/model-id' format, got: {model}"
            )
        provider, model_id = model.split("/", 1)
        return provider.lower(), model_id

    def _get_provider(self, provider_name: str) -> Any:
        """Get or create a provider adapter instance."""
        if provider_name in self._providers:
            return self._providers[provider_name]

        adapter = self._create_provider(provider_name)
        self._providers[provider_name] = adapter
        return adapter

    def _create_provider(self, provider_name: str) -> Any:
        """Create a new provider adapter instance."""
        from unifai.providers.anthropic import AnthropicAdapter
        from unifai.providers.bedrock import BedrockAdapter
        from unifai.providers.gemini import GeminiAdapter
        from unifai.providers.generic_openai import GenericOpenAIAdapter
        from unifai.providers.openai import OpenAIAdapter
        from unifai.providers.vertex import VertexAdapter

        cfg = self._config

        if provider_name == "openai":
            key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            return OpenAIAdapter(api_key=key)

        if provider_name == "anthropic":
            key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            return AnthropicAdapter(api_key=key)

        if provider_name == "bedrock":
            return BedrockAdapter(
                region=cfg.aws_region,
                boto3_session=cfg.boto3_session,
            )

        if provider_name == "gemini":
            key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            return GeminiAdapter(api_key=key)

        if provider_name == "vertex":
            return VertexAdapter(credentials=cfg.vertex_credentials)

        if provider_name == "groq":
            key = cfg.groq_api_key or os.environ.get("GROQ_API_KEY", "")
            return GenericOpenAIAdapter(base_url=cfg.groq_base_url, api_key=key)

        if provider_name == "ollama":
            return GenericOpenAIAdapter(base_url=cfg.ollama_base_url)

        raise ValueError(f"Unknown provider: {provider_name}")

    def _coerce_messages(
        self, messages: list[dict[str, Any] | Message]
    ) -> list[Message]:
        """Accept raw dicts or Message objects, return list[Message]."""
        result: list[Message] = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append(msg)
            elif isinstance(msg, dict):
                result.append(Message.model_validate(msg))
            else:
                raise TypeError(f"Expected dict or Message, got {type(msg)}")
        return result

    # ── Public API ───────────────────────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> Response:
        """Non-streaming chat completion routed to the appropriate provider."""
        provider_name, model_id = self._parse_model_string(model)
        provider = self._get_provider(provider_name)
        coerced_messages = self._coerce_messages(messages)

        return await provider.chat(
            messages=coerced_messages,
            model=model_id,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> AsyncIterator[ContentBlock | Usage]:
        """Streaming chat completion routed to the appropriate provider."""
        provider_name, model_id = self._parse_model_string(model)
        provider = self._get_provider(provider_name)
        coerced_messages = self._coerce_messages(messages)

        async for chunk in provider.stream(
            messages=coerced_messages,
            model=model_id,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk
