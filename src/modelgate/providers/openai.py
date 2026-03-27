"""OpenAI adapter — thin wrapper over GenericOpenAIAdapter for api.openai.com."""

from __future__ import annotations

import os

from .generic_openai import GenericOpenAIAdapter


class OpenAIAdapter(GenericOpenAIAdapter):
    """Adapter for the official OpenAI API."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(base_url="https://api.openai.com/v1", api_key=key)
