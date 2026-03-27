"""Vertex AI adapter — extends GeminiAdapter with Vertex endpoint and OAuth2 auth."""

from __future__ import annotations

import os
from typing import Any

from .gemini import GeminiAdapter


class VertexAdapter(GeminiAdapter):
    """Adapter for Google Vertex AI. Same API format as Gemini, different auth + endpoint."""

    def __init__(
        self,
        credentials: Any | None = None,
        project: str | None = None,
        region: str | None = None,
    ) -> None:
        # Don't call super().__init__ — we don't use an API key
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._region = region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

        # Lazy-import google.auth to keep it optional
        if credentials:
            self._credentials = credentials
        else:
            try:
                import google.auth  # type: ignore[import-untyped]
                import google.auth.transport.requests  # type: ignore[import-untyped]

                creds, found_project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._credentials = creds
                if not self._project and found_project:
                    self._project = found_project
            except ImportError as exc:
                raise ImportError(
                    "google-auth is required for VertexAdapter. "
                    "Install it with: pip install google-auth"
                ) from exc
            except Exception as exc:
                from modelgate.errors import VertexError

                raise VertexError(f"Failed to get default credentials: {exc}") from exc

    # ── URL overrides ────────────────────────────────────────────────────

    @property
    def _endpoint(self) -> str:
        return (
            f"https://{self._region}-aiplatform.googleapis.com/v1"
            f"/projects/{self._project}/locations/{self._region}"
            f"/publishers/google/models"
        )

    def _chat_url(self, model: str) -> str:
        return f"{self._endpoint}/{model}:generateContent"

    def _stream_url(self, model: str) -> str:
        return f"{self._endpoint}/{model}:streamGenerateContent?alt=sse"

    def _headers(self) -> dict[str, str]:
        """Return headers with OAuth2 bearer token."""
        self._refresh_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._credentials.token}",
        }

    def _refresh_token(self) -> None:
        """Refresh the OAuth2 token if expired."""
        if not self._credentials.valid:
            try:
                import google.auth.transport.requests  # type: ignore[import-untyped]

                self._credentials.refresh(google.auth.transport.requests.Request())
            except Exception as exc:
                from modelgate.errors import VertexError

                raise VertexError(f"Failed to refresh credentials: {exc}") from exc
