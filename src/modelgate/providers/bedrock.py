"""Bedrock adapter — AWS Converse API with SigV4 request signing."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from modelgate.errors import BedrockError, StreamingError, map_http_status
from modelgate.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Response,
    Role,
    Tool,
    Usage,
)

from .base import BaseProvider

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP,
    "content_filtered": FinishReason.STOP,
}


class BedrockAdapter(BaseProvider):
    """Adapter for AWS Bedrock Converse API with SigV4 signing via botocore."""

    def __init__(
        self,
        region: str | None = None,
        boto3_session: Any | None = None,
    ) -> None:
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Lazy-import boto3/botocore to keep it optional at import time
        import boto3
        import botocore.auth
        import botocore.awsrequest
        import botocore.credentials

        if boto3_session:
            self._session = boto3_session
        else:
            self._session = boto3.Session(region_name=self._region)

        credentials = self._session.get_credentials()
        if credentials is None:
            raise BedrockError("No AWS credentials found")
        self._credentials = credentials.get_frozen_credentials()
        self._signer = botocore.auth.SigV4Auth(
            self._credentials, "bedrock", self._region
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def _endpoint(self) -> str:
        return f"https://bedrock-runtime.{self._region}.amazonaws.com"

    def _sign_request(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> dict[str, str]:
        """Sign an HTTP request using SigV4."""
        import botocore.awsrequest

        aws_request = botocore.awsrequest.AWSRequest(
            method=method,
            url=url,
            headers=headers,
            data=body,
        )
        self._signer.add_auth(aws_request)
        return dict(aws_request.headers)

    def _build_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert canonical Messages to Bedrock Converse format."""
        out: list[dict[str, object]] = []
        for msg in messages:
            out.append(self._convert_message(msg))
        return out

    def _convert_message(self, msg: Message) -> dict[str, object]:
        """Convert a single canonical Message to Bedrock Converse format."""
        if isinstance(msg.content, str):
            return {
                "role": msg.role.value,
                "content": [{"text": msg.content}],
            }

        # TOOL role → user message with toolResult blocks
        if msg.role == Role.TOOL:
            results = []
            for block in msg.content:
                results.append(
                    {
                        "toolResult": {
                            "toolUseId": block.tool_call_id or "",
                            "content": [{"text": block.tool_result_content or ""}],
                        }
                    }
                )
            return {"role": "user", "content": results}

        # Regular content blocks
        content_blocks: list[dict[str, object]] = []
        for block in msg.content:
            if block.type == ContentType.TEXT:
                content_blocks.append({"text": block.text or ""})
            elif block.type == ContentType.TOOL_USE:
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": block.tool_call_id or "",
                            "name": block.tool_name or "",
                            "input": block.tool_input or {},
                        }
                    }
                )

        return {"role": msg.role.value, "content": content_blocks}

    def _build_tools(self, tools: list[Tool]) -> dict[str, object]:
        """Convert canonical Tool list to Bedrock toolConfig format."""
        tool_specs = []
        for tool in tools:
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    name: {
                                        k: v
                                        for k, v in param.model_dump().items()
                                        if v is not None
                                    }
                                    for name, param in tool.parameters.items()
                                },
                                "required": tool.required,
                            }
                        },
                    }
                }
            )
        return {"tools": tool_specs}

    def _parse_response(
        self, data: dict[str, object], model: str, *, request_id: str = ""
    ) -> Response:
        """Parse a Bedrock Converse response into canonical Response."""
        output = data.get("output", {})
        message = output.get("message", {})  # type: ignore[union-attr]

        content_blocks: list[ContentBlock] = []
        for block in message.get("content", []):  # type: ignore[union-attr]
            if "text" in block:
                content_blocks.append(
                    ContentBlock(type=ContentType.TEXT, text=block["text"])
                )
            elif "toolUse" in block:
                tu = block["toolUse"]
                content_blocks.append(
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id=tu.get("toolUseId"),
                        tool_name=tu.get("name"),
                        tool_input=tu.get("input", {}),
                    )
                )

        stop_reason = str(data.get("stopReason", "end_turn"))
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, FinishReason.STOP)

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("inputTokens", 0)  # type: ignore[union-attr]
        output_tokens = usage_data.get("outputTokens", 0)  # type: ignore[union-attr]

        return Response(
            id=request_id,
            model=model,
            content=content_blocks,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=finish_reason,
        )

    # ── Public API ───────────────────────────────────────────────────────

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
        url = f"{self._endpoint}/model/{model}/converse"

        payload: dict[str, object] = {
            "messages": self._build_messages(messages),
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = [{"text": system}]
        if tools:
            payload["toolConfig"] = self._build_tools(tools)

        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        signed_headers = self._sign_request("POST", url, headers, body)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    url,
                    headers=signed_headers,
                    content=body,
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise BedrockError(exc.response.text, exc.response.status_code) from exc

        # Extract request ID from HTTP headers (not in the JSON body)
        request_id = resp.headers.get("x-amzn-requestid", "")
        return self._parse_response(resp.json(), model, request_id=request_id)

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
        """Streaming via boto3 converse_stream (handles binary event stream).

        Uses an asyncio.Queue to bridge the synchronous boto3 event stream
        iterator into async land without blocking the event loop.
        """
        import asyncio

        converse_kwargs: dict[str, object] = {
            "modelId": model,
            "messages": self._build_messages(messages),
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            converse_kwargs["system"] = [{"text": system}]
        if tools:
            converse_kwargs["toolConfig"] = self._build_tools(tools)

        try:
            client = self._session.client(
                "bedrock-runtime", region_name=self._region
            )
            # Run synchronous boto3 call in a thread
            resp = await asyncio.to_thread(
                client.converse_stream, **converse_kwargs
            )

            event_stream = resp.get("stream", [])

            # Bridge synchronous iterator → async via a queue + background thread
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _consume() -> None:
                for event in event_stream:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            loop.run_in_executor(None, _consume)

            tool_use_id = ""
            tool_use_name = ""
            tool_input_parts: list[str] = []

            while True:
                event = await queue.get()
                if event is None:
                    break

                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})  # type: ignore[union-attr]
                    if "toolUse" in start:
                        tool_use_id = start["toolUse"].get("toolUseId", "")
                        tool_use_name = start["toolUse"].get("name", "")
                        tool_input_parts = []

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})  # type: ignore[union-attr]
                    if "text" in delta:
                        yield ContentBlock(
                            type=ContentType.TEXT,
                            text=delta["text"],
                        )
                    elif "toolUse" in delta:
                        tool_input_parts.append(
                            delta["toolUse"].get("input", "")
                        )

                elif "contentBlockStop" in event:
                    if tool_use_id:
                        input_str = "".join(tool_input_parts)
                        try:
                            parsed_input = (
                                json.loads(input_str) if input_str else {}
                            )
                        except json.JSONDecodeError:
                            parsed_input = {}
                        yield ContentBlock(
                            type=ContentType.TOOL_USE,
                            tool_call_id=tool_use_id,
                            tool_name=tool_use_name,
                            tool_input=parsed_input,
                        )
                        tool_use_id = ""

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})  # type: ignore[union-attr]
                    input_t = usage.get("inputTokens", 0)
                    output_t = usage.get("outputTokens", 0)
                    yield Usage(
                        input_tokens=input_t,
                        output_tokens=output_t,
                        total_tokens=input_t + output_t,
                    )

        except Exception as exc:
            if isinstance(exc, (BedrockError, StreamingError)):
                raise
            raise StreamingError(str(exc)) from exc
