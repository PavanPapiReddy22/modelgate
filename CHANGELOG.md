# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-27

### Changed
- `ContentBlock` and `Response` now exclude null fields from `model_dump()` and `model_dump_json()` by default

## [0.1.1] - 2026-03-27

### Changed
- Renamed `UnifAI` → `ModelGate`, `UnifAIConfig` → `ModelGateConfig`, `UnifAIError` → `ModelGateError`

## [0.1.0] - 2026-03-27

### Added
- Initial release of unifAI
- Unified, type-safe adapter layer for LLM providers
- Support for Anthropic, OpenAI, AWS Bedrock, Groq, Ollama, Google Gemini, and Vertex AI
- `UnifAI` client with `complete()` and `stream()` methods
- Canonical `Message`, `Response`, `ContentBlock`, and `Tool` types via Pydantic v2
- Tool use / function calling support across all providers
- Extended thinking support for Anthropic models
- Streaming support with async generators
- Structured error hierarchy: `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderError`, `StreamingError`
- Provider-specific error wrappers: `BedrockError`, `VertexError`
- Zero external SDK dependencies — pure `httpx` for all HTTP calls (except optional AWS/Vertex auth)
