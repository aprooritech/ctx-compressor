# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-21

### Added

- OpenAI-compatible `POST /v1/chat/completions` endpoint with streaming (SSE).
- Dual-provider architecture: Ollama (native `/api/chat`, NDJSON → OpenAI SSE)
  and OpenRouter (OpenAI passthrough with bearer auth and attribution headers).
- Provider routing via `X-Provider` header, model prefix, or configured default,
  with conflict detection and optional failover (`ENABLE_PROVIDER_FALLBACK`).
- Context compression engine with `none`/`trim`/`summarize`/`hybrid` strategies,
  recursive map-reduce summarization and token-exact truncation.
- Pluggable token counting: `tiktoken`, HuggingFace tokenizers, or a built-in
  conservative approximation, with automatic degradation between them.
- Metric headers on every response (`X-Original-Tokens`, `X-Compressed-Tokens`,
  `X-Saved-Ratio`, …) exposed to browser clients via CORS.
- Persistence and observability with `aiosqlite` in WAL mode, plus
  `GET /v1/stats` and `GET /v1/stats/requests` aggregations.
- `GET /v1/models` aggregating both backend catalogues and `GET /healthz`.
- Optional proxy-level authentication via `PROXY_API_KEYS`.
- 61 tests (unit, HTTP-mocked providers, ASGI end-to-end) plus
  `scripts/live_check.py` for verification against a real backend.

### Fixed

Both found while verifying against the live OpenRouter API:

- **Startup could hang indefinitely.** `tiktoken` downloads its vocabulary on
  first use; with an unreachable CDN that download blocked application startup
  forever, because the fallback only handled exceptions, not stalls. Startup now
  enforces a socket-level deadline plus an `asyncio.wait_for` guard and degrades
  to the approximation.
- **A single failed chunk discarded the whole summary.** One rate-limited
  summarization call caused the engine to drop all long-term memory and fall
  back to trimming. Chunks are now retried (`SUMMARY_RETRY_ATTEMPTS`) and partial
  results are kept; trimming only takes over when every chunk fails.
