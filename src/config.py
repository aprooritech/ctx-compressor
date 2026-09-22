from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "openrouter"]
CompressionStrategy = Literal["none", "trim", "summarize", "hybrid"]
TokenizerBackend = Literal["tiktoken", "huggingface", "approx"]
MessageRole = Literal["system", "user", "assistant"]

PROVIDER_NAMES: tuple[str, ...] = ("ollama", "openrouter")

DEFAULT_SUMMARY_PROMPT = (
    "You are a context compression engine inside an LLM proxy. Condense the "
    "conversation excerpt below into dense factual notes that let another model "
    "continue the conversation without having seen the original text.\n"
    "Rules:\n"
    "- Preserve: decisions, constraints, requirements, names, identifiers, "
    "numbers, file paths, code symbols, open questions and unresolved tasks.\n"
    "- Drop: greetings, filler, apologies, repetition, meta talk.\n"
    "- Keep the chronological order and attribute statements to user/assistant.\n"
    "- Output compact bullet points only. No preamble, no closing remarks.\n"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    app_name: str = "Context-Window Compressor Proxy"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    proxy_api_keys_raw: str = Field(default="", alias="PROXY_API_KEYS")
    cors_allow_origins_raw: str = Field(default="*", alias="CORS_ALLOW_ORIGINS")

    default_provider: ProviderName = "ollama"
    enable_provider_fallback: bool = False
    fallback_provider: ProviderName = "openrouter"

    ollama_host: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1"
    ollama_keep_alive: str = "5m"

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: SecretStr | None = None
    openrouter_default_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_site_url: str = "http://localhost:8000"
    openrouter_app_name: str = "context-compressor-proxy"

    request_timeout_seconds: float = Field(default=120.0, gt=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    max_connections: int = Field(default=100, ge=1)
    max_keepalive_connections: int = Field(default=20, ge=1)

    compression_enabled: bool = True
    max_context_tokens: int = Field(default=8192, ge=256)
    reserve_output_tokens: int = Field(default=1024, ge=0)
    compression_target_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    keep_last_messages: int = Field(default=6, ge=0)
    compression_strategy: CompressionStrategy = "hybrid"

    summary_provider: ProviderName | None = None
    summary_model: str | None = None
    summary_max_tokens: int = Field(default=512, ge=32)
    summary_chunk_tokens: int = Field(default=1500, ge=128)
    summary_max_depth: int = Field(default=3, ge=1, le=10)
    summary_message_role: MessageRole = "system"
    summary_timeout_seconds: float = Field(default=60.0, gt=0)
    summary_retry_attempts: int = Field(default=2, ge=1, le=5)
    summary_retry_backoff_seconds: float = Field(default=1.0, ge=0.0)
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT

    trim_message_max_tokens: int = Field(default=200, ge=16)

    tokenizer_backend: TokenizerBackend = "tiktoken"
    tiktoken_encoding: str = "cl100k_base"
    hf_tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_init_timeout_seconds: float = Field(default=10.0, gt=0)

    db_enabled: bool = True
    db_path: Path = Path("data/proxy.db")
    db_busy_timeout_ms: int = Field(default=5000, ge=0)
    persist_message_preview: bool = False

    @property
    def proxy_api_keys(self) -> tuple[str, ...]:
        return _split_csv(self.proxy_api_keys_raw)

    @property
    def cors_allow_origins(self) -> list[str]:
        origins = _split_csv(self.cors_allow_origins_raw)
        return list(origins) if origins else ["*"]

    @property
    def compression_trigger_tokens(self) -> int:
        return max(256, self.max_context_tokens - self.reserve_output_tokens)

    @property
    def compression_target_tokens(self) -> int:
        return max(128, int(self.compression_trigger_tokens * self.compression_target_ratio))

    def default_model_for(self, provider: ProviderName) -> str:
        return self.ollama_default_model if provider == "ollama" else self.openrouter_default_model

    def has_openrouter_credentials(self) -> bool:
        key = self.openrouter_api_key
        return key is not None and bool(key.get_secret_value().strip())

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @field_validator("ollama_host", "openrouter_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("summary_provider", "summary_model", "openrouter_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.reserve_output_tokens >= self.max_context_tokens:
            raise ValueError("reserve_output_tokens must be smaller than max_context_tokens")
        if self.enable_provider_fallback and self.fallback_provider == self.default_provider:
            raise ValueError(
                "fallback_provider must differ from default_provider when fallback is enabled"
            )
        return self


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
