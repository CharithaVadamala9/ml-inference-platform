"""Central, typed configuration loaded from environment / .env.

Every component (serving, eval, CLI) imports `settings` from here so there is a
single source of truth for provider keys, model names, and service URLs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM judge / generation provider ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    judge_provider: Literal["openai", "anthropic"] = "anthropic"
    judge_model: str = "claude-haiku-4-5-20251001"

    # Embedding model used by the retriever and by RAGAS semantic scoring.
    embed_model: str = "all-MiniLM-L6-v2"

    # ---- Eval judge cache ----
    # Disk-cache judge verdicts (keyed by question+answer+model) so re-running the
    # same eval during development does not re-call the paid Claude API.
    judge_cache_enabled: bool = True
    judge_cache_dir: str = ".cache/judge"

    # ---- Serving backend ----
    serving_backend: Literal["ollama", "vllm", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = ""

    # ---- MLflow ----
    mlflow_tracking_uri: str = "http://localhost:5001"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()


settings = get_settings()
