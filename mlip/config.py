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
    judge_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    judge_model: str = "llama3.2:1b"

    # Embedding model used by the retriever and by RAGAS semantic scoring.
    embed_model: str = "all-MiniLM-L6-v2"

    # ---- Eval judge cache ----
    # Disk-cache judge verdicts (keyed by question+answer+model) so re-running the
    # same eval during development does not re-call the paid Claude API.
    judge_cache_enabled: bool = True
    judge_cache_dir: str = ".cache/judge"

    # ---- Statistical gate ----
    # Paired bootstrap CI + significance testing for the quality gate.
    gate_alpha: float = 0.05  # 1 - alpha = 95% CI
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 42  # seed so gate verdicts are reproducible
    correction_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg"
    judge_pass_threshold: int = 4  # judge_raw >= this counts as a "pass" (McNemar)
    gate_judge: bool = False  # judge metric is informational by default
    min_paired_questions: int = 5  # warn/fail if champion<->candidate overlap is smaller
    # Faithfulness is ill-defined for abstentions, so it is excluded for these
    # categories (the bucket is gated on answer_correctness only).
    faithfulness_skip_categories: list[str] = ["unanswerable"]

    # ---- Judge-calibration audit ----
    # Agreement (Cohen's kappa) between the LLM judge and a human-labeled gold set.
    # If kappa drops below the threshold the judge has drifted and the gate fails.
    calibration_path: str = "data/calibration.jsonl"
    # Measured on the gold set: healthy Claude judge ~0.92, swapped 1B judge ~-0.08.
    # 0.6 leaves a comfortable margin below healthy and well above a drifted judge.
    kappa_threshold: float = 0.6
    gate_calibration: bool = True  # run the calibration audit as part of the gate

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
