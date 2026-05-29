"""Scaffold smoke tests: the package imports and config has sane defaults."""

from mlip import __version__
from mlip.config import Settings


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)  # ignore any local .env for a deterministic test
    assert s.serving_backend in {"ollama", "vllm", "openai"}
    assert s.judge_provider in {"openai", "anthropic"}
    assert s.mlflow_tracking_uri.startswith("http")
