"""Backend registry + factory.

`get_backend("ollama")` returns a ready-to-use engine. The serving gateway and
the RAG generator use the configured default; the eval judge asks for a
specific provider explicitly.
"""

from __future__ import annotations

from mlip.config import settings
from mlip.serving.backends.anthropic import AnthropicBackend
from mlip.serving.backends.base import ChatBackend, Message
from mlip.serving.backends.ollama import OllamaBackend

__all__ = ["ChatBackend", "Message", "OllamaBackend", "AnthropicBackend", "get_backend"]

_BACKENDS: dict[str, type[ChatBackend]] = {
    "ollama": OllamaBackend,
    "anthropic": AnthropicBackend,
}


def get_backend(name: str | None = None, *, model: str | None = None) -> ChatBackend:
    """Instantiate a backend by name (defaults to the configured serving backend)."""
    name = name or settings.serving_backend
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {', '.join(sorted(_BACKENDS))}"
        ) from None
    return cls(model=model) if model else cls()
