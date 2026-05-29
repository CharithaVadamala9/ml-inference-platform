"""Backend abstraction shared by the RAG generator, the eval judge, and the
serving gateway.

A backend is anything that can turn a list of chat messages into a completion.
Keeping one interface means the *engine* (Ollama locally, vLLM on GPU, an API
provider) is a swappable detail — the rest of the platform never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

# A chat message is the standard {"role": ..., "content": ...} shape used by
# every major LLM API, so backends can pass it through with minimal translation.
Message = dict[str, str]


class ChatBackend(ABC):
    """Minimal synchronous chat interface every engine implements."""

    #: Short engine identifier, e.g. "ollama", "anthropic", "vllm".
    name: str

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Return the assistant's reply text for the given messages."""

    def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Yield reply text in chunks. Default: one chunk from the non-streamed reply.

        Backends that support token streaming (e.g. Ollama, vLLM) override this so
        the serving layer can measure time-to-first-token.
        """
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(model={self.model!r})"
