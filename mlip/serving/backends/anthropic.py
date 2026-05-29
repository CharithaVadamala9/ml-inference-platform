"""Anthropic backend — used as the LLM-as-a-judge (and optionally as generator).

The judge should be at least as capable as the model it grades, so we point it
at a frontier API model rather than the small local one under test.
"""

from __future__ import annotations

from functools import cached_property

from mlip.config import settings
from mlip.serving.backends.base import ChatBackend, Message


class AnthropicBackend(ChatBackend):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model or settings.judge_model)
        self._api_key = api_key or settings.anthropic_api_key

    @cached_property
    def _client(self):  # lazy import so the SDK isn't required unless used
        import anthropic

        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env to use the Anthropic backend."
            )
        return anthropic.Anthropic(api_key=self._api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        # Anthropic takes the system prompt as a separate argument.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or None,
            messages=convo,
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()
