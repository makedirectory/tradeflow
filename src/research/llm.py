"""Provider-agnostic LLM clients for the research proposer.

The research loop only needs one thing from a model: turn a prompt into text (and
report how many tokens it cost, for the budget). That contract is captured by
:class:`LLMClient`, with interchangeable backends so a user can run the agent
against a hosted frontier model or a local one:

* :class:`AnthropicClient` - Claude via the ``anthropic`` SDK (``ai`` extra).
* :class:`OpenAIClient` - GPT models via the ``openai`` SDK (``openai`` extra).
* :class:`OllamaClient` - any local model served by Ollama, over its HTTP API
  (no extra dependency - uses the standard library).

Each SDK is imported lazily inside its client, so the base install needs none of
them and an unused provider never has to be installed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.settings import get_credential

#: Default model per provider (override with ``--model`` / the ``model`` arg).
DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",
    "ollama": "llama3.1",
}


@dataclass
class LLMResponse:
    """A completion plus its token cost (for the research token budget)."""

    text: str
    tokens: int = 0


class LLMClient(ABC):
    """Minimal text-completion contract shared by every provider."""

    model: str

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Return the model's completion for a system + user prompt."""


class AnthropicClient(LLMClient):
    """Claude via the Anthropic SDK.

    The API key is resolved from ``ANTHROPIC_API_KEY`` via the standard settings
    chain (environment / ``.env`` / legacy ``config.py``).
    """

    def __init__(self, model: str = "claude-opus-4-8", client=None):
        self.model = model
        self._client = client

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy: only needed when actually calling Claude

            key = get_credential("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        response = self._ensure().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        tokens = (usage.input_tokens + usage.output_tokens) if usage else 0
        return LLMResponse(text=text, tokens=tokens)


class OpenAIClient(LLMClient):
    """GPT models via the OpenAI SDK.

    The API key is resolved from ``OPENAI_API_KEY`` via the standard settings
    chain (environment / ``.env`` / legacy ``config.py``).
    """

    def __init__(self, model: str = "gpt-4o", client=None):
        self.model = model
        self._client = client

    def _ensure(self):
        if self._client is None:
            import openai  # lazy

            key = get_credential("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        response = self._ensure().chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens = usage.total_tokens if usage else 0
        return LLMResponse(text=text, tokens=tokens)


class OllamaClient(LLMClient):
    """Any local model served by Ollama, via its HTTP API (no SDK dependency).

    No API key. The server URL comes from ``OLLAMA_BASE_URL`` (environment /
    ``.env`` / legacy ``config.py``), defaulting to ``http://localhost:11434``.
    """

    def __init__(self, model: str = "llama3.1", base_url: Optional[str] = None, timeout: float = 120.0):
        self.model = model
        self.base_url = (base_url or get_credential("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - local URL
            data = json.loads(response.read().decode("utf-8"))
        text = data.get("message", {}).get("content", "")
        tokens = int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
        return LLMResponse(text=text, tokens=tokens)


def build_llm_client(provider: str = "anthropic", model: str = None, **kwargs) -> LLMClient:
    """Construct an :class:`LLMClient` for ``provider`` (anthropic | openai | ollama)."""
    key = provider.lower()
    chosen = model or DEFAULT_MODELS.get(key)
    if key == "anthropic":
        return AnthropicClient(chosen, **kwargs)
    if key == "openai":
        return OpenAIClient(chosen, **kwargs)
    if key == "ollama":
        return OllamaClient(chosen, **kwargs)
    raise ValueError(f"Unknown LLM provider {provider!r}. Choose from {sorted(DEFAULT_MODELS)}.")
