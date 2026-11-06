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

from tradeflow.research.provider_errors import ModelProviderError, as_provider_error
from tradeflow.settings import (
    get_credential,
    normalize_ollama_url,
    ollama_base_url,
    ollama_num_ctx,
    ollama_timeout,
)

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

    provider: str = "unknown"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Return the model's completion for a system + user prompt.

        A provider failure is re-raised as :class:`ModelProviderError` naming what
        happened (auth, quota, timeout, ...) so the research loop can stop with a
        sentence instead of a vendor traceback. A missing SDK is not a provider failure
        and passes through.
        """
        try:
            return self._complete(system, user, max_tokens)
        except (ModelProviderError, ImportError):
            raise
        except Exception as exc:  # noqa: BLE001 - vendor exception types are their own
            raise as_provider_error(self.provider, exc, detail=self._detail_for(exc)) from exc

    def _detail_for(self, exc: BaseException):
        """Hook for a client that knows more than the classifier (e.g. its endpoint)."""
        return None

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """The provider call itself."""


class AnthropicClient(LLMClient):
    """Claude via the Anthropic SDK.

    The API key is resolved from ``ANTHROPIC_API_KEY`` via the standard settings
    chain (environment / ``.env``).
    """

    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", client=None):
        self.model = model
        self._client = client

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy: only needed when actually calling Claude

            key = get_credential("ANTHROPIC_API_KEY")
            # No SDK retries: a credit error must surface once, not be re-sent (the SDK
            # default retries 429s), and the session's trial budget should not hide
            # repeated requests.
            self._client = (
                anthropic.Anthropic(api_key=key, max_retries=0) if key else anthropic.Anthropic(max_retries=0)
            )
        return self._client

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
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
    chain (environment / ``.env``).
    """

    provider = "openai"

    def __init__(self, model: str = "gpt-4o", client=None):
        self.model = model
        self._client = client

    def _ensure(self):
        if self._client is None:
            import openai  # lazy

            key = get_credential("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key, max_retries=0) if key else openai.OpenAI(max_retries=0)
        return self._client

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
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

    No API key. Endpoint, timeout and context length come from ``OLLAMA_BASE_URL``,
    ``OLLAMA_TIMEOUT`` and ``OLLAMA_NUM_CTX`` (see :mod:`tradeflow.settings`); the
    context length is sent only when set.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ):
        self.model = model
        self.base_url = normalize_ollama_url(base_url) if base_url else ollama_base_url()
        self.timeout = timeout if timeout is not None else ollama_timeout()
        self.num_ctx = num_ctx if num_ctx is not None else ollama_num_ctx()

    def _detail_for(self, exc: BaseException):
        from tradeflow.research.provider_errors import classify_provider_error

        category, _ = classify_provider_error(exc)
        if category == "unreachable":
            return f"nothing answered at {self.base_url}"
        if category == "timeout":
            return f"no response from {self.base_url} within {self.timeout:g}s (OLLAMA_TIMEOUT)"
        if category == "model_not_found":
            return f"{self.base_url} has no model named {self.model!r}"
        return None

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        import json
        import urllib.request

        options = {"num_predict": max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
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
