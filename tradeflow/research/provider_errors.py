"""Naming what went wrong when a model provider refuses or cannot be reached.

One classification, used twice: by the opt-in readiness probe, and by the research
loop, because a key that passes a probe can still run out of credit twenty minutes
into a session. Both surfaces must name the same failure the same way, so the
mapping lives here and imports nothing heavy - the provider SDKs are never imported,
the exceptions are recognised by their shape (``status_code``, class names).

**Messages never carry the exception's own text.** A vendor error string can echo
request parameters, and the request carries the key. The detail is composed here from
the category, the HTTP status and the exception's type name.
"""

from __future__ import annotations

import socket
import urllib.error
from typing import Any, Literal, Optional, Tuple

ProbeCategory = Literal[
    "ok", "auth", "quota", "entitlement", "unreachable", "timeout", "model_not_found", "unknown"
]

#: Fragments that mark a 429/400 as "the account is out of credit" rather than "slow down".
_CREDIT_MARKERS = (
    "insufficient_quota",
    "credit_balance",
    "credit balance",
    "billing",
    "exceeded your current quota",
)


class ModelProviderError(RuntimeError):
    """A provider call failed in a way the user can act on.

    ``category`` is one of :data:`ProbeCategory`; ``detail`` is safe to print.
    """

    def __init__(self, provider: str, category: ProbeCategory, detail: str):
        super().__init__(f"model provider {provider}: {category} - {detail}")
        self.provider = provider
        self.category = category
        self.detail = detail


def _class_names(exc: BaseException) -> set:
    return {cls.__name__ for cls in type(exc).__mro__}


def _status_of(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _searchable_text(exc: BaseException) -> str:
    """Everything that might say "credit" - used only to decide, never to print."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        parts.append(code)
    return " ".join(parts).lower()


def _is_credit_error(exc: BaseException) -> bool:
    text = _searchable_text(exc)
    return any(marker in text for marker in _CREDIT_MARKERS)


def classify_provider_error(exc: BaseException) -> Tuple[ProbeCategory, str]:
    """Map a provider exception to ``(category, safe detail)``.

    Duck-typed so it needs no SDK. A 429 is ``quota`` either way - neither is retried
    here - but the detail says whether it is exhausted credit (billing action) or a
    rate limit (wait), because the remedies are opposite.
    """
    names = _class_names(exc)

    if isinstance(exc, TimeoutError) or isinstance(exc, socket.timeout) or "APITimeoutError" in names:
        return "timeout", "the provider did not answer in time"
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout", "the provider did not answer in time"
        return "unreachable", "the connection failed before any response"

    status = _status_of(exc)
    if status == 401:
        return "auth", "the provider rejected the credentials (HTTP 401)"
    if status == 403:
        return "entitlement", "the credentials are valid but not permitted to use this model (HTTP 403)"
    if status == 404:
        return "model_not_found", "the provider has no such model (HTTP 404)"
    if status == 429:
        if _is_credit_error(exc):
            return "quota", "credit exhausted (HTTP 429) - the account has no usable balance"
        return "quota", "rate limited (HTTP 429) - too many requests, not an empty balance"
    if status == 400 and _is_credit_error(exc):
        return "quota", "credit exhausted (HTTP 400) - the account has no usable balance"

    if "APIConnectionError" in names or isinstance(exc, (ConnectionError, OSError)):
        return "unreachable", "the connection failed before any response"

    suffix = f" (HTTP {status})" if status is not None else ""
    return "unknown", f"unexpected {type(exc).__name__}{suffix}"


def as_provider_error(
    provider: str, exc: BaseException, *, detail: Optional[Any] = None
) -> ModelProviderError:
    """Wrap ``exc`` as a :class:`ModelProviderError`, optionally overriding the detail."""
    category, classified = classify_provider_error(exc)
    return ModelProviderError(provider, category, str(detail) if detail else classified)
