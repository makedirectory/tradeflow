"""Naming what went wrong when a model provider refuses or cannot be reached.

One classification, used twice: by the opt-in readiness probe, and by the research
loop, because a key that passes a probe can still run out of credit twenty minutes
into a session. Both surfaces must name the same failure the same way, so the
mapping lives here and imports nothing heavy - the provider SDKs are never imported,
the exceptions are recognised by their shape (``status_code``, ``code``, ``type``,
class names).

**Structured fields decide; the message is the last resort.** Both SDKs carry a
machine-readable error code/type (``insufficient_quota`` against
``rate_limit_exceeded``; ``billing_error``, ``rate_limit_error``,
``overloaded_error``). A message is free text that can mention billing for any reason -
a rate-limit reply that links the billing page is still a rate limit - so wording is
consulted only when no structured field settles the question.

**Messages never carry the exception's own text.** A vendor error string can echo
request parameters, and the request carries the key. The detail is composed here from
the category, the HTTP status and the exception's type name.
"""

from __future__ import annotations

import http.client
import socket
import urllib.error
from typing import Any, List, Literal, Optional, Tuple

ProbeCategory = Literal[
    "ok",
    "auth",
    "quota",
    "rate_limited",
    "unavailable",
    "entitlement",
    "unreachable",
    "timeout",
    "model_not_found",
    "unknown",
]

#: Categories worth sending again: the condition is the provider's or the network's and
#: passes by itself. ``quota`` (an empty balance), ``auth``, ``entitlement`` and
#: ``model_not_found`` are not - repeating the request cannot change the answer, only
#: bill or hammer an account. ``timeout`` is left out too: a provider that did not answer
#: in the configured time is not helped by waiting that long again, three times.
RETRYABLE: Tuple[str, ...] = ("rate_limited", "unavailable", "unreachable")

#: Structured ``code`` / ``type`` values (OpenAI and Anthropic) and the category each settles.
_STRUCTURED = {
    "insufficient_quota": "quota",
    "billing_error": "quota",
    "rate_limit_exceeded": "rate_limited",
    "rate_limit_error": "rate_limited",
    "overloaded_error": "unavailable",
    "server_error": "unavailable",
    "api_error": "unavailable",
    "authentication_error": "auth",
    "invalid_api_key": "auth",
    "permission_error": "entitlement",
    "not_found_error": "model_not_found",
    "model_not_found": "model_not_found",
}

#: Wording that marks a 400/429 as "out of credit" - consulted only when no structured
#: field settled it. Anthropic reports an empty balance as a plain 400 with no code.
_CREDIT_MARKERS = (
    "insufficient_quota",
    "credit balance",
    "exceeded your current quota",
)

_HTTP_STATUS = {
    401: ("auth", "the provider rejected the credentials (HTTP 401)"),
    402: ("quota", "credit exhausted (HTTP 402) - the account has no usable balance"),
    403: ("entitlement", "the credentials are valid but not permitted to use this model (HTTP 403)"),
    404: ("model_not_found", "the provider has no such model (HTTP 404)"),
}


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


def _structured_values(exc: BaseException) -> List[str]:
    """The machine-readable code/type strings an error carries, lower-cased."""
    values: List[Any] = [getattr(exc, "code", None), getattr(exc, "type", None)]
    body = getattr(exc, "body", None)
    for level in (body, body.get("error") if isinstance(body, dict) else None):
        if isinstance(level, dict):
            values.extend((level.get("code"), level.get("type")))
    return [v.lower() for v in values if isinstance(v, str)]


def _message_text(exc: BaseException) -> str:
    """Free text that might say "credit" - used only to decide, never to print."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return " ".join(parts).lower()


def _structured_category(exc: BaseException) -> Optional[str]:
    for value in _structured_values(exc):
        if value in _STRUCTURED:
            return _STRUCTURED[value]
    return None


def is_provider_exception(exc: BaseException) -> bool:
    """Whether ``exc`` is a vendor/network failure rather than a bug in this code.

    Recognised by shape, like the classifier: an HTTP status carried by the exception,
    an SDK ``API*Error`` class, or a socket/urllib failure. Anything else - an
    ``AttributeError`` parsing a reply, a ``KeyError`` - is ours and keeps its traceback.
    """
    if isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            http.client.HTTPException,
        ),
    ):
        return True
    if _status_of(exc) is not None:
        return True
    return bool(_class_names(exc) & {"APIError", "APIStatusError", "APIConnectionError", "APITimeoutError"})


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The provider's own ``Retry-After`` (seconds), when the error carries one."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_provider_error(exc: BaseException) -> Tuple[ProbeCategory, str]:
    """Map a provider exception to ``(category, safe detail)``.

    Duck-typed so it needs no SDK. Order: transport failures by type, then the
    structured code/type, then the HTTP status, and only for a 400/429 that nothing
    structured explained, the message wording. ``quota`` means credit (a billing
    action); ``rate_limited`` and ``unavailable`` mean wait.
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
    suffix = f" (HTTP {status})" if status is not None else ""
    structured = _structured_category(exc)
    if structured == "quota":
        return "quota", f"credit exhausted{suffix} - the account has no usable balance"
    if structured == "rate_limited":
        return "rate_limited", f"rate limited{suffix} - too many requests, not an empty balance; wait"
    if structured == "unavailable":
        return "unavailable", f"the provider is overloaded or failing{suffix}; try again shortly"
    if structured in ("auth", "entitlement", "model_not_found") and status in (None, 400, 401, 403, 404):
        return _HTTP_STATUS[{"auth": 401, "entitlement": 403, "model_not_found": 404}[structured]]

    if status in _HTTP_STATUS:
        return _HTTP_STATUS[status]
    if status == 429 or status == 400:
        # No structured field settled it: the message is all there is.
        if any(marker in _message_text(exc) for marker in _CREDIT_MARKERS):
            return "quota", f"credit exhausted{suffix} - the account has no usable balance"
        if status == 429:
            return "rate_limited", f"rate limited{suffix} - too many requests, not an empty balance; wait"
    if status is not None and status >= 500:
        return "unavailable", f"the provider is overloaded or failing{suffix}; try again shortly"

    if "APIConnectionError" in names or isinstance(exc, (ConnectionError, OSError)):
        return "unreachable", "the connection failed before any response"

    return "unknown", f"unexpected {type(exc).__name__}{suffix}"


def as_provider_error(
    provider: str, exc: BaseException, *, detail: Optional[Any] = None
) -> ModelProviderError:
    """Wrap ``exc`` as a :class:`ModelProviderError`, optionally overriding the detail."""
    category, classified = classify_provider_error(exc)
    return ModelProviderError(provider, category, str(detail) if detail else classified)
