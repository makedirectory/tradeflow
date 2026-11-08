"""Model-provider failures: named apart, never retried, never carrying the key.

Offline: provider SDKs are replaced by fakes that raise the shapes the real ones do
(``status_code``, ``body``, the class names), so no SDK, key or network is needed.
"""

import json
import urllib.error

import pytest

from quantred import cli, settings
from quantred.research import llm
from quantred.research.llm import LLMResponse, OllamaClient, OpenAIClient
from quantred.research.provider_errors import ModelProviderError, classify_provider_error
from quantred.services import setup

FAKE_KEY = "sk-test-0123456789abcdefSECRET"  # gitleaks:allow - a fake key the test plants to check it is never printed


class _StatusError(Exception):
    """What both SDKs' status errors carry: status_code, body (the inner error dict),
    and the ``code`` / ``type`` they lift out of it - captured from real
    ``anthropic`` and ``openai`` errors raised through a mock transport."""

    def __init__(
        self, status_code, body=None, message="request failed", *, code=None, type=None, headers=None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.code = code
        self.type = type
        if headers is not None:

            class _Response:
                pass

            self.response = _Response()
            self.response.headers = headers


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


# --- classification ---------------------------------------------------------
@pytest.mark.parametrize(
    "exc, category",
    [
        (_StatusError(401), "auth"),
        (_StatusError(403), "entitlement"),
        (_StatusError(404), "model_not_found"),
        (_StatusError(429), "rate_limited"),
        (_StatusError(402, type="billing_error"), "quota"),
        (_StatusError(529, type="overloaded_error"), "unavailable"),
        (_StatusError(500, type="api_error"), "unavailable"),
        (_StatusError(503, type="server_error"), "unavailable"),
        (APITimeoutError("t"), "timeout"),
        (APIConnectionError("c"), "unreachable"),
        (TimeoutError(), "timeout"),
        (urllib.error.URLError(ConnectionRefusedError()), "unreachable"),
        (urllib.error.HTTPError("http://x", 404, "nf", {}, None), "model_not_found"),
        (RuntimeError("???"), "unknown"),
    ],
)
def test_each_provider_failure_is_named_apart(exc, category):
    assert classify_provider_error(exc)[0] == category


def test_a_credit_exhausted_429_is_distinguished_from_a_plain_rate_limit():
    credit = _StatusError(429, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}})
    plain = _StatusError(429, {"error": {"type": "rate_limit_exceeded"}}, "slow down")
    assert classify_provider_error(credit)[0] == "quota"
    assert "credit exhausted" in classify_provider_error(credit)[1]
    assert classify_provider_error(plain)[0] == "rate_limited"
    assert "rate limited" in classify_provider_error(plain)[1]
    assert "credit" not in classify_provider_error(plain)[1]


def test_a_structured_code_outranks_wording_that_mentions_billing():
    """A rate-limit reply can link the billing page; that is still "wait", not "pay".
    The old classifier searched the message for "billing" and called it exhausted credit
    - reproduced through the real openai and anthropic SDKs."""
    openai_rate = _StatusError(
        429,
        {"message": "Rate limit reached - see platform billing settings", "code": "rate_limit_exceeded"},
        "Rate limit reached - see account/billing",
        code="rate_limit_exceeded",
        type="requests",
    )
    anthropic_rate = _StatusError(
        429,
        {"type": "rate_limit_error", "message": "billing"},
        "see Plans & Billing",
        type="rate_limit_error",
    )
    assert classify_provider_error(openai_rate)[0] == "rate_limited"
    assert classify_provider_error(anthropic_rate)[0] == "rate_limited"
    openai_credit = _StatusError(429, {}, "quota", code="insufficient_quota", type="insufficient_quota")
    assert classify_provider_error(openai_credit)[0] == "quota"


def test_wording_is_the_fallback_when_nothing_structured_settles_it():
    """Anthropic reports an empty balance as a plain 400 `invalid_request_error`."""
    credit = _StatusError(
        400,
        {"type": "invalid_request_error", "message": "Your credit balance is too low to access the API."},
        type="invalid_request_error",
    )
    assert classify_provider_error(credit)[0] == "quota"
    # ... but a bare mention of "billing" is not evidence of an empty balance.
    assert (
        classify_provider_error(_StatusError(429, None, "slow down; billing page: ..."))[0] == "rate_limited"
    )
    other = _StatusError(400, {"message": "max_tokens must be positive"}, type="invalid_request_error")
    assert classify_provider_error(other)[0] == "unknown"


def test_a_detail_never_repeats_the_exception_text():
    exc = _StatusError(401, message=f"bad key {FAKE_KEY}")
    assert FAKE_KEY not in classify_provider_error(exc)[1]


# --- the probe --------------------------------------------------------------
class _CountingClient:
    model = "m"

    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def complete(self, system, user, max_tokens=1024):
        self.calls += 1
        assert max_tokens == 1
        if self.error:
            raise self.error
        return LLMResponse("pong", 2)


def test_the_probe_makes_exactly_one_request_even_on_failure():
    for error in (None, _StatusError(401), APIConnectionError("c")):
        client = _CountingClient(error)
        setup.probe_model("openai", client=client)
        assert client.calls == 1


def test_the_probe_reports_ok_and_failures():
    assert setup.probe_model("anthropic", client=_CountingClient()).category == "ok"
    assert setup.probe_model("anthropic", client=_CountingClient(_StatusError(401))).category == "auth"


def test_no_probe_output_contains_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    client = _CountingClient(_StatusError(401, message=f"Incorrect API key provided: {FAKE_KEY}"))
    result = setup.probe_model("openai", client=client)
    assert FAKE_KEY not in f"{result.detail} {result}"


def test_the_probe_model_defaults_to_the_research_default():
    result = setup.probe_model("ollama", client=_CountingClient())
    assert result.model == llm.DEFAULT_MODELS["ollama"]


# --- ollama over urllib ------------------------------------------------------
class _Resp:
    def __init__(self, payload):
        self._p = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._p).encode()


def test_ollama_failures_are_named_and_say_where_nothing_answered(monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    result = setup.probe_model("ollama", client=OllamaClient("m", base_url="http://127.0.0.1:11500"))
    assert result.category == "unreachable"
    assert "nothing answered at http://127.0.0.1:11500" in result.detail

    def missing(request, timeout):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    assert setup.probe_model("ollama", client=OllamaClient("m")).category == "model_not_found"

    def slow(request, timeout):
        raise TimeoutError()

    monkeypatch.setattr("urllib.request.urlopen", slow)
    timed_out = setup.probe_model("ollama", client=OllamaClient("m", timeout=300))
    assert timed_out.category == "timeout" and "300s" in timed_out.detail


def _capture(monkeypatch):
    seen = {}

    def fake(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return _Resp({"message": {"content": "x"}, "eval_count": 1})

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return seen


def test_num_ctx_is_sent_only_when_set(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    OllamaClient("m").complete("s", "u", 5)
    assert "num_ctx" not in seen["payload"]["options"]
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    OllamaClient("m").complete("s", "u", 5)
    assert seen["payload"]["options"] == {"num_predict": 5, "num_ctx": 4096}


def test_ollama_settings_reach_the_request(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11500")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "300")
    OllamaClient("m").complete("s", "u", 5)
    assert seen["url"] == "http://127.0.0.1:11500/api/chat"
    assert seen["timeout"] == 300.0


def test_a_full_chat_url_is_accepted_as_the_ollama_endpoint(monkeypatch):
    seen = _capture(monkeypatch)
    for given in (
        "http://127.0.0.1:11500/api/chat",
        "http://127.0.0.1:11500/api/generate/",
        "http://127.0.0.1:11500/",
    ):
        monkeypatch.setenv("OLLAMA_BASE_URL", given)
        assert settings.ollama_base_url() == "http://127.0.0.1:11500"
        OllamaClient("m").complete("s", "u", 5)
        assert seen["url"] == "http://127.0.0.1:11500/api/chat"


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "inf", "nan"])
def test_an_invalid_timeout_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("OLLAMA_TIMEOUT", bad)
    with pytest.raises(settings.SettingsError, match="OLLAMA_TIMEOUT"):
        settings.ollama_timeout()


def test_an_invalid_num_ctx_names_the_variable(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "lots")
    with pytest.raises(settings.SettingsError, match="OLLAMA_NUM_CTX"):
        settings.ollama_num_ctx()


# --- research-time classification -------------------------------------------
class _FakeOpenAI:
    """Just enough of the SDK surface; counts every request."""

    def __init__(self, error):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                raise error

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _run_research_with(monkeypatch, error):
    from quantred.research import agent as agent_mod
    from quantred.research import proposer as proposer_mod
    from quantred.services import data as data_mod

    sdk = _FakeOpenAI(error)
    client = OpenAIClient("gpt-x", client=sdk)

    class _Agent:
        session_id = "s"

        def __init__(self, strategy, data_client, proposer, cfg, seed=None, observer=None):
            self.proposer = proposer

        def run(self, universe, start, end):
            ctx = proposer_mod.ProposalContext(
                goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
            )
            self.proposer.propose(ctx)

    monkeypatch.setattr(cli, "extras_installed", lambda extra: True)
    monkeypatch.setattr(data_mod, "build_data_client", lambda *a, **k: object())
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])
    monkeypatch.setattr(agent_mod, "ResearchAgent", _Agent)
    monkeypatch.setattr(proposer_mod, "build_proposer", lambda *a, **k: proposer_mod.LLMProposer(client))
    args = cli.parse_cli(["research", "--provider", "openai", "--start", "2024-01-02", "--end", "2024-06-01"])
    with pytest.raises(SystemExit) as caught:
        cli.cmd_research(args)
    return sdk, caught.value


def test_a_credit_exhausted_429_is_quota_and_the_wrapper_does_not_retry_it(monkeypatch):
    """The SDK ctor test below covers "the SDK does not retry"; this one covers the
    wrapper's own retry loop, which would send a credit error `max_retries` more times
    if `quota` were ever counted transient."""
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    error = _StatusError(429, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}})
    sdk, exit_ = _run_research_with(monkeypatch, error)
    assert sdk.calls == 1
    assert exit_.code != 0
    message = str(exit_.code)
    assert "model provider openai: quota" in message
    assert "No retry was attempted" in message and "billing" in message


def test_a_plain_rate_limit_is_not_called_a_billing_problem_by_the_research_path(monkeypatch):
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    sdk, exit_ = _run_research_with(
        monkeypatch, _StatusError(429, {"error": {"type": "rate_limit_exceeded"}})
    )
    assert "rate limited" in str(exit_.code)
    assert "billing" not in str(exit_.code)
    assert sdk.calls == 1 + llm.LLMClient.max_retries  # retried, then gave up with the sentence


# --- the wrapper's retry policy ---------------------------------------------
class _Scripted(llm.LLMClient):
    """Fails with each scripted error in turn, then answers."""

    provider = "scripted"
    model = "m"

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def _complete(self, system, user, max_tokens):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse("done", 1)


@pytest.fixture
def waits(monkeypatch):
    recorded = []
    monkeypatch.setattr(llm, "_sleep", recorded.append)
    return recorded


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(529, type="overloaded_error"),
        _StatusError(500),
        _StatusError(429, type="rate_limit_error"),
    ],
)
def test_a_transient_failure_is_retried_and_then_succeeds(error, waits):
    client = _Scripted(error, error)
    assert client.complete("s", "u").text == "done"
    assert client.calls == 3
    assert waits == list(llm.RETRY_DELAYS)


def test_a_transient_failure_that_persists_stops_after_the_bounded_retries(waits):
    client = _Scripted(*[_StatusError(529, type="overloaded_error")] * 10)
    with pytest.raises(ModelProviderError) as caught:
        client.complete("s", "u")
    assert caught.value.category == "unavailable"
    assert client.calls == 1 + client.max_retries and len(waits) == client.max_retries


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(429, type="insufficient_quota", code="insufficient_quota"),
        _StatusError(402, type="billing_error"),
        _StatusError(401),
        _StatusError(403),
        _StatusError(404),
        APITimeoutError("t"),
    ],
)
def test_credit_auth_entitlement_missing_model_and_timeouts_are_never_retried(error, waits):
    client = _Scripted(error, error)
    with pytest.raises(ModelProviderError):
        client.complete("s", "u")
    assert client.calls == 1 and waits == []


def test_a_single_shot_client_never_retries(waits):
    client = _Scripted(_StatusError(529, type="overloaded_error"))
    client.max_retries = 0
    with pytest.raises(ModelProviderError):
        client.complete("s", "u")
    assert client.calls == 1 and waits == []


def test_the_probe_is_single_shot_even_for_a_transient_failure(waits):
    client = _Scripted(*[_StatusError(529, type="overloaded_error")] * 5)
    result = setup.probe_model("openai", client=client)
    assert result.category == "unavailable"
    assert client.calls == 1 and waits == []


def test_the_providers_retry_after_is_honoured_but_not_a_long_one(waits):
    advised = _StatusError(429, type="rate_limit_error", headers={"retry-after": "7"})
    client = _Scripted(advised)
    client.complete("s", "u")
    assert waits == [7.0]
    long = _StatusError(429, type="rate_limit_error", headers={"retry-after": "600"})
    client = _Scripted(long)
    with pytest.raises(ModelProviderError):
        client.complete("s", "u")
    assert client.calls == 1


@pytest.mark.parametrize("bug", [KeyError("choices"), AttributeError("no text"), ValueError("bad")])
def test_a_local_bug_propagates_with_its_traceback_instead_of_blaming_the_provider(bug, waits):
    client = _Scripted(bug)
    with pytest.raises(type(bug)) as caught:
        client.complete("s", "u")
    assert not isinstance(caught.value, ModelProviderError)
    assert client.calls == 1


def test_a_reply_that_is_not_json_is_the_endpoints_fault(monkeypatch):
    class _Html(_Resp):
        def __init__(self):
            super().__init__(None)

        def read(self):
            return b"<html>bad gateway</html>"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Html())
    with pytest.raises(ModelProviderError) as caught:
        OllamaClient("m", base_url="http://127.0.0.1:11500").complete("s", "u")
    assert caught.value.category == "unavailable"


def test_the_sdk_clients_are_built_with_retries_off(monkeypatch):
    """A credit error must reach the user once; the SDKs retry 429s by default."""
    import sys
    import types

    built = {}

    def stub(name, cls):
        module = types.ModuleType(name)

        def ctor(**kwargs):
            built[name] = kwargs
            return object()

        setattr(module, cls, ctor)
        monkeypatch.setitem(sys.modules, name, module)

    stub("anthropic", "Anthropic")
    stub("openai", "OpenAI")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    llm.AnthropicClient()._ensure()
    llm.OpenAIClient()._ensure()
    assert built["anthropic"]["max_retries"] == 0
    assert built["openai"]["max_retries"] == 0


def test_a_provider_error_is_not_swallowed_into_the_wrapper_twice():
    client = OpenAIClient("m", client=_FakeOpenAI(ModelProviderError("openai", "auth", "x")))
    with pytest.raises(ModelProviderError) as caught:
        client.complete("s", "u")
    assert caught.value.category == "auth"
