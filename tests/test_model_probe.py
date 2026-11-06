"""Model-provider failures: named apart, never retried, never carrying the key.

Offline: provider SDKs are replaced by fakes that raise the shapes the real ones do
(``status_code``, ``body``, the class names), so no SDK, key or network is needed.
"""

import json
import urllib.error

import pytest

from tradeflow import cli, settings
from tradeflow.research import llm
from tradeflow.research.llm import LLMResponse, OllamaClient, OpenAIClient
from tradeflow.research.provider_errors import ModelProviderError, classify_provider_error
from tradeflow.services import setup

FAKE_KEY = "sk-test-0123456789abcdefSECRET"


class _StatusError(Exception):
    def __init__(self, status_code, body=None, message="request failed"):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


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
        (_StatusError(429), "quota"),
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
    assert classify_provider_error(plain)[0] == "quota"
    assert "rate limited" in classify_provider_error(plain)[1]
    assert "credit" not in classify_provider_error(plain)[1]


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
    from tradeflow.research import agent as agent_mod
    from tradeflow.research import proposer as proposer_mod
    from tradeflow.services import data as data_mod

    sdk = _FakeOpenAI(error)
    client = OpenAIClient("gpt-x", client=sdk)

    class _Agent:
        session_id = "s"

        def __init__(self, strategy, data_client, proposer, cfg, seed=None):
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


def test_a_credit_exhausted_429_is_quota_and_is_not_retried(monkeypatch):
    error = _StatusError(429, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}})
    sdk, exit_ = _run_research_with(monkeypatch, error)
    assert sdk.calls == 1
    assert exit_.code != 0
    message = str(exit_.code)
    assert "model provider openai: quota" in message
    assert "No retry was attempted" in message and "billing" in message


def test_a_plain_rate_limit_is_not_called_a_billing_problem_by_the_research_path(monkeypatch):
    _, exit_ = _run_research_with(monkeypatch, _StatusError(429, {"error": {"type": "rate_limit_exceeded"}}))
    assert "rate limited" in str(exit_.code)


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
