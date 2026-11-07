"""What `research` and `demo-agent` print and refuse, driven through the real command.

A scripted model client and fake bars stand in for the provider and the market; nothing
here reaches a network or a key.
"""

import json

import pytest

from tests.fakes import FakeMarketData
from tests.test_research import _isolate  # noqa: F401 - registers the test strategy
from tests.test_research_rounds import ScriptedClient, ScriptedProposer
from tradeflow import cli
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.research.proposer import Proposal
from tradeflow.research.provider_errors import ModelProviderError
from tradeflow.services import audit
from tradeflow.services import data as data_mod

_ARGV = [
    "research",
    "--provider",
    "ollama",
    "--strategy",
    "periodic",
    "--start",
    "2024-01-02",
    "--end",
    "2025-06-01",
    "--folds",
    "3",
    "--embargo-days",
    "2",
]


def _run(monkeypatch, tmp_path, capsys, replies, extra=()):
    from tradeflow.research import proposer as proposer_mod

    monkeypatch.setattr(audit, "default_trial_journal", lambda: tmp_path / "journal.jsonl")
    monkeypatch.setattr(
        data_mod,
        "build_data_client",
        lambda *a, **k: MarketDataClient(FakeMarketData(["AAA", "BBB"], n=600, freq="1D")),
    )
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA", "BBB"])
    monkeypatch.setattr(
        proposer_mod, "build_proposer", lambda *a, **k: ScriptedProposer(ScriptedClient(replies))
    )
    args = cli.parse_cli([*_ARGV, *extra])
    try:
        cli.cmd_research(args)
    except SystemExit as exc:
        return capsys.readouterr().out, exc
    return capsys.readouterr().out, None


def _reply(params):
    return json.dumps({"hypothesis": "cycle captures swings", "params": params})


def test_a_session_of_only_rejections_says_nothing_was_evaluated_and_names_each_round(
    monkeypatch, tmp_path, capsys
):
    out, _ = _run(monkeypatch, tmp_path, capsys, ["I think you should try a faster EMA!", _reply([1, 2])])
    assert "round 1: rejected - unparseable reply" in out
    assert "round 2: rejected - params must be an object" in out
    assert "0 evaluated, 2 rejected" in out
    assert "Nothing was evaluated" in out
    # Not the "cumulative trials" line over a count of zero, and not a holdout that was scored.
    assert "cumulative" not in out
    assert "Holdout: not scored" in out and "Holdout (scored once" not in out


def test_an_evaluated_round_is_printed_with_the_basis_of_its_sharpe(monkeypatch, tmp_path, capsys):
    out, _ = _run(monkeypatch, tmp_path, capsys, [_reply({"buy_every": 5})])
    assert "round 1: evaluated - median OOS Sharpe across folds" in out
    assert "Trials this session: 1" in out
    assert "net of costs" in out


def test_gross_is_a_flag_and_net_is_the_default(monkeypatch, tmp_path, capsys):
    out, _ = _run(monkeypatch, tmp_path, capsys, [], extra=["--gross"])
    assert "Cost basis: gross returns (--gross" in out
    journal = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert journal[0]["inputs"]["cost"] == {"gross": True}


def test_the_default_session_is_priced_like_every_other_command(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, capsys, [])
    journal = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert journal[0]["inputs"]["cost"] == cli._cost_key(cli.parse_cli([*_ARGV]))
    assert journal[0]["inputs"]["cost"]["gross"] is False


def test_a_second_session_over_the_same_holdout_is_warned_about(monkeypatch, tmp_path, capsys):
    first, _ = _run(monkeypatch, tmp_path, capsys, [_reply({"buy_every": 5})])
    assert "WARNING" not in first
    # The shortlist is what gets scored, so use the relaxed gates by writing a scored look.
    from tradeflow.research.agent import prior_holdout_looks

    journal = tmp_path / "journal.jsonl"
    start = json.loads(journal.read_text().splitlines()[0])["inputs"]
    with journal.open("a") as handle:
        handle.write(
            json.dumps({"tool": "research:holdout_score", "inputs": {"session_id": start["session_id"]}})
            + "\n"
        )
    assert prior_holdout_looks(str(journal), start["holdout_window"], start["symbols"]) == 1
    second, _ = _run(monkeypatch, tmp_path, capsys, [_reply({"buy_every": 7})])
    assert "WARNING: 1 earlier session" in second


@pytest.mark.parametrize(
    "flags",
    [
        ["--holdout-days", "0"],
        ["--max-trials", "0"],
        ["--max-trials", "-3"],
        ["--max-dry-rounds", "0"],
        ["--max-tokens", "0"],
    ],
)
def test_degenerate_budgets_are_refused_by_the_parser(flags, capsys):
    with pytest.raises(SystemExit) as refused:
        cli.parse_cli([*_ARGV, *flags])
    assert refused.value.code != 0


def test_a_holdout_longer_than_the_range_is_a_clean_refusal(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        _run_via_main(monkeypatch, tmp_path, ["--holdout-days", "5000"])
    assert "leaves no research window" in str(refused.value)


def _run_via_main(monkeypatch, tmp_path, extra):
    from tradeflow.research import proposer as proposer_mod

    monkeypatch.setattr(audit, "default_trial_journal", lambda: tmp_path / "journal.jsonl")
    monkeypatch.setattr(
        data_mod,
        "build_data_client",
        lambda *a, **k: MarketDataClient(FakeMarketData(["AAA", "BBB"], n=600, freq="1D")),
    )
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA", "BBB"])
    monkeypatch.setattr(proposer_mod, "build_proposer", lambda *a, **k: ScriptedProposer(ScriptedClient([])))
    monkeypatch.setattr("sys.argv", ["tradeflow", *_ARGV, *extra])
    cli.main()


# --- the provider's failure, said as it happened ---
def test_a_long_retry_after_is_reported_as_not_retried():
    exc = ModelProviderError("openai", "rate_limited", "rate limited", attempts=1, advised_wait=120.0)
    text = "\n".join(cli._provider_failure_lines(exc))
    assert "Not retried" in text and "120s" in text
    assert "Retried a few times" not in text


def test_a_failure_after_retries_counts_them():
    exc = ModelProviderError("openai", "unreachable", "down", attempts=3)
    assert "Retried 2 times" in "\n".join(cli._provider_failure_lines(exc))


def test_a_failure_mid_session_says_what_it_left_behind():
    class _Agent:
        session_id = "abc"
        progress = {"rounds": 4, "evaluated": 3}

    exc = ModelProviderError("openai", "quota", "credit exhausted")
    text = "\n".join(cli._provider_failure_lines(exc, _Agent()))
    assert "abc" in text and "4 rounds" in text
    assert "3 trials" in text and "multiple-testing" in text
    assert "No holdout was scored" in text


def test_retries_are_announced_while_they_happen(monkeypatch):
    from tradeflow.research import llm

    seen = []

    class _Flaky(llm.LLMClient):
        provider = "p"
        model = "m"

        def __init__(self):
            self.n = 0

        def _complete(self, system, user, max_tokens):
            self.n += 1
            if self.n < 2:
                raise ConnectionError("down")
            return llm.LLMResponse("ok", 1)

    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    client = _Flaky()
    client.on_retry = lambda *a: seen.append(a)
    client.complete("s", "u")
    assert seen == [("unreachable", 1, client.max_retries, 1.0)]


# --- demo-agent narrates each rejection accurately ---
@pytest.mark.parametrize(
    ("reason", "fragment", "absent"),
    [
        (
            "duplicate: identical config already tried this campaign",
            "found in the trial history",
            "no bars loaded",
        ),
        ("unrunnable: backtest failed for all 2 symbol(s)", "bars were loaded", "no bars loaded"),
        ("sandbox: generated code may not import 'os'", "no bars loaded", "bars were loaded"),
        ("unparseable reply", "no bars loaded", "bars were loaded"),
    ],
)
def test_demo_agent_narrates_each_rejection_by_what_it_actually_cost(reason, fragment, absent):
    text = cli._reject_consequence(reason)
    assert fragment in text and absent not in text


def test_demo_agent_with_no_journal_adds_nothing_to_the_real_journal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(audit, "default_trial_journal", lambda: tmp_path / "journal.jsonl")
    monkeypatch.setattr(
        data_mod,
        "build_data_client",
        lambda *a, **k: MarketDataClient(FakeMarketData(["AAA", "BBB"], n=600, freq="1D")),
    )
    monkeypatch.setattr(
        "tradeflow.research.demo_proposals.DEMO_PROPOSALS",
        [Proposal(hypothesis="cycle", kind="tune", strategy="periodic", params={"buy_every": 5})],
    )
    args = cli.parse_cli(
        [
            "demo-agent",
            "--strategy",
            "periodic",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2025-06-01",
            "--folds",
            "3",
            "--no-journal",
        ]
    )
    cli.cmd_demo_agent(args)
    out = capsys.readouterr().out
    assert "discarded: --no-journal" in out
    assert not (tmp_path / "journal.jsonl").exists()
