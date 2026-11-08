"""A reused (memoized) backtest says what it lacks instead of going quiet.

The trial store keeps the headline metrics and an identity block, not the whole
result. Serving that as though it were a fresh payload left capital, cost and trade
fields blank or zero with nothing saying they were absent. These tests pin the
payload, its two builders (CLI and service) and the HTML card that renders it.
"""

import json
from datetime import datetime

import pytest

from quantred import cli as main
from quantred.analytics.htmlreport import render_html
from quantred.analytics.performance import LOW_SAMPLE_TRADES
from quantred.marketdata.client import MarketDataClient
from quantred.services import analysis, audit
from tests.fakes import FakeMarketData

SYMBOLS = ["AAA", "BBB"]
START, END = "2024-01-02", "2024-06-01"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


@pytest.fixture
def client(monkeypatch):
    c = MarketDataClient(FakeMarketData(SYMBOLS, n=400, freq="1D"))
    monkeypatch.setattr(main, "build_research_data", lambda **kwargs: c)
    return c


def _service(client, **kw):
    return analysis.run_backtest(client, "demo_trend", SYMBOLS, main._date(START), main._date(END), **kw)


def _cli_payload(monkeypatch, capsys):
    """Run the CLI backtest and return the dict it hands the HTML writer."""
    seen = []
    monkeypatch.setattr(main, "_write_html", lambda args, result, kind, extras=None: seen.append(result))
    argv = ["backtest", "--strategy", "demo_trend", "--scanner", "none", "--symbols", "AAA,BBB"]
    args = main.build_parser().parse_args(argv + ["--start", START, "--end", END, "--html", "x.html"])
    args.func(args)
    return seen[-1], capsys.readouterr().out


def test_a_reused_backtest_names_what_it_lacks_and_keeps_what_the_request_fixes(client):
    """The failure: the reused payload had a few metrics and nothing said the rest was absent."""
    fresh = _service(client)
    assert not fresh.get("memoized")
    reused = _service(client)

    assert reused["memoized"] is True
    assert reused["trial_id"] and reused["trial_ts"]
    for field in ("total_cost", "cost_drag_pct", "executability", "trades_csv", "resolved_config"):
        assert field in reused["not_recorded"], field
        assert field not in reused, f"{field} is absent, so it must not appear as a value"
    assert "force" in reused["note"]
    # Request-determined values are kept and labelled as the request's.
    assert reused["initial_capital"] == 100_000.0
    assert reused["symbols"] == SYMBOLS
    assert reused["gross"] is False
    assert reused["benchmark_requested"] == "SPY"
    assert "benchmark" not in reused  # whether it was scored against is not recorded
    assert {"initial_capital", "window", "symbols"} <= set(reused["from_request"])
    json.dumps(reused)


def test_low_sample_is_derived_from_the_stored_trade_count_by_the_metrics_threshold(client):
    reused = (_service(client), _service(client))[1]
    trades = reused["metrics"]["total_trades"]
    assert reused["total_trades"] == trades
    assert reused["metrics"]["low_sample"] is (trades < LOW_SAMPLE_TRADES)
    assert "low_sample" in reused["derived"]
    assert reused["final_capital_derived"] == pytest.approx(
        100_000.0 * (1 + reused["metrics"]["total_return"] / 100)
    )
    assert "final_capital" not in reused


def test_low_sample_is_not_recorded_when_the_trade_count_was_not_stored():
    cached = {
        "id": "t1",
        "ts": "2024-01-01T00:00:00+00:00",
        "metrics_json": json.dumps({"sharpe_ratio": 1.0}),
    }
    payload = analysis.memoized_backtest_payload(
        cached,
        run_id="r",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 2),
        end=datetime(2024, 6, 1),
        capital=1000.0,
        gross=False,
    )
    assert "total_trades" in payload["not_recorded"] and "low_sample" in payload["not_recorded"]
    assert payload["total_trades"] is None
    assert payload["final_capital_derived"] is None
    assert "low_sample" not in payload["metrics"]


def test_cli_and_service_build_the_same_reused_payload(client, monkeypatch, capsys):
    """One trial recorded over the CLI, reused over both surfaces, must read identically."""
    _cli_payload(monkeypatch, capsys)  # records the trial
    cli_reused, out = _cli_payload(monkeypatch, capsys)
    service_reused = _service(client)

    assert cli_reused["memoized"] and service_reused["memoized"]
    cli_reused.pop("run_id"), service_reused.pop("run_id")
    assert cli_reused == service_reused
    assert "Not recorded for this reused result:" in out
    assert "total_cost" in out and "--force" in out


def test_html_run_card_never_renders_a_blank_or_a_zero_for_an_absent_value(client):
    _service(client)
    reused = _service(client)
    html = render_html(reused, "backtest")

    assert "<dt>Transaction cost</dt><dd>not recorded</dd>" in html
    assert "<dt>Cost drag</dt><dd>not recorded</dd>" in html
    for bad in ("0.00%", "—%", "<dd></dd>", "<dd>None</dd>", "<dd>0</dd>", "<dd>0.00</dd>"):
        assert bad not in html, bad
    assert "(as requested)" in html and "derived from total return" in html


def test_html_run_card_escapes_the_not_recorded_names():
    payload = {"metrics": {}, "memoized": True, "not_recorded": ["<script>x</script>"]}
    html = render_html(payload, "backtest")
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_asking_the_cli_for_a_trade_table_is_not_answered_by_a_memo_without_one(client, monkeypatch, capsys):
    """`--record-trades` over an identical trial that kept no table was served from the
    memo, so nothing was recorded and the reader was told nothing was missing. The service
    gained this rule with its own test; this is the CLI side of the same rule."""
    from quantred.store.trials import TrialStore, db_path_for_journal

    _cli_payload(monkeypatch, capsys)
    argv = ["backtest", "--strategy", "demo_trend", "--scanner", "none", "--symbols", "AAA,BBB"]
    args = main.build_parser().parse_args(argv + ["--start", START, "--end", END, "--record-trades"])
    args.func(args)

    journal = audit.DEFAULT_TRIAL_JOURNAL
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        rows = store.list_trials(kind="backtest", limit=10, all_accounting=True)
        assert len(rows) == 2
        assert any(store.get_trial(r["id"]).get("trades") for r in rows)


def test_a_backtest_at_another_capital_is_not_served_the_first_ones_result(client):
    """Found in review: capital was not in the dedup key, so a request at $4,000 was
    answered by a run at $100,000, its card said the capital was as requested, and it
    derived a final capital that no run produced."""
    first = _service(client, capital=100_000.0)
    assert not first.get("memoized")
    other = _service(client, capital=4_000.0)
    assert not other.get("memoized")
    assert other["initial_capital"] == 4_000.0
    assert _service(client, capital=4_000.0)["memoized"] is True


def test_asking_a_cli_walk_forward_for_its_trade_table_is_not_answered_by_a_memo_without_one(
    client, monkeypatch, capsys
):
    """The walk-forward half of the rule above: without it, deleting the CLI walk-forward's
    require_trade_table passed the whole suite."""
    from quantred.store.trials import TrialStore, db_path_for_journal

    base = ["walkforward", "--strategy", "demo_trend", "--scanner", "none", "--symbols", "AAA,BBB"]
    base += ["--start", "2024-01-02", "--end", "2024-12-31", "--folds", "2", "--max-evals", "2"]
    for extra in ([], ["--record-trades"]):
        args = main.build_parser().parse_args(base + extra)
        args.func(args)
    capsys.readouterr()

    journal = audit.DEFAULT_TRIAL_JOURNAL
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        rows = store.list_trials(kind="walkforward", limit=10, all_accounting=True)
        assert len(rows) == 2
        assert any(store.get_trial(r["id"]).get("trades") for r in rows)


def test_a_reused_backtest_still_runs_the_diagnostics_it_can(client, monkeypatch, capsys):
    """After an identical earlier backtest, `--causality` and `--cost-stress` printed
    nothing: the reused result returned before any diagnostic ran, and nothing said so.
    The probes and the stress curve run their own backtests, so they run; excursion
    needs the original simulation and says how to get it."""
    argv = ["backtest", "--strategy", "demo_trend", "--scanner", "none", "--symbols", "AAA,BBB"]
    argv += ["--start", START, "--end", END]
    main.build_parser().parse_args(argv).func(main.build_parser().parse_args(argv))
    capsys.readouterr()
    args = main.build_parser().parse_args(argv + ["--causality", "--cost-stress", "--excursion"])
    args.func(args)
    out = capsys.readouterr().out
    assert "REUSED" in out or "reused" in out.lower()
    assert "Causality probes" in out and "=== Cost stress" in out
    assert "--excursion did not run" in out and "--force" in out
