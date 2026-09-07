"""Dry run: what a contract would try to do, proven not to try it.

The load-bearing test here is not that the report says "dry run". It is that a signal
which *would* submit under a live broker cannot reach a real submit path — asserted by
making every real submit method fail loudly if it is ever called, and then driving the
decision path with a signal that reaches submission.

A report can say anything. The point of the mode is that the capability is absent.
"""

import json
from datetime import datetime

import pytest

from tests.fakes import ScriptedStrategy
from tradeflow.brokers.base import AccountSnapshot, Position
from tradeflow.brokers.dryrun import BANNER, DryRunBroker, DryRunTradingAttempted
from tradeflow.brokers.errors import BrokerError
from tradeflow.execution import decision as decisions
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.services import dryrun
from tradeflow.strategies import signals

CAPITAL = 250_000.0


def _trader(broker, **overrides):
    strategy = ScriptedStrategy(
        {
            "pivot": 100.0,
            "risk_per_trade": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "position_limits": {
                "max_positions": 4,
                "max_position_size": 100_000.0,
                "max_total_risk": 0.5,
                **overrides,
            },
        }
    )
    return LiveTrader(broker, strategy, respect_market_hours=False)


# --- the capability is absent, not gated -------------------------------------------
def test_a_signal_that_would_submit_never_reaches_a_real_submit_path(monkeypatch):
    """The test that matters. Every real submit method is replaced with something that
    fails the test outright, and then a BUY is driven all the way through the decision
    path. If the mode is a flag somebody forgot on one branch, this fails.

    Asserting the report says "dry run" would prove nothing: a report is a string.
    """
    from tradeflow.brokers import base as broker_base

    reached = []

    def _never(*args, **kwargs):
        reached.append("a real broker submit path was reached during a dry run")
        raise AssertionError(reached[-1])

    for method in (
        "submit_market_order",
        "submit_bracket_order",
        "close_position",
        "close_all_positions",
        "cancel_order",
        "cancel_all_orders",
    ):
        monkeypatch.setattr(broker_base.Broker, method, _never, raising=False)

    broker = DryRunBroker(capital=CAPITAL)
    decision = _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)

    assert reached == []
    # And it got far enough to be a real intent rather than an early skip.
    assert decision.allowed is False
    assert decision.plan is not None
    assert decision.plan.qty > 0


def test_every_order_method_refuses():
    broker = DryRunBroker(capital=CAPITAL)

    for call in (
        lambda: broker.submit_market_order("AAA", 1, "buy"),
        lambda: broker.submit_bracket_order("AAA", 1, "buy", 90.0, 110.0),
        lambda: broker.cancel_order("id"),
        lambda: broker.cancel_all_orders(),
        lambda: broker.close_position("AAA"),
        lambda: broker.close_all_positions(),
    ):
        with pytest.raises(DryRunTradingAttempted):
            call()


def test_the_refusal_is_a_broker_error_so_the_intent_survives_it():
    """The design turns on this. `LiveTrader` builds its plan *before* submitting so a
    broker refusal still records what would have been sent — a refusal that was not a
    `BrokerError` would escape that handler and lose the intent, and the report would
    need a parallel capture path kept in step with the real one."""
    assert issubclass(DryRunTradingAttempted, BrokerError)

    decision = _trader(DryRunBroker(capital=CAPITAL)).handle_signal("AAA", signals.BUY, price=100.0)

    assert decision.plan.stop_loss < 100.0 < decision.plan.take_profit


# --- the account is stated, never discovered ----------------------------------------
def test_capital_comes_from_the_config_then_an_explicit_flag_and_says_which():
    """The source travels with the number: a reader judging whether these caps are the
    right ones needs to know whether the capital came from the validated config or was
    typed at the prompt. Those are different claims about what is being rehearsed."""
    assert dryrun.resolve_capital(config_capital=250_000.0) == (250_000.0, "from config")
    assert dryrun.resolve_capital(config_capital=250_000.0, explicit=50_000.0) == (
        50_000.0,
        "--capital",
    )
    assert dryrun.resolve_capital(explicit=50_000.0) == (50_000.0, "--capital")


def test_a_dry_run_with_no_stated_capital_refuses_rather_than_defaulting():
    """No fallback to broker equity, no default account size. The caps a dry run reports
    are only meaningful against the capital they bound, so inventing one answers a
    question about a book nobody chose."""
    with pytest.raises(ValueError, match="needs a stated capital"):
        dryrun.resolve_capital()
    with pytest.raises(ValueError, match="positive capital"):
        dryrun.resolve_capital(config_capital=0.0)
    with pytest.raises(ValueError, match="stated capital"):
        DryRunBroker(capital=None)


def test_supplied_positions_reserve_buying_power():
    """Otherwise "dry run from this book" would secretly mean "from flat capital plus
    this book", and every size would be overstated."""
    held = Position(
        symbol="AAA",
        qty=100,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=10_000.0,
        unrealized_pl=0.0,
    )
    flat = DryRunBroker(capital=CAPITAL).get_account()
    loaded = DryRunBroker(capital=CAPITAL, positions=[held]).get_account()

    assert isinstance(flat, AccountSnapshot)
    assert flat.buying_power == pytest.approx(CAPITAL)
    assert loaded.buying_power == pytest.approx(CAPITAL - 10_000.0)
    assert loaded.equity == pytest.approx(CAPITAL)  # equity is the stated capital


def test_the_starting_book_records_where_it_came_from():
    """A dry run over an existing book and one from flat answer different questions, so
    an empty book must never be ambiguous between "chosen" and "assumed"."""
    assert dryrun.build_broker(capital=CAPITAL).positions_source == "flat"
    held = [Position("AAA", 1, "long", 100.0, 100.0, 100.0, 0.0)]
    assert dryrun.build_broker(capital=CAPITAL, positions=held).positions_source == "adopted"
    assert (
        dryrun.build_broker(
            capital=CAPITAL, positions=held, positions_source="paper snapshot"
        ).positions_source
        == "paper snapshot"
    )


def test_no_trade_update_stream_is_offered():
    """No fills means no stream, and claiming otherwise would have a caller wait on
    updates that can never arrive."""
    assert DryRunBroker(capital=CAPITAL).supports_trade_updates() is False
    assert DryRunBroker(capital=CAPITAL).list_open_orders() == []


# --- the vocabulary -----------------------------------------------------------------
def _decision(reason, code=None, allowed=False, plan=None):
    return decisions.Decision(
        symbol="AAA",
        signal=signals.BUY,
        allowed=allowed,
        reason=reason,
        guards_consulted=(),
        reason_code=code,
        plan=plan,
    )


def test_a_capped_book_is_would_bind_and_a_closed_market_is_would_skip():
    """Only a configured cap is a bind. Calling a market-hours skip a "bind" would
    attribute to the book a limit the clock imposed."""
    report = dryrun.dry_run_report(
        [
            _decision("this is a dry run: ..."),
            _decision("book is full: 4 of 4", code=decisions.BOOK_FULL),
            _decision("gross exposure capped: $9k of $8k", code=decisions.GROSS_EXPOSURE),
            _decision("market is closed"),
            _decision("no signal"),
        ],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="flat",
        universe=["AAA"],
    )

    assert report["counts"] == {
        "would_submit": 1,
        "would_bind": 2,
        "would_skip": 2,
        "unable_to_evaluate": 0,
    }


def test_the_report_never_borrows_executions_words():
    """A report that says "submitted" or "filled" will be read as execution evidence,
    and the one thing this mode must not be mistaken for is the small-real session."""
    report = dryrun.dry_run_report(
        [_decision("this is a dry run: ...")],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="flat",
        universe=["AAA"],
    )
    # The banner and the not-covered note are *allowed* to use those words — they say
    # what did not happen. What must not use them is anything describing a decision.
    findings = {k: v for k, v in report.items() if k not in ("banner", "not_covered")}
    text = repr(findings).lower()

    for forbidden in ("submitted", "filled", "slippage", "blocked trade", "executed"):
        assert forbidden not in text, f"the report uses execution's vocabulary: {forbidden}"
    assert "would_submit" in report and "would_skip" in report and "would_bind" in report
    # And the banner does use "submitted" — in the negative, which is the point.
    assert "no orders can be submitted" in report["banner"]


def test_the_banner_is_always_present_and_says_the_capability_is_absent():
    report = dryrun.dry_run_report(
        [], capital=CAPITAL, capital_source="from config", positions_source="flat", universe=[]
    )

    assert report["banner"] == BANNER
    assert "no trading capability" in report["banner"]


def test_the_report_journals_nothing_and_carries_no_trial_id():
    """A dry run measures nothing — no fills, no realized P&L, no return series — so
    recording it would spend a family's statistical budget on a rehearsal."""
    report = dryrun.dry_run_report(
        [], capital=CAPITAL, capital_source="from config", positions_source="flat", universe=["AAA"]
    )

    assert report["journaled"] is False
    assert "trial_id" not in report
    for absent in ("fills", "slippage", "fees", "cost_drag", "metrics"):
        assert absent not in report


def test_what_the_mode_cannot_observe_is_stated_rather_than_left_absent():
    """A reader looking for slippage should learn here that this mode cannot produce it,
    not conclude from an empty section that the run was clean."""
    report = dryrun.dry_run_report(
        [], capital=CAPITAL, capital_source="from config", positions_source="flat", universe=[]
    )

    assert "slippage" in report["not_covered"]
    assert "small-real" in report["not_covered"]


def test_every_bucket_is_reported_even_when_empty():
    """ "No signal bound a cap" and "the caps were never reached" are different findings,
    and an omitted section renders them alike."""
    report = dryrun.dry_run_report(
        [], capital=CAPITAL, capital_source="from config", positions_source="flat", universe=[]
    )

    assert report["would_submit"] == [] and report["would_bind"] == [] and report["would_skip"] == []
    assert report["unable_to_evaluate"] == []
    assert report["counts"] == {
        "would_submit": 0,
        "would_bind": 0,
        "would_skip": 0,
        "unable_to_evaluate": 0,
    }


def test_the_stated_capital_and_book_source_travel_in_the_report():
    report = dryrun.dry_run_report(
        [],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="paper snapshot",
        universe=["AAA", "BBB"],
        as_of=datetime(2024, 6, 1),
    )

    assert report["capital"] == CAPITAL
    assert report["starting_positions"] == "paper snapshot"
    assert report["universe"] == ["AAA", "BBB"]


# --- the CLI branch builds no broker at all -----------------------------------------
def test_the_dry_run_branch_never_calls_the_broker_factory(monkeypatch, capsys):
    """The guarantee that keeps this from becoming "paper trading, but don't submit".

    A dry run must not construct a real broker and then decline to use it — the factory
    is never reached. Asserted by making it fail the test outright, and by making the
    Alpaca broker builder fail too, so neither the CLI's wrapper nor the layer under it
    can be entered.
    """
    import pandas as pd

    from tradeflow import cli
    from tradeflow.utils.timeutils import NEW_YORK

    def _never(*args, **kwargs):
        raise AssertionError("a dry run reached the broker factory")

    monkeypatch.setattr(cli, "build_data_and_broker", _never)
    monkeypatch.setattr("tradeflow.brokers.alpaca.factory.build_broker", _never, raising=False)

    idx = pd.date_range("2024-01-02", periods=60, freq="D", tz=NEW_YORK)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}, index=idx
    )

    class _Data:
        def get_bars(self, symbols, start, end, **kwargs):
            return {s: bars for s in symbols}

    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda **kw: _Data())
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])

    args = cli.build_parser().parse_args(
        ["live", "--dry-run", "--strategy", "demo_trend", "--symbols", "AAA", "--capital", "8000"]
    )
    args.flags_given = {"dry_run", "capital"}
    cli.cmd_live(args)

    printed = capsys.readouterr().out
    assert "DRY RUN" in printed
    assert "no trading capability" in printed


def test_a_dry_run_needs_no_broker_credentials(monkeypatch, capsys):
    """Credentials gate a *broker*, and a dry run has none. Requiring them would make
    the mode unusable in exactly the situation it is most useful — before an account
    exists."""
    import pandas as pd

    from tradeflow import cli
    from tradeflow.utils.timeutils import NEW_YORK

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(cli, "build_data_and_broker", lambda *a, **k: pytest.fail("broker factory reached"))

    idx = pd.date_range("2024-01-02", periods=60, freq="D", tz=NEW_YORK)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}, index=idx
    )

    class _Data:
        def get_bars(self, symbols, start, end, **kwargs):
            return {s: bars for s in symbols}

    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda **kw: _Data())
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])

    args = cli.build_parser().parse_args(
        ["live", "--dry-run", "--strategy", "demo_trend", "--symbols", "AAA", "--capital", "8000"]
    )
    args.flags_given = {"dry_run", "capital"}
    cli.cmd_live(args)

    assert "DRY RUN" in capsys.readouterr().out


def test_dry_run_and_live_money_together_are_refused():
    """Refused rather than ignored: someone who typed --live-money believes this run can
    trade, and silently honouring --dry-run leaves them right about the intent and wrong
    about the run."""
    from tradeflow.cli import _refuse_inert_flags, build_parser

    args = build_parser().parse_args(["live", "--dry-run", "--live-money"])
    args.flags_given = {"dry_run", "live_money"}

    with pytest.raises(SystemExit, match="contradict each other"):
        _refuse_inert_flags(args)


def test_dry_run_alone_is_accepted():
    """Both directions: the guard must not reject the flag it exists to permit."""
    from tradeflow.cli import _refuse_inert_flags, build_parser

    args = build_parser().parse_args(["live", "--dry-run"])
    args.flags_given = {"dry_run"}

    _refuse_inert_flags(args)  # no SystemExit


# --- a symbol nobody could evaluate does not vanish ----------------------------------
def test_a_symbol_with_too_little_history_is_reported_not_dropped():
    """`WOULD SKIP` is an outcome the strategy reached; this is the absence of one.
    Folding the two together reports a universe as fully evaluated when part of it was
    never asked, and the symbol simply disappears."""
    report = dryrun.dry_run_report(
        [_decision("no signal")],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="flat",
        universe=["AAA", "XYZ"],
        unable_to_evaluate=[{"symbol": "XYZ", "reason": "insufficient history: needs 102 bars, has 57"}],
    )

    assert report["counts"]["unable_to_evaluate"] == 1
    assert report["unable_to_evaluate"][0]["symbol"] == "XYZ"
    assert "needs 102 bars, has 57" in report["unable_to_evaluate"][0]["reason"]
    # It is not counted as a skip, which would make it look like a decision.
    assert report["counts"]["would_skip"] == 1


def test_the_shortfall_reaches_the_summary_not_only_the_detail():
    """A count buried in a section somebody has to scroll to is still easy to miss, and
    "12 symbols evaluated" and "12 symbols, 7 evaluated" are different runs."""
    from tradeflow.analytics.reporting import format_dry_run

    report = dryrun.dry_run_report(
        [_decision("no signal")],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="flat",
        universe=["AAA", "XYZ"],
        unable_to_evaluate=[{"symbol": "XYZ", "reason": "insufficient history: needs 102 bars, has 57"}],
    )
    printed = format_dry_run(report)

    assert report["n_evaluated"] == 1 and report["n_unable_to_evaluate"] == 1
    assert "could not be evaluated" in printed  # in the summary
    assert "UNABLE TO EVALUATE" in printed  # and its own section
    assert "needs 102 bars, has 57" in printed  # with the shortfall, not a verdict


def test_a_fully_evaluated_run_says_nothing_about_a_shortfall():
    """Both directions: the summary must not carry a caveat it did not earn."""
    from tradeflow.analytics.reporting import format_dry_run

    report = dryrun.dry_run_report(
        [_decision("no signal")],
        capital=CAPITAL,
        capital_source="from config",
        positions_source="flat",
        universe=["AAA"],
    )

    assert "could not be evaluated" not in format_dry_run(report)
    assert report["n_unable_to_evaluate"] == 0


def test_a_short_history_symbol_reaches_the_unevaluated_bucket_through_the_cli(monkeypatch, capsys):
    """End to end through `_run_dry`'s own loop, because the bucket tests above hand the
    report a ready-made list and so never exercise the code that decides *which* bucket a
    symbol lands in. A mutation counting shortfalls as WOULD SKIP passed every test until
    this one existed."""
    import pandas as pd

    from tradeflow import cli
    from tradeflow.utils.timeutils import NEW_YORK

    long_enough = pd.date_range("2024-01-02", periods=60, freq="D", tz=NEW_YORK)
    too_short = pd.date_range("2024-01-02", periods=3, freq="D", tz=NEW_YORK)

    def _frame(index):
        return pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
            index=index,
        )

    class _Data:
        def get_bars(self, symbols, start, end, **kwargs):
            return {s: _frame(too_short if s == "XYZ" else long_enough) for s in symbols}

    monkeypatch.setattr(cli, "build_data_and_broker", lambda *a, **k: pytest.fail("broker built"))
    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda **kw: _Data())
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA", "XYZ"])

    args = cli.build_parser().parse_args(
        [
            "live",
            "--dry-run",
            "--strategy",
            "demo_trend",
            "--symbols",
            "AAA,XYZ",
            "--capital",
            "8000",
            "--json",
        ]
    )
    args.flags_given = {"dry_run", "capital"}
    cli.cmd_live(args)

    report = json.loads(capsys.readouterr().out)

    assert [r["symbol"] for r in report["unable_to_evaluate"]] == ["XYZ"]
    assert "insufficient history" in report["unable_to_evaluate"][0]["reason"]
    # And it is *not* hiding among the decisions, which is the mutation this catches.
    assert "XYZ" not in [r["symbol"] for r in report["would_skip"]]
    assert report["n_evaluated"] == 1 and report["n_unable_to_evaluate"] == 1
