"""The screen's promises, checked against a real sweep rather than its docstring.

Three of them carry the whole feature. It must journal nothing, because a researcher
who cannot ask a cheap question without permanently raising the deflation bar for the
family will either ask carelessly or stop asking. It must fetch once and evaluate many,
because the thing it replaces is a shell loop that refetched every time. And confirming
must be able to record exactly one point, because a confirm that took a set would be a
screen that journals, with the budget problem back in through the door it came out of.
"""

from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis

_SYMBOLS = [f"S{i}" for i in range(5)]
_START, _END = datetime(2024, 1, 2), datetime(2024, 12, 31)


class _CountingFeed(FakeMarketData):
    """Counts how many times the *source* was asked for bars."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetches = 0

    def get_bars(self, *args, **kwargs):
        self.fetches += 1
        return super().get_bars(*args, **kwargs)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A journal of this test's own, redirected the way the rest of the suite does it.

    Setting `TRADEFLOW_HOME` alone is not enough and the reason is worth knowing: once
    any test in the session has overridden `audit.DEFAULT_TRIAL_JOURNAL`, restoring it
    binds a resolved path into the module, and a later change of root no longer
    reaches the writer. Redirecting the constant is what the suite already does, so
    this follows it rather than inventing a second way that works in isolation and not
    in a run.
    """
    from tradeflow.services import audit
    from tradeflow.store import trials

    path = tmp_path / "research_journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", path)
    monkeypatch.setattr(trials, "DEFAULT_JOURNAL_PATH", path)
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    return path


def _feed():
    return _CountingFeed([*_SYMBOLS, "SPY"], n=300, freq="1D")


def _screen(feed=None, **kwargs):
    feed = feed or _feed()
    client = MarketDataClient(feed)
    defaults = dict(method="random", max_evals=8, seed=3)
    return analysis.run_screen(client, "demo_trend", _SYMBOLS, _START, _END, **{**defaults, **kwargs}), feed


# --- it must not spend statistical budget ----------------------------------------
def test_a_screen_writes_no_trial(journal):
    """The whole point. Every journaled trial raises the deflated-Sharpe bar for its
    family permanently, so a broad sweep run to answer "is there anything here
    at all" would make the eventual real candidate harder to promote than it should be."""
    report, _ = _screen()

    assert report["journaled"] is False
    assert not journal.exists() or journal.read_text() == ""


def test_a_screen_says_it_journaled_nothing_rather_than_leaving_it_to_be_inferred(journal):
    """Silence about cost reads as cost. An agent that assumes every call spends a
    trial will ration the cheap question exactly as hard as the expensive one."""
    report, _ = _screen()

    assert "journal" in report["note"].lower()
    assert "promotable" in report["note"].lower()


def test_a_screen_is_not_served_from_the_trial_store(journal):
    """Not memoized either. A sweep where some points came from recorded evidence and
    others were simulated is neither reconnaissance nor evidence, and the honest label
    for it does not exist."""
    report, _ = _screen()
    again, _ = _screen()

    assert "memoized" not in report
    assert report["distribution"]["n_finite"] == again["distribution"]["n_finite"]


# --- one process, one data fetch, N evaluations -----------------------------------
def test_the_window_is_fetched_once_however_many_points_are_evaluated(journal):
    """What it replaces: a shell loop over N invocations with temporary configs, each
    refetching the same window."""
    report, feed = _screen(max_evals=8)

    assert report["searched"]["evaluated"] == 8
    assert feed.fetches == 1


# --- the summary leads with the distribution --------------------------------------
def test_the_report_carries_a_distribution_and_a_null_beside_any_best_point(journal):
    report, _ = _screen()

    assert report["distribution"]["n_finite"] > 0
    assert report["noise_baseline"]["n_draws"] == report["distribution"]["n_finite"]
    assert report["best_point"] is not None
    # The best point never travels without the number that says what best-of-N is worth.
    assert report["noise_baseline"]["observed_best"] == pytest.approx(report["distribution"]["max"])


def test_a_grid_larger_than_the_budget_says_it_was_sampled(journal):
    """Silence about what was dropped reads as "everything ran". A cap that quietly
    samples 50 points from 358,400 and reports a distribution is describing a sweep
    nobody performed."""
    report, _ = _screen(method="grid", max_evals=6)

    assert report["searched"]["sampled_from_grid"] is True
    assert report["searched"]["grid_size"] > report["searched"]["requested"]


def test_a_grid_that_fits_the_budget_does_not_claim_to_have_been_sampled(journal):
    """Both directions."""
    report, _ = _screen(
        method="grid",
        max_evals=500,
        param_ranges={
            "fast_ema_period": {"min": 5, "max": 6, "step": 1},
            "slow_ema_period": {"min": 21, "max": 22, "step": 1},
            "risk_per_trade": {"min": 0.02, "max": 0.02, "step": 0.01},
            "stop_loss": {"min": 0.02, "max": 0.02, "step": 0.01},
            "take_profit": {"min": 0.04, "max": 0.04, "step": 0.02},
        },
    )

    assert report["searched"]["sampled_from_grid"] is False
    assert report["searched"]["evaluated"] == report["searched"]["grid_size"] == 4


# --- narrowing is refused loudly when it is wrong ---------------------------------
def test_narrowing_a_parameter_the_strategy_does_not_declare_is_refused(journal):
    """A typo that silently screened the full range would report a distribution for a
    space the user never asked about, and finding nothing there means nothing."""
    with pytest.raises(ValueError, match="no parameter"):
        _screen(param_ranges={"lookback_period": {"min": 5, "max": 10, "step": 1}})


def test_narrowing_only_one_bound_keeps_the_rest_of_the_declaration(journal):
    """An override supplies what it names and inherits the rest, so narrowing an axis
    cannot silently change a parameter's type or drop its default."""
    from tradeflow.demo.strategies import DemoTrendStrategy

    ranges = analysis.screen_ranges(DemoTrendStrategy, {"fast_ema_period": {"max": 9}})

    assert ranges["fast_ema_period"]["max"] == 9
    assert ranges["fast_ema_period"]["min"] == DemoTrendStrategy.PARAM_RANGES["fast_ema_period"]["min"]
    assert (
        ranges["fast_ema_period"]["default"] == DemoTrendStrategy.PARAM_RANGES["fast_ema_period"]["default"]
    )


def test_an_empty_narrowed_range_is_refused_rather_than_screened(journal):
    from tradeflow.demo.strategies import DemoTrendStrategy

    with pytest.raises(ValueError, match="empty"):
        analysis.screen_ranges(DemoTrendStrategy, {"fast_ema_period": {"min": 15, "max": 6}})


# --- the book the points were screened at -----------------------------------------
def test_a_screen_evaluates_the_book_it_was_given(journal):
    """`position_limits` is not a tunable parameter, so a sweep built from params alone
    drops it and screens a one-position book while reporting on an eight-position one."""
    one, _ = _screen(position_limits={"max_positions": 1}, seed=9)
    many, _ = _screen(position_limits={"max_positions": 5}, seed=9)

    assert one["position_limits"] == {"max_positions": 1}
    assert one["distribution"] != many["distribution"]


# --- confirming is exactly one point ----------------------------------------------
def test_confirming_a_point_records_exactly_one_trial(journal):
    report, feed = _screen()
    params = {k: report["best_point"][k] for k in report["searched"]["parameters"]}

    confirmed = analysis.confirm_screen_point(
        MarketDataClient(feed), "demo_trend", _SYMBOLS, _START, _END, params
    )

    assert confirmed["journaled"] is True
    assert len([line for line in journal.read_text().splitlines() if line.strip()]) == 1


def test_a_confirmed_point_is_the_same_trial_a_backtest_of_it_would_be(journal):
    """It delegates rather than journaling its own way. A second definition of what a
    trial is would split the campaign's count in two — a confirmed point and the same
    backtest run directly would stop deduping against each other."""
    report, feed = _screen()
    params = {k: report["best_point"][k] for k in report["searched"]["parameters"]}
    client = MarketDataClient(feed)

    analysis.confirm_screen_point(client, "demo_trend", _SYMBOLS, _START, _END, params)
    repeat = analysis.run_backtest(client, "demo_trend", _SYMBOLS, _START, _END, config=dict(params))

    # Served from the trial the confirm wrote: the two computed the same dedup identity,
    # which is the only thing that makes them one trial rather than two.
    assert repeat.get("memoized") is True
    assert repeat["trial_id"]

    # Both directions: a different point is not answered by the confirmed one.
    other = analysis.run_backtest(
        client,
        "demo_trend",
        _SYMBOLS,
        _START,
        _END,
        config={**params, "stop_loss": round(float(params["stop_loss"]) + 0.01, 4)},
    )
    assert not other.get("memoized")


def test_the_cli_can_only_confirm_a_point_the_screen_actually_evaluated(journal):
    """`--confirm` reads its point back out of the screen's own rows. There is no way
    to spell a set, and no way to confirm something that was never scored."""
    from tradeflow.cli import _screen_confirm_target

    report, _ = _screen()

    assert _screen_confirm_target(report, "best") is not None
    assert _screen_confirm_target(report, "1") is not None
    assert _screen_confirm_target(report, "9999") is None


def test_confirming_the_best_names_the_same_point_the_report_did(journal):
    from tradeflow.cli import _screen_confirm_target

    report, _ = _screen()

    chosen = _screen_confirm_target(report, "best")
    assert chosen == {k: report["best_point"][k] for k in report["searched"]["parameters"]}
