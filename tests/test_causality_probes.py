"""Each probe, against an engine deliberately broken in the way it exists to catch.

A probe asserted only against correct code is the one that passed for three days. The
leakage probe did exactly that: it ran on a candidate and cleared it while the engine
executed every signal one bar before it could have known. So every test here breaks
something on purpose first, and only then checks that the probe says so.

The reference defect is the accounting-v3 one, restored by making `_decided` return the
current bar's signal instead of the previous bar's — a one-bar look-ahead on every entry
and every signal exit, invisible to a feed shift because a shift moves signal and price
together.
"""

from datetime import datetime

import pandas as pd
import pytest

import tradeflow.engine.backtest as backtest_module
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.optimization.causality import (
    AS_OF,
    INTRA_BAR,
    CausalityProbes,
    probe_as_of_scanner,
    run_causality_probes,
)
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

_INDEX = pd.date_range("2024-01-02", periods=10, freq="D", tz=NEW_YORK)
_START, _END = datetime(2024, 1, 2), datetime(2024, 1, 20)


class _CloseDriven(Strategy):
    """Scores from this bar's close — the convention every shipped strategy uses."""

    PARAM_RANGES: dict = {}
    LONG_ONLY = True

    def calculate_required_lookback(self) -> int:
        return 1

    def initialize(self) -> None:
        pass

    def process_data(self, data):
        return data

    def calculate_scores(self, data):
        return pd.Series([1.0 if close > 150 else -1.0 for close in data["close"]], index=data.index)


def _factory(**limits):
    def build():
        return _CloseDriven(
            {
                "timeframe": "1Day",
                "stop_loss": 0.5,  # wide, so stops never pre-empt the signal under test
                "take_profit": 9.0,
                "risk_per_trade": 0.2,
                "position_limits": {
                    "max_positions": 2,
                    "max_position_size": 1e9,
                    "max_total_risk": 1e9,
                    **limits,
                },
            }
        )

    return build


def _bar(open_, close):
    return {
        "open": open_,
        "high": max(open_, close) * 1.02,
        "low": min(open_, close) * 0.98,
        "close": close,
        "volume": 100_000,
    }


def _frames():
    """Two names that cross up on different bars, plus a benchmark that always moves."""
    a = [(100, 100), (100, 200), (200, 200), (200, 200), (200, 100)] * 2
    b = [(100, 100), (100, 100), (100, 200), (200, 200), (200, 200)] * 2
    return {
        "AAA": pd.DataFrame([_bar(*p) for p in a], index=_INDEX),
        "BBB": pd.DataFrame([_bar(*p) for p in b], index=_INDEX),
        "SPY": pd.DataFrame([_bar(300, 300 + i * 3) for i in range(10)], index=_INDEX),
    }


@pytest.fixture
def one_bar_lookahead(monkeypatch):
    """The accounting-v3 defect, restored: act on this bar's signal at this bar's open."""
    monkeypatch.setattr(
        BacktestEngine,
        "_decided",
        staticmethod(lambda panel, i: (panel.sig[i], panel.score[i]) if i >= 0 else (signals.HOLD, 0.0)),
    )


def _probes(**kwargs):
    return CausalityProbes(_factory(), _frames(), ["AAA", "BBB"], _START, _END, benchmark="SPY", **kwargs)


# --- 1. the execution clock -------------------------------------------------------
def test_the_execution_clock_probe_catches_a_one_bar_look_ahead(one_bar_lookahead):
    """The defect that cost three days, and the whole reason this class of probe exists.
    Withholding a bar's close changes what that bar decided, which can only mean the
    decision was made with information that did not exist when the fill priced."""
    result = _probes().execution_clock()

    assert result.passed is False
    assert result.probe_class == INTRA_BAR
    assert "did not exist" in result.reason
    assert result.detail["findings"], "a failure must name the instant and the input"


def test_the_execution_clock_probe_clears_the_causal_engine():
    """Both directions. A probe nothing passes is indistinguishable from one that
    always fails, and it would be discovered only by the person it stopped."""
    result = _probes().execution_clock()

    assert result.passed is True
    assert result.detail["sampled"] > 0


def test_a_probe_that_had_nothing_to_look_at_does_not_report_a_pass():
    """`None`, never `True`. A run that never traded has not been cleared by a probe
    about trading, and rendering that as a pass is the reassurance this whole
    subsystem exists to withhold."""
    flat = {
        "AAA": pd.DataFrame([_bar(100, 100)] * 10, index=_INDEX),
        "SPY": pd.DataFrame([_bar(300, 300)] * 10, index=_INDEX),
    }
    probes = CausalityProbes(_factory(), flat, ["AAA"], _START, _END)

    result = probes.execution_clock()

    assert result.passed is None
    assert "no fills" in result.reason


def test_the_probe_reports_how_many_instants_it_actually_examined():
    """It samples. A probe that examined three of forty entries and said "passed"
    without saying so would be claiming far more than it checked."""
    result = _probes(sample=2).execution_clock()

    assert result.detail["sampled"] <= 2
    assert str(result.detail["sampled"]) in result.reason


# --- 2. same-bar ranking ----------------------------------------------------------
def test_the_ranking_probe_catches_an_ordering_that_reads_its_own_bar(one_bar_lookahead):
    """A second decision, distinct from the signal: when candidates outnumber slots,
    *which* of them is admitted is decided by ranking, and a ranking computed from the
    transacting bar's own score reintroduces the look-ahead every individual signal
    just lost."""
    result = _probes().same_bar_ranking()

    assert result.passed is False
    assert "ranked on the bar it transacted on" in result.reason


def test_the_ranking_probe_clears_a_causal_ordering():
    assert _probes().same_bar_ranking().passed is True


def test_the_ranking_probe_says_when_ordering_never_had_to_choose():
    """With slots to spare every candidate is admitted and the ordering decides
    nothing, so there is nothing here to have cleared."""
    probes = CausalityProbes(
        _factory(max_positions=50), _frames(), ["AAA", "BBB"], _START, _END, benchmark="SPY"
    )

    result = probes.same_bar_ranking()

    assert result.passed is None
    assert "free slots" in result.reason


# --- 3. benchmark alignment -------------------------------------------------------
def test_the_benchmark_probe_catches_a_series_paired_one_step_ahead(monkeypatch):
    """A benchmark shifted against the strategy's clock makes every alpha, beta and
    information ratio a measurement against a series the strategy was not running
    beside — and nothing about the numbers looks wrong."""
    original = backtest_module._benchmark_returns

    def shifted(closes, equity_times):
        series = original(closes, equity_times)
        return None if series is None else series.shift(-1)

    monkeypatch.setattr(backtest_module, "_benchmark_returns", shifted)

    result = _probes().benchmark_alignment()

    assert result.passed is False
    assert result.detail["steps_changed_before_the_perturbation"] > 0


def test_the_benchmark_probe_clears_an_aligned_series():
    result = _probes().benchmark_alignment()

    assert result.passed is True
    assert result.detail["steps_changed_before_the_perturbation"] == 0


def test_no_benchmark_means_the_probe_did_not_run_rather_than_passed():
    probes = CausalityProbes(_factory(), _frames(), ["AAA", "BBB"], _START, _END)

    result = probes.benchmark_alignment()

    assert result.passed is None
    assert "no benchmark" in result.reason


# --- 4. the as-of scanner clock ---------------------------------------------------
def _selection_frames():
    """Before the clock CCC leads; after it AAA runs away. A scanner reading the whole
    feed picks AAA, one honouring its clock picks CCC — so the two are distinguishable,
    which a fixture where they agree would not be."""

    def series(values):
        return pd.DataFrame(
            [{"open": v, "high": v, "low": v, "close": v, "volume": 1_000} for v in values], index=_INDEX
        )

    return {"AAA": series([10] * 4 + [500] * 6), "BBB": series([20] * 10), "CCC": series([30] * 4 + [31] * 6)}


def _rank_all(frames, as_of):
    """Ranks on the last bar on disk, whatever its stated clock says."""
    return sorted(frames, key=lambda s: -float(frames[s]["close"].iloc[-1]))[:1]


def _rank_as_of(frames, as_of):
    cut = pd.Timestamp(as_of).tz_localize(NEW_YORK)
    trimmed = {s: f.loc[f.index <= cut] for s, f in frames.items()}
    usable = [s for s, f in trimmed.items() if len(f)]
    return sorted(usable, key=lambda s: -float(trimmed[s]["close"].iloc[-1]))[:1]


def test_a_scanner_reading_past_its_clock_is_caught():
    """Universe selection is the quietest place hindsight hides: it applies to every
    symbol for the whole window, and a backtest over a universe chosen with hindsight
    produces results that look entirely ordinary."""
    result = probe_as_of_scanner(_rank_all, _selection_frames(), datetime(2024, 1, 5))

    assert result.passed is False
    assert result.probe_class == AS_OF
    assert result.detail["with_future_bars"] != result.detail["without_them"]


def test_a_scanner_honouring_its_clock_is_cleared():
    result = probe_as_of_scanner(_rank_as_of, _selection_frames(), datetime(2024, 1, 5))

    assert result.passed is True


def test_a_clock_with_no_bars_behind_it_is_not_a_pass():
    result = probe_as_of_scanner(_rank_as_of, _selection_frames(), datetime(2020, 1, 1))

    assert result.passed is None


def test_two_empty_selections_agreeing_is_not_a_pass():
    """They agree about nothing. A scanner that was never asked to choose has not been
    shown to honour any clock, and calling that a pass clears it for free."""
    result = probe_as_of_scanner(lambda frames, as_of: [], _selection_frames(), datetime(2024, 1, 5))

    assert result.passed is None
    assert "never exercised" in result.reason


# --- the suite --------------------------------------------------------------------
def test_the_suite_refuses_an_overall_pass_when_a_probe_never_ran():
    """Three passed and one never ran is not "passed". A partial run gets no verdict."""
    report = run_causality_probes(_factory(), _frames(), ["AAA", "BBB"], _START, _END, benchmark="SPY")

    assert report["verdict"] == "incomplete"
    assert report["not_exercised"] == ["as_of_scanner"]


def test_the_suite_is_causal_only_when_every_probe_ran_and_passed():
    report = run_causality_probes(
        _factory(),
        _frames(),
        ["AAA", "BBB"],
        _START,
        _END,
        benchmark="SPY",
        scan=_rank_as_of,
        scan_as_of=datetime(2024, 1, 5),
    )

    assert report["verdict"] == "causal"
    assert report["not_exercised"] == []


def test_the_suite_names_the_failures_rather_than_summarising_them_away(one_bar_lookahead):
    report = run_causality_probes(_factory(), _frames(), ["AAA", "BBB"], _START, _END, benchmark="SPY")

    assert report["verdict"] == "non-causal"
    assert set(report["failed"]) == {"execution_clock", "same_bar_ranking"}


def test_the_report_says_these_are_not_the_leakage_probe():
    """The single most expensive confusion in this project's history: a feed-shift probe
    passing was read as "causality checked". It cannot see a one-bar look-ahead at all,
    because a shift moves signal and price together."""
    report = run_causality_probes(_factory(), _frames(), ["AAA", "BBB"], _START, _END)

    assert "different" in report["note"]
    assert "leakage probe" in report["note"]


def test_a_feed_shift_does_not_notice_what_these_probes_catch(one_bar_lookahead):
    """The claim, demonstrated rather than asserted in prose. Under the same defect that
    fails two probes above, shifting the whole feed forward five bars still changes the
    results — which is exactly what the leakage probe reads as a pass.
    """
    frames = _frames()
    shifted = {}
    for symbol, frame in frames.items():
        moved = frame.copy()
        for column in ("open", "high", "low", "close", "volume"):
            moved[column] = moved[column].shift(-5)
        shifted[symbol] = moved.dropna()

    probes = CausalityProbes(_factory(), frames, ["AAA", "BBB"], _START, _END)
    base = probes._run(frames)
    moved_run = probes._run(shifted)

    # The leakage probe fails a run only when the shift leaves results *identical*.
    assert len(base.trades) != len(moved_run.trades) or not base.trades.equals(moved_run.trades)


# --- robustness of the perturbation itself ----------------------------------------
def test_integer_price_columns_do_not_break_the_probe():
    """A user's bars arrive with whatever dtype their source produced. pandas refuses a
    float into an int column rather than truncating it, so a probe that skipped the cast
    would raise on real data while passing every test written against float fixtures."""
    frames = {
        symbol: frame.astype({"open": "int64", "high": "int64", "low": "int64", "close": "int64"})
        for symbol, frame in _frames().items()
    }

    result = CausalityProbes(_factory(), frames, ["AAA", "BBB"], _START, _END).execution_clock()

    assert result.passed is not None


def test_a_flat_bar_offers_nothing_to_withhold_and_is_not_a_failure():
    """A bar whose high equals its low carries no post-open information, so there is
    nothing to take away — and inventing a perturbation outside the bar's range would
    move the high or low a stop legitimately reads."""
    from tradeflow.optimization.causality import _perturbed_close

    flat = pd.DataFrame([{"open": 5, "high": 5, "low": 5, "close": 5, "volume": 1}], index=_INDEX[:1])

    assert _perturbed_close(flat, _INDEX[0]) is None
