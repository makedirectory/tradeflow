"""Portfolio-level adverse and favourable excursion.

The equity curve is marked at each bar's close, so everything that happened inside the
bar is invisible to it. These tests pin the two bounds that puts around the truth — the
closing mark below, the simultaneous-extremes mark above — and the distinction between
them, which is the entire diagnostic.

Per-trade MAE cannot answer this question. A position deep underwater may be a small
part of the book, and one offsetting another may leave the book barely moved while both
trades look terrible.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytest

from tests.fakes import DictMarketData
from tradeflow.analytics.excursion import excursion_lines, portfolio_excursion
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

#: Wide stops on purpose. The point of these fixtures is a bar the book rides *through*
#: — a stop that fires closes the position, which bounds the excursion and measures the
#: stop rather than the diagnostic.
_CONFIG = {
    "timeframe": "1Day",
    "risk_per_trade": 1.0,
    "stop_loss": 0.50,
    "take_profit": 0.90,
    "position_limits": {"max_positions": 1, "max_position_size": 100_000.0, "max_total_risk": 1.0},
}


class _Scripted(Strategy):
    """Signals scripted per bar and applied to every symbol, so a book can be driven
    into several positions at once."""

    PARAM_RANGES: Dict = {}

    def __init__(self, per_bar_signals: List[str], overrides: Optional[Dict] = None):
        self._signals = per_bar_signals
        super().__init__({**_CONFIG, **(overrides or {})})

    def calculate_required_lookback(self) -> int:
        return 1

    def initialize(self) -> None:
        pass

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        return dict(zip(data.index, self._signals))


def _frame(rows: List[dict]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=len(rows), freq="D", tz=NEW_YORK)
    return pd.DataFrame(rows, index=index)


def _run(frames: Dict[str, List[dict]], per_bar_signals, *, overrides=None, capital=100_000.0):
    strategy = _Scripted(per_bar_signals, overrides)
    client = MarketDataClient(DictMarketData({s: _frame(r) for s, r in frames.items()}))
    return BacktestEngine(strategy, client).run(
        list(frames), datetime(2024, 1, 2), datetime(2024, 1, 20), capital
    )


# --- the unit, over samples the engine shapes ---------------------------------------
def _sample(close, adverse, favourable, *, time="t", gross=0.0, net=0.0, positions=1):
    return {
        "time": time,
        "equity_close": close,
        "equity_adverse": adverse,
        "equity_favourable": favourable,
        "gross": gross,
        "net": net,
        "open_positions": positions,
    }


def test_the_intra_bar_worst_is_reported_against_what_the_curve_showed():
    """The finding the whole diagnostic exists for: a shallow closing drawdown sitting
    over a bar the book spent much deeper underwater."""
    report = portfolio_excursion(
        [
            _sample(100_000, 100_000, 100_000),
            _sample(99_500, 88_000, 100_200, time="deep", gross=60_000, net=60_000, positions=3),
            _sample(99_800, 99_000, 100_000),
        ]
    )

    assert report["available"] is True
    assert report["max_adverse_excursion_pct"] == pytest.approx(12.0)
    assert report["closing_mark"]["max_drawdown_pct"] == pytest.approx(0.5)
    assert report["understatement_pct"] == pytest.approx(11.5)
    assert report["sampled_the_same_pain"] is False


def test_the_moment_carries_the_shape_of_the_book_at_it():
    """A depth with no context is unactionable — three positions at 60% gross is a
    different finding from one at 5%."""
    report = portfolio_excursion(
        [
            _sample(100_000, 100_000, 100_000),
            _sample(99_500, 88_000, 100_200, time="deep", gross=60_000, net=-20_000, positions=3),
        ]
    )

    at = report["adverse_at"]
    assert at["time"] == "deep"
    assert at["open_positions"] == 3
    assert at["gross_exposure_pct"] == pytest.approx(60_000 / 99_500 * 100.0)
    assert at["net_exposure_pct"] == pytest.approx(-20_000 / 99_500 * 100.0)
    assert at["drawdown_at_close_pct"] == pytest.approx(0.5)


def test_a_curve_that_did_sample_the_same_pain_says_so():
    """Both directions. A diagnostic that always reports a gap is not a diagnostic, and
    the disconfirming answer is the one this is most likely to be asked for."""
    report = portfolio_excursion(
        [
            _sample(100_000, 100_000, 100_000),
            _sample(97_000, 96_950, 100_050, positions=2),
            _sample(98_000, 97_900, 98_100, positions=2),
        ]
    )

    assert report["max_adverse_excursion_pct"] == pytest.approx(3.05)
    assert report["closing_mark"]["max_drawdown_pct"] == pytest.approx(3.0)
    assert report["sampled_the_same_pain"] is True


def test_a_run_that_opened_nothing_is_unavailable_not_a_zero_excursion():
    """ "The book never went underwater" and "there was no book" are different facts,
    and only one of them is about the strategy."""
    report = portfolio_excursion([])

    assert report["available"] is False
    assert "opened no positions" in report["reason"]
    assert "max_adverse_excursion_pct" not in report


def test_the_report_says_its_upper_bound_is_an_assumption():
    """Marking every position at its own worst tick assumes they all got there at once.
    They did not, and a number that does not say so will be quoted as a measurement."""
    report = portfolio_excursion([_sample(100_000, 90_000, 101_000)])

    assert "assumes they all got there at once" in report["basis"]
    printed = "\n".join(excursion_lines(report))
    assert "upper bound" in printed
    assert "lower bound" in printed


def test_the_favourable_side_is_measured_too():
    report = portfolio_excursion(
        [_sample(100_000, 100_000, 100_000), _sample(100_100, 99_000, 104_000, time="high")]
    )

    # Against the *running* peak of the closing curve (100_100 after bar two), not the
    # starting capital — a drawdown measured from a level the book never printed would
    # carry the same intra-bar noise this diagnostic exists to separate out.
    assert report["max_favourable_excursion_pct"] == pytest.approx((104_000 - 100_100) / 100_100 * 100)
    assert report["favourable_at"]["time"] == "high"


# --- the engine actually produces those samples -------------------------------------
def test_the_engine_records_the_book_at_its_worst_tick_inside_a_bar():
    """End to end. The middle bar dives and recovers by its close, so the closing curve
    barely moves while the book was materially underwater inside it."""
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 101, "low": 70, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
    ]
    result = _run({"AAA": rows}, [signals.BUY, signals.HOLD, signals.HOLD])

    report = result.excursion
    assert report["available"] is True
    assert report["max_adverse_excursion_pct"] > report["closing_mark"]["max_drawdown_pct"]
    assert report["sampled_the_same_pain"] is False
    assert report["adverse_at"]["open_positions"] == 1


def test_a_book_whose_closes_track_its_lows_shows_no_gap():
    """The boundary the check must not reject: an honest curve."""
    rows = [
        {"open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1_000_000},
    ]
    result = _run({"AAA": rows}, [signals.BUY, signals.HOLD, signals.HOLD])

    assert result.excursion["available"] is True
    assert result.excursion["sampled_the_same_pain"] is True


def test_two_positions_underwater_in_one_bar_aggregate_at_the_book_level():
    """The question per-trade MAE cannot answer. Each position's own excursion is the
    same in both runs; what changes is how much of the book was carrying it."""
    deep = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 101, "low": 80, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
    ]
    one = _run(
        {"AAA": deep},
        [signals.BUY, signals.HOLD, signals.HOLD],
        overrides={"risk_per_trade": 0.4},
    )
    two = _run(
        {"AAA": deep, "BBB": deep},
        [signals.BUY, signals.HOLD, signals.HOLD],
        overrides={
            # Half the risk each, so both fit: at the default the first position takes
            # the whole budget and the second is refused, which would make this a test
            # about the risk cap rather than about aggregation.
            "risk_per_trade": 0.4,
            "position_limits": {
                "max_positions": 2,
                "max_position_size": 100_000.0,
                "max_total_risk": 1.0,
            },
        },
    )

    assert two.excursion["adverse_at"]["open_positions"] == 2
    assert two.excursion["max_adverse_excursion_pct"] > one.excursion["max_adverse_excursion_pct"]


def test_marking_at_a_price_override_agrees_with_the_book_s_own_mark():
    """`equity_at` exists so a diagnostic can ask what the book was worth at some other
    price without a second copy of the sum. With no override it must be the same
    number, or the diagnostic and the engine are measuring different books."""
    from tradeflow.engine.backtest import _Book

    book = _Book(cash=50_000.0)
    book.positions["AAA"] = {
        "side": signals.BUY,
        "size": 100,
        "entry_price": 100.0,
        "last_price": 110.0,
        "notional": 10_000.0,
        "risk": 0.0,
    }

    assert book.equity_at() == book.equity()
    assert book.equity_at({}) == book.equity()
    assert book.equity_at({"AAA": 90.0}) == pytest.approx(book.equity() - 2_000.0)


def test_the_excursion_diagnostic_changes_no_metric():
    """It is reported and never consulted. If adding it had moved a metric it would
    need an accounting bump, and every recorded trial would become a different era."""
    rows = [
        {"open": 100, "high": 101, "low": 70, "close": 100, "volume": 1_000_000},
        {"open": 100, "high": 105, "low": 70, "close": 104, "volume": 1_000_000},
        {"open": 108, "high": 112, "low": 70, "close": 111, "volume": 1_000_000},
    ]
    result = _run({"AAA": rows}, [signals.BUY, signals.HOLD, signals.HOLD])

    # The curve is still the end-of-bar mark-to-market it has always been: no step of
    # it is the intra-bar figure the excursion report is built from.
    assert result.excursion["max_adverse_excursion_pct"] > 0
    assert min(result.equity_curve) > result.initial_capital * (
        1 - result.excursion["max_adverse_excursion_pct"] / 100.0
    )
