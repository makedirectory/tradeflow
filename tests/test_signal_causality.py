"""A signal may only be acted on after the bar that produced it.

Through accounting v3 the backtest executed a signal derived from bar `i`'s close
against bar `i`'s *open* — for entries and signal exits alike. That is a one-bar
look-ahead applied to every trade in every recorded result, and it is invisible to the
leakage probe: a feed shift moves signal and price together, so the relationship
survives the shift intact.

It also made the backtest structurally impossible to match live, where a closed bar
arrives, a signal is emitted, and a market order fills strictly afterwards. These pin
the property on both clocks, because the two implement it separately and nothing in the
codebase connects them.
"""

from datetime import datetime

import pandas as pd
import pytest

from tests.fakes import DictMarketData, RecordingBroker
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

#: Bar 1 opens at 100 and closes at 200. A score that flips only above a close of 150
#: is therefore knowable at bar 1's close and at no earlier moment, so a fill at bar 1's
#: open would be a price that existed hours before the information did.
_REVEALS_AT_CLOSE = [
    {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000},
    {"open": 100, "high": 200, "low": 100, "close": 200, "volume": 1_000},
    {"open": 200, "high": 200, "low": 200, "close": 200, "volume": 1_000},
    {"open": 200, "high": 200, "low": 200, "close": 100, "volume": 1_000},
    {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000},
]


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


def _run(rows=None):
    frames = {
        "AAA": pd.DataFrame(
            rows or _REVEALS_AT_CLOSE,
            index=pd.date_range("2024-01-02", periods=len(rows or _REVEALS_AT_CLOSE), freq="D", tz=NEW_YORK),
        )
    }
    strategy = _CloseDriven(
        {
            "timeframe": "1Day",
            "stop_loss": 0.9,  # wide, so stops never pre-empt the signal being tested
            "take_profit": 9.0,
            "risk_per_trade": 0.9,
            "position_limits": {"max_positions": 1, "max_position_size": 100_000.0, "max_total_risk": 1.0},
        }
    )
    return BacktestEngine(strategy, MarketDataClient(DictMarketData(frames))).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 20), 100_000
    )


# --- research clock ---------------------------------------------------------------
def test_an_entry_cannot_fill_at_the_open_of_the_bar_that_produced_it():
    """The defect, at full size: the engine filled at 100 on a bar that closed at 200,
    capturing an intrabar move it could only have learned about at the close."""
    trades = _run().trades

    assert len(trades) == 1
    assert trades.iloc[0]["entry_price"] == 200.0, "filled before the signal existed"


def test_a_signal_exit_obeys_the_same_rule_as_an_entry():
    """Exits were the other half of it, and a fix that moved only entries would leave
    the book exiting on information it did not have."""
    trades = _run().trades

    # The score falls back below its threshold on bar 3 (close 100), so the exit is
    # actionable at bar 4's open — not at bar 3's, which was 200.
    assert trades.iloc[0]["exit_price"] == 100.0


def test_the_first_bar_can_never_trade():
    """There is no prior bar to have decided on, so bar 0 cannot transact — its own
    signal being actionable on itself is the same look-ahead in its smallest form.

    Bar 1 acting on bar 0's signal is correct and expected; the assertion is about
    *which* bar fills, not about whether anything does.
    """
    immediate = [{"open": 100, "high": 100, "low": 100, "close": 200, "volume": 1_000}] * 3
    index = pd.date_range("2024-01-02", periods=3, freq="D", tz=NEW_YORK)

    trades = _run(immediate).trades

    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == index[1]  # not index[0]


def test_a_flat_series_still_produces_no_trades():
    """Both directions: the lag must not manufacture activity where there is none."""
    flat = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000}] * 5

    assert len(_run(flat).trades) == 0


# --- trade clock ------------------------------------------------------------------
def test_live_acts_on_a_closed_bar_at_that_bar_s_close():
    """Live has always been causal, and this states why so a change cannot quietly
    break the parity the backtest was just brought into.

    A bar arrives already closed; the signal it produces is handed to execution with
    that bar's close as the reference price, and the resulting order fills strictly
    afterwards. The open of that bar is never a price live can transact at.
    """
    import asyncio

    from tradeflow.engine.live import LiveEngine
    from tradeflow.execution.live_trader import LiveTrader
    from tradeflow.marketdata.base import BarEvent
    from tradeflow.services.registry import STRATEGIES

    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    strategy.process_bar = lambda symbol, bar, ts: "BUY"
    trader = LiveTrader(RecordingBroker(), strategy, respect_market_hours=False)
    seen = {}

    def capture(symbol, signal, price, bar_timestamp=None):
        from tradeflow.execution import decision as decisions

        seen["price"] = price
        return decisions.decline(symbol, signal, "captured", ())

    trader.handle_signal = capture
    engine = LiveEngine(strategy, MarketDataClient(DictMarketData({})), trader, reconcile_every=0)
    event = BarEvent(
        symbol="AAA",
        timestamp=datetime(2024, 1, 2, 16, 0, tzinfo=NEW_YORK),
        open=100.0,
        high=200.0,
        low=100.0,
        close=200.0,
        volume=1_000,
    )

    asyncio.run(engine._on_bar(event))

    assert seen["price"] == pytest.approx(event.close)
    assert seen["price"] != event.open, "live would be transacting before its own signal"


# --- the book's value at a decision is the book's value at that decision ----------
def _cap_run(other_close, cap):
    """Two names. AAA is already open on bar 4; BBB tries to enter at bar 4's open.
    Only AAA's *close* on bar 4 differs between the two runs — information that does
    not exist at the instant BBB's fill prices."""
    index = pd.date_range("2024-01-02", periods=6, freq="D", tz=NEW_YORK)

    def bar(open_, close):
        return {
            "open": open_,
            "high": max(open_, close),
            "low": min(open_, close),
            "close": close,
            "volume": 100_000,
        }

    held = [bar(200, 200)] * 6
    held[4] = bar(200, other_close)
    entering = [bar(100, 100)] * 3 + [bar(100, 200), bar(200, 200), bar(200, 200)]

    strategy = _CloseDriven(
        {
            "timeframe": "1Day",
            "stop_loss": 0.5,  # wide: no stop fires in either run
            "take_profit": 9.0,
            "risk_per_trade": 0.2,
            "position_limits": {
                "max_positions": 2,
                "max_position_size": 1e9,
                "max_total_risk": 1e9,
                "max_gross_exposure": cap,
            },
        }
    )
    frames = {
        "AAA": pd.DataFrame(held, index=index),
        "BBB": pd.DataFrame(entering, index=index),
    }
    trades = (
        BacktestEngine(strategy, MarketDataClient(DictMarketData(frames)))
        .run(["AAA", "BBB"], datetime(2024, 1, 2), datetime(2024, 1, 20), 100_000)
        .trades
    )
    at_bar_4 = trades[trades["entry_time"] == index[4]] if len(trades) else trades
    return sorted(set(at_bar_4["symbol"])) if len(at_bar_4) else []


@pytest.mark.parametrize("cap", [0.6, 0.65, 0.7])
def test_an_entry_is_not_admitted_or_refused_by_the_close_of_the_bar_it_fills_at(cap):
    """The same defect as the one above, one layer down, and equally invisible to a
    feed shift: it was in the admission gate rather than the signal.

    The book was marked to the bar's close before entries were considered, and the
    exposure caps are tested against `equity * cap` — so whether an entry filling at
    that bar's *open* was admitted could depend on where the bar finished hours later.
    At these caps it flipped outright: a position that was admitted when another symbol
    closed down was refused when it closed up.

    Nothing about the two runs differs except one close on one bar, and no stop or
    take-profit fires in either, so a difference in what was admitted can only have come
    from the mark.
    """
    assert _cap_run(150, cap) == _cap_run(400, cap)


def test_the_cap_still_binds_on_information_that_did_exist():
    """Both directions. A cap that stopped binding altogether would pass the test above
    trivially, and would be a worse bug than the one it replaced."""
    assert _cap_run(200, 0.4) == []  # tight: the book cannot fund the second name
    assert _cap_run(200, 0.9) == ["BBB"]  # loose: it can


def test_the_equity_curve_is_still_marked_at_the_close():
    """The decisions move to the open; the curve must not. An equity curve marked at
    each bar's open would report a different volatility and drawdown for every result
    ever recorded, which is a far larger change than the one being made."""
    rows = [
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1_000},
        {"open": 100, "high": 300, "low": 100, "close": 300, "volume": 1_000},
        {"open": 300, "high": 400, "low": 300, "close": 400, "volume": 1_000},
        {"open": 400, "high": 400, "low": 400, "close": 400, "volume": 1_000},
        {"open": 400, "high": 400, "low": 400, "close": 400, "volume": 1_000},
    ]
    result = _run(rows)

    # 333 shares open at bar 2's open of 300, and bar 2 closes at 400: a close-marked
    # step is worth 100,000 + 333 x 100. Marked at the open it would still read
    # 100,000 — no gain at all on the step the position was taken.
    assert result.equity_curve[0] == 100_000
    assert result.equity_curve[3] == pytest.approx(133_300.0)
