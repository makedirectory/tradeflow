"""The spread a cost model is configured with, and the cost stress that scales it.

`Trade.spread` defaulted to 5 bp and the engine never passed one, so
`ParametricCostModel(default_spread_bps=...)` was dead in `cost()`, and a stress that
scaled commission, impact and borrow but not the spread priced "3x" at about 1.7x.
"""

from datetime import datetime

import pandas as pd
import pytest

from quantred.costs import ParametricCostModel, Trade
from quantred.engine.backtest import BacktestEngine
from quantred.marketdata.client import MarketDataClient
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData, FakeMarketData
from tests.test_backtest import ScriptedStrategy


def _flat_round_trip(cost_model):
    """One 10-share round trip at $100 on flat bars with no ADV history, so the whole
    cost is commission plus half-spread: 2 legs x $1,000 x (commission + spread/2)."""
    index = pd.date_range("2024-01-02", periods=5, freq="D", tz=NEW_YORK)
    rows = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}] * 5
    frame = pd.DataFrame(rows, index=index)
    strategy = ScriptedStrategy([signals.BUY, signals.HOLD, signals.CLOSE_BUY, signals.HOLD, signals.HOLD], {})
    client = MarketDataClient(DictMarketData({"AAA": frame}))
    return BacktestEngine(strategy, client, cost_model=cost_model).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000.0
    )


def test_the_models_configured_spread_is_the_one_the_engine_charges():
    """A model built at 50 bp charged 5 bp, because the trade carried its own default."""
    result = _flat_round_trip(ParametricCostModel(default_spread_bps=50.0))

    per_leg = 1_000.0 * (1e-4 + 50e-4 / 2.0)
    assert result.total_cost == pytest.approx(2 * per_leg)
    assert result.total_cost > _flat_round_trip(ParametricCostModel(default_spread_bps=5.0)).total_cost


def test_a_trade_that_names_its_spread_still_overrides_the_model():
    """The other direction: dropping the default must not drop the override."""
    model = ParametricCostModel(default_spread_bps=50.0)
    named = model.cost(Trade("A", 10, 100.0, adv=0.0, daily_vol=0.0, spread=0.0002))
    unnamed = model.cost(Trade("A", 10, 100.0, adv=0.0, daily_vol=0.0))

    assert named.spread_cost == pytest.approx(1_000.0 * 0.0002 / 2)
    assert unnamed.spread_cost == pytest.approx(1_000.0 * 0.0050 / 2)


def test_a_cost_stress_of_three_triples_the_linear_cost():
    """With impact and borrow off every cost term is linear in the multiple, so the 3x
    point has to cost 3x (within the sizing drift of a slightly poorer book). It rose
    by about 1.7x while the spread stayed put."""
    from quantred.services import analysis

    symbols = ["S0", "S1", "S2", "S3"]
    client = MarketDataClient(FakeMarketData([*symbols, "SPY"], n=500, freq="1D"))
    report = analysis.run_cost_stress(
        client,
        "demo_trend",
        symbols,
        datetime(2024, 1, 2),
        datetime(2025, 3, 1),
        capital=100_000.0,
        impact_eta=0.0,
        borrow_bps=0.0,
        multiples=(1.0, 3.0),
    )
    one, three = (p["total_cost"] for p in report["points"])

    assert one > 0
    assert three / one == pytest.approx(3.0, rel=0.03)


def test_a_non_default_spread_is_part_of_a_trials_identity():
    from quantred.services.analysis import _cost_key

    default = _cost_key(False, 1.0, 0.3, 50.0)
    assert "spread_bps" not in default  # identities recorded before the knob keep their hash
    assert _cost_key(False, 1.0, 0.3, 50.0, spread_bps=15.0) != default


def test_the_cli_and_the_service_build_the_same_cost_model():
    """The CLI built its own and left out `participation_cap`; compared on every field
    the model carries, for a request that is not at the defaults."""
    from argparse import Namespace

    from quantred.cli import build_cost_model
    from quantred.services.analysis import build_cost_model as service_build

    args = Namespace(
        gross=False, commission_bps=2.5, impact_eta=0.4, borrow_bps=75.0, participation_cap=0.07
    )
    cli_model = build_cost_model(args)
    service_model = service_build(False, 2.5, 0.4, 0.07, 75.0)

    assert vars(cli_model) == pytest.approx(vars(service_model))
    assert cli_model.participation_cap == 0.07
    assert build_cost_model(Namespace(**{**vars(args), "gross": True})) is None


def _same_bar_fill_cost(volume, close):
    """Cost of a fill at bar 25's open (entered on the signal of bar 24, stopped out on
    the same bar), with bar 25's own volume and close set by the caller."""
    n = 40
    index = pd.date_range("2024-01-02", periods=n, freq="D", tz=NEW_YORK)
    closes = [100.0 + (i % 2) for i in range(n)]
    rows = [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1e6} for c in closes]
    rows[25] = {"open": 100.0, "high": 140.0, "low": 90.0, "close": close, "volume": volume}
    strategy = ScriptedStrategy(["HOLD"] * 24 + [signals.BUY] + ["HOLD"] * 15, {"stop_loss": 0.03})
    client = MarketDataClient(DictMarketData({"AAA": pd.DataFrame(rows, index=index)}))
    result = BacktestEngine(strategy, client, cost_model=ParametricCostModel()).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 3, 1), 100_000.0
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP_LOSS" and trade["entry_time"] == trade["exit_time"]
    return float(trade["cost"])


def test_a_fills_cost_does_not_depend_on_its_own_bars_volume_or_close():
    """ADV and volatility were rolling windows ending at the fill's own bar, but the fill
    is at that bar's open: a spike in the bar's own volume or close moved the cost of the same fill, and the
    docs called the windows trailing as-of. Both legs of this trade fill at bar 25."""
    quiet = _same_bar_fill_cost(volume=1e6, close=100.0)
    spiked = _same_bar_fill_cost(volume=5e7, close=130.0)

    assert quiet > 0
    assert spiked == pytest.approx(quiet, rel=1e-12)


def test_a_command_without_the_cap_flag_gets_the_models_default():
    from argparse import Namespace

    from quantred.cli import build_cost_model

    model = build_cost_model(Namespace(gross=False, commission_bps=1.0, impact_eta=0.3, borrow_bps=50.0))

    assert model.participation_cap == ParametricCostModel().participation_cap
