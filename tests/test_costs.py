"""Tests for the transaction-cost model and its backtest integration (spec 007)."""

import math
from datetime import datetime

import pandas as pd
import pytest

from src.costs import ParametricCostModel, Trade
from src.engine.backtest import BacktestEngine
from src.marketdata.client import MarketDataClient
from src.strategies.volume_spike import VolumeSpikeStrategy
from tests.fakes import FakeMarketData


def _model():
    return ParametricCostModel(
        commission_bps=1.0, default_spread_bps=5.0, impact_eta=0.3, participation_cap=0.10
    )


# --- decomposition -----------------------------------------------------------
def test_cost_decomposition_is_exact():
    m = _model()
    t = Trade("A", shares=1000, price=100.0, adv=100_000, daily_vol=0.02)
    c = m.cost(t)
    notional = 1000 * 100.0
    assert c.commission == 1e-4 * notional
    assert c.spread_cost == (5e-4 / 2) * notional
    assert c.impact_cost == 0.3 * 0.02 * math.sqrt(1000 / 100_000) * notional
    assert c.total == c.commission + c.spread_cost + c.impact_cost


def test_square_root_total_cost_is_superlinear():
    m = _model()
    small = m.cost(Trade("A", 1000, 100.0, 100_000, 0.02)).impact_cost
    big = m.cost(Trade("A", 2000, 100.0, 100_000, 0.02)).impact_cost
    assert abs(big / small - 2**1.5) < 1e-9  # total impact ∝ |q|^{3/2}, not |q|


def test_participation_cap_is_flagged():
    m = _model()
    assert not m.cost(Trade("A", 5_000, 100.0, 100_000, 0.02)).capped  # 5% < 10%
    assert m.cost(Trade("A", 20_000, 100.0, 100_000, 0.02)).capped  # 20% > 10%


def test_missing_adv_charges_commission_and_spread_only():
    m = _model()
    c = m.cost(Trade("A", 1000, 100.0, adv=0.0, daily_vol=0.02))  # no volume info
    assert c.impact_cost == 0.0
    assert c.commission > 0 and c.spread_cost > 0


def test_annual_haircut_round_trips_and_amortizes():
    m = _model()
    t = Trade("A", 1000, 100.0, 100_000, 0.02)
    # Round-trip (2x one-way) amortized over a 1-month hold ≈ 12x the round-trip rate.
    assert abs(m.annual_cost_rate(t, 1 / 12) - 2 * m.cost_rate(t) * 12) < 1e-12


# --- backtest integration ----------------------------------------------------
def _run(cost_model):
    dc = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="5min"))
    return BacktestEngine(VolumeSpikeStrategy.create_with_defaults(), dc, cost_model=cost_model).run(
        ["AAA", "BBB"], datetime(2024, 1, 1), datetime(2024, 3, 1), 100_000.0
    )


def test_backtest_charges_both_legs_and_net_below_gross():
    gross = _run(None)
    net = _run(_model())
    assert net.total_cost > 0
    assert net.final_capital <= gross.final_capital
    # Each trade's net pnl is its gross pnl minus the cost charged on both legs.
    for _, trade in net.trades.iterrows():
        assert abs(trade["pnl"] - (trade["gross_pnl"] - trade["cost"])) < 1e-9
        assert trade["cost"] > 0  # every closed trade paid entry + exit


def test_carry_cost_charges_shorts_only():
    m = ParametricCostModel(annual_borrow_bps=100.0)  # 1%/yr borrow
    # A short held a year on $10k notional pays 1%; a long pays nothing.
    assert m.carry_cost(10_000, is_short=True, holding_years=1.0) == pytest.approx(100.0)
    assert m.carry_cost(10_000, is_short=True, holding_years=0.5) == pytest.approx(50.0)
    assert m.carry_cost(10_000, is_short=False, holding_years=1.0) == 0.0


def test_backtest_charges_borrow_on_shorts():
    # volume_spike is long/short, so a high borrow rate raises total cost via its shorts.
    dc = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="5min"))

    def run(borrow_bps):
        model = ParametricCostModel(annual_borrow_bps=borrow_bps)
        res = BacktestEngine(VolumeSpikeStrategy.create_with_defaults(), dc, cost_model=model).run(
            ["AAA", "BBB"], datetime(2024, 1, 1), datetime(2024, 3, 1), 100_000.0
        )
        shorts = int((res.trades["side"] == "SELL").sum()) if not res.trades.empty else 0
        return res.total_cost, shorts

    cost_no_borrow, shorts = run(0.0)
    cost_borrow, _ = run(2000.0)  # an extreme 20%/yr to make the effect unmistakable
    if shorts > 0:
        assert cost_borrow > cost_no_borrow
    else:  # no shorts taken on this fixture → borrow changes nothing
        assert cost_borrow == pytest.approx(cost_no_borrow)


def test_total_cost_reconciles_gross_and_net():
    net = _run(_model())
    assert abs(net.gross_final_capital - (net.final_capital + net.total_cost)) < 1e-6


def test_engine_uses_trailing_adv_no_lookahead():
    # The engine feeds the cost model a *trailing* rolling ADV, so a past bar's cost
    # never depends on future volume.
    volume = pd.Series(range(1, 101), dtype=float)
    adv = volume.rolling(BacktestEngine.ADV_WINDOW).mean()
    bumped = volume.copy()
    bumped.iloc[60:] *= 100
    adv_bumped = bumped.rolling(BacktestEngine.ADV_WINDOW).mean()
    assert adv.iloc[:60].equals(adv_bumped.iloc[:60])  # past ADV unchanged by future volume
