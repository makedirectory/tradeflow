"""Cross-strategy contract tests: every registered strategy must behave.

Parametrized over the registry, so adding a strategy automatically inherits this
coverage - the architecture's "a strategy is one file" promise, enforced.
"""

from datetime import datetime

import pytest

from tests.fakes import FakeMarketData, make_ohlcv
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals
from tradeflow.strategies.ma_crossover import MovingAverageCrossoverStrategy
from tradeflow.strategies.mean_reversion import MeanReversionStrategy

_VALID_SIGNALS = {signals.BUY, signals.SELL, signals.CLOSE_BUY, signals.CLOSE_SELL, signals.HOLD}


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_create_with_defaults_sets_timeframe_and_lookback(name):
    strategy = STRATEGIES[name].create_with_defaults()
    assert strategy.config["timeframe"]
    assert strategy.config["required_lookback_periods"] >= 1


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_generate_signals_uses_only_the_shared_vocabulary(name):
    strategy = STRATEGIES[name].create_with_defaults()
    strategy.initialize()
    out = strategy.generate_signals(strategy.process_data(make_ohlcv(n=300)))
    assert set(out.values()) <= _VALID_SIGNALS


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_backtest_runs_end_to_end(name):
    strategy = STRATEGIES[name].create_with_defaults()
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400))
    result = BacktestEngine(strategy, client).run(
        ["AAA", "BBB"], datetime(2024, 1, 1), datetime(2024, 3, 1), 100_000.0
    )
    assert result.metrics
    assert result.final_capital > 0


def test_ma_crossover_requires_fast_below_slow():
    config = {p: spec["default"] for p, spec in MovingAverageCrossoverStrategy.PARAM_RANGES.items()}
    config["fast_ema_period"] = config["slow_ema_period"]
    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(config).initialize()


def test_mean_reversion_requires_oversold_below_overbought():
    config = {p: spec["default"] for p, spec in MeanReversionStrategy.PARAM_RANGES.items()}
    config["oversold"] = config["overbought"]
    with pytest.raises(ValueError):
        MeanReversionStrategy(config).initialize()
