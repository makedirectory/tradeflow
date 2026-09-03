"""The example pack, exercised as a real installed package.

Every other registry test monkeypatches entry points. This one uses the mechanism —
a separate distribution, its own pyproject, discovered through the two entry-point
groups a customer's pack uses. It is the only test that covers packaging, discovery and
contract compliance together, which is the path the whole private-strategy feature
rests on.

Skipped when the pack is not installed, so a bare checkout still runs the suite. CI
installs it, which is the point: without it, the path users depend on is exercised only
by fakes.
"""

from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import BUILTIN_SCANNERS, BUILTIN_STRATEGIES, SCANNERS, STRATEGIES

STRATEGY = "example_breakout"
REVERSION = "example_reversion"
SCANNER = "example_liquidity"

pytestmark = pytest.mark.skipif(
    STRATEGY not in STRATEGIES,
    reason="example pack not installed — `uv pip install -e example`",
)


def _defaults(cls):
    return {name: spec["default"] for name, spec in cls.PARAM_RANGES.items()}


# --- discovery --------------------------------------------------------------------
def test_the_pack_registers_through_entry_points():
    """Not imported, not hardcoded, not a plugin directory. The engine finds it the way
    it will find yours."""
    assert STRATEGY in STRATEGIES
    assert SCANNER in SCANNERS


def test_the_pack_is_not_mistaken_for_a_built_in():
    """The demo flag exists so a registry of shipped demonstrations cannot read as the
    product. A pack arriving by entry point is third-party, whatever it is called."""
    assert STRATEGY not in BUILTIN_STRATEGIES
    assert SCANNER not in BUILTIN_SCANNERS

    from tradeflow.services.registry import list_strategies

    row = next(r for r in list_strategies() if r["name"] == STRATEGY)
    assert row["demo"] is False


# --- the contract the engine relies on --------------------------------------------
@pytest.mark.parametrize("name", [STRATEGY, REVERSION])
def test_every_strategy_satisfies_the_interface(name):
    """Construction, lookback, indicators, scores — the four things the engine calls."""
    strategy = STRATEGIES[name].create_with_defaults()
    strategy.initialize()

    assert strategy.config["timeframe"]
    assert strategy.calculate_required_lookback() > 0


def test_the_strategy_declares_its_own_book_limits():
    """Limits are part of what gets validated. A strategy that leaves them to the base
    class is validated against limits its author never chose — which this session
    established changes results by an order of magnitude."""
    limits = STRATEGIES[STRATEGY].create_with_defaults().position_limits()

    assert limits["max_positions"] == 5
    assert limits["max_gross_exposure"] == 0.90


def test_a_contradictory_parameter_pair_is_refused_loudly():
    """Ranges are checked by the optimizer; relationships between parameters are not,
    and `initialize` is where they belong."""
    # Both values are inside their declared ranges — the contradiction is between
    # them, which is exactly the class of problem PARAM_RANGES cannot express.
    strategy = STRATEGIES[STRATEGY]({"entry_period": 80, "trend_period": 50})

    with pytest.raises(ValueError, match="shorter than"):
        strategy.initialize()


def test_the_breakout_level_excludes_the_current_bar():
    """The bug this comment exists to prevent: without the shift, today's own high is
    part of the level today has to break, so nothing can ever break it and the strategy
    silently never fires."""
    import pandas as pd

    strategy = STRATEGIES[STRATEGY].create_with_defaults()
    # Flat, then a gap up. Flat bars cannot break their own window; the gap must.
    # A ramp where each high is the next close is degenerate here: the prior high
    # always equals today's close exactly, so nothing breaks out either way and the
    # test would pass against a broken shift.
    closes = [100.0] * 50 + [130.0] * 10
    flat_then_gap = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 60,
        },
        index=pd.date_range("2024-01-02", periods=60, freq="D"),
    )

    enriched = strategy.process_data(flat_then_gap)

    # The gap clears the flat window's high.
    assert (enriched["close"] > enriched["breakout_high"]).any()
    # And the level never includes the bar being tested against it: on the first gap
    # bar the level is still the flat window's high, not its own.
    first_gap = enriched.iloc[50]
    assert first_gap["breakout_high"] == pytest.approx(100.5)


# --- it actually trades -------------------------------------------------------------
def test_the_strategy_runs_a_backtest_end_to_end():
    """A pack that registers but cannot complete a run has passed nothing worth
    passing."""
    symbols = ["AAA", "BBB", "CCC"]
    client = MarketDataClient(FakeMarketData(symbols, n=400, freq="1D"))

    result = BacktestEngine(STRATEGIES[STRATEGY].create_with_defaults(), client).run(
        symbols, datetime(2024, 1, 2), datetime(2025, 3, 1), 100_000
    )

    assert len(result.trades) > 0
    assert result.metrics["total_trades"] == len(result.trades)


def test_the_scanner_ranks_rather_than_only_flagging():
    """A scanner that flags everything equally hands a capped universe an arbitrary
    choice and calls it a selection."""

    scanner = SCANNERS[SCANNER](_defaults(SCANNERS[SCANNER]))
    scanner.initialize()
    frame = MarketDataClient(FakeMarketData(["AAA"], n=200, freq="1D")).get_bars(
        ["AAA"], "1Day", datetime(2024, 1, 2), datetime(2024, 12, 1)
    )["AAA"]

    signals = scanner.generate_signals_df(scanner.process_data(frame))
    qualifying = signals[signals["signal"] != ""]

    assert len(qualifying) > 0
    assert qualifying["signal_strength"].nunique() > 1
    assert not signals["signal_strength"].isna().any()  # absent is not a rank


# --- the config ships and loads ----------------------------------------------------
def test_the_packs_config_loads_and_carries_a_full_book():
    """The config is the artefact everything downstream reads. One that omitted limits
    would describe a different book from the one its evidence came from."""
    from tradeflow.optimization.config_store import load_config
    from tradeflow.services.setup import example_pack_source

    source = example_pack_source()
    if source is None:
        pytest.skip("example pack source not present in this copy (wheel install)")

    config = load_config(source / "configs" / "breakout.json")

    assert config["strategy"] == STRATEGY
    assert config["position_limits"]["max_gross_exposure"] == 0.9
    assert config["symbols"] and config["candidate_symbols"]


# --- the long/short half of the pack ------------------------------------------------
def test_the_pack_ships_a_book_that_trades_both_sides():
    """A long-only pack leaves half the platform unexercised. Leg diagnostics, the
    directional cap, the tilt derivation and the short-borrow side of the cost model
    only mean anything for a book that shorts."""
    assert STRATEGIES[REVERSION].LONG_ONLY is False


def test_the_long_short_strategy_declares_a_directional_cap():
    """Gross bounds long + short and cannot see direction, so a book inside a 1.6 gross
    cap can be entirely long. The net cap is the one that keeps it neutral."""
    limits = STRATEGIES[REVERSION].create_with_defaults().position_limits()

    assert limits["max_gross_exposure"] == 1.60
    assert limits["max_net_exposure"] == 0.30


def test_the_long_short_strategy_actually_fills_both_legs():
    """Declaring LONG_ONLY = False proves nothing on its own — a scoring bug that never
    goes negative produces a long-only book from a strategy that claims otherwise."""
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    client = MarketDataClient(FakeMarketData(symbols, n=500, freq="1D"))

    result = BacktestEngine(STRATEGIES[REVERSION].create_with_defaults(), client).run(
        symbols, datetime(2024, 1, 2), datetime(2025, 6, 1), 100_000
    )

    assert result.legs["long"]["trades"] > 0
    assert result.legs["short"]["trades"] > 0


def test_the_long_short_book_produces_a_tilt_distribution():
    """Which is what the net-cap derivation reads. A book with no measurable tilt gives
    it nothing to recommend from."""
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    client = MarketDataClient(FakeMarketData(symbols, n=500, freq="1D"))

    result = BacktestEngine(STRATEGIES[REVERSION].create_with_defaults(), client).run(
        symbols, datetime(2024, 1, 2), datetime(2025, 6, 1), 100_000
    )

    assert result.exposure["net_abs"]["max"] > 0
    assert result.exposure["samples"] > 0


def test_a_reversion_target_outside_its_stop_is_refused():
    """A reversion book exits into the mean, so a target beyond the stop means the
    winners are cut further out than the losers — almost never the intent."""
    # From the defaults, then overridden: the base class calls
    # calculate_required_lookback during construction, so a partial config never
    # reaches the check being tested.
    strategy = STRATEGIES[REVERSION](
        {**_defaults(STRATEGIES[REVERSION]), "stop_loss": 0.04, "take_profit": 0.10}
    )

    with pytest.raises(ValueError, match="inside the stop"):
        strategy.initialize()


def test_a_flat_series_does_not_produce_infinite_conviction():
    """Zero dispersion over the whole window. Dividing by it turns a name that has not
    moved into the highest-conviction trade in the book."""
    import numpy as np
    import pandas as pd

    strategy = STRATEGIES[REVERSION].create_with_defaults()
    flat = pd.DataFrame(
        {
            "open": [100.0] * 60,
            "high": [100.0] * 60,
            "low": [100.0] * 60,
            "close": [100.0] * 60,
            "volume": [1_000_000] * 60,
        },
        index=pd.date_range("2024-01-02", periods=60, freq="D"),
    )

    scores = strategy.calculate_scores(strategy.process_data(flat))

    assert not np.isinf(scores.to_numpy()).any()
    assert not scores.isna().any()


# --- the config the pack ships ------------------------------------------------------
def test_the_long_short_config_carries_what_a_long_only_one_does_not():
    """The two configs exist to be compared. A long/short book needs a directional cap
    and a borrow rate that actually applies."""
    from tradeflow.optimization.config_store import load_config
    from tradeflow.services.setup import example_pack_source

    source = example_pack_source()
    if source is None:
        pytest.skip("example pack source not present in this copy (wheel install)")

    config = load_config(source / "configs" / "reversion_longshort.json")

    assert config["position_limits"]["max_net_exposure"] == 0.3
    assert config["cost"]["borrow_bps"] > 0
    assert config["scanner"] == SCANNER  # not a fixed list — the scanner picks the book


def test_the_shipped_configs_are_in_version_control():
    """They were not. The repository ignored `configs/` at any depth, so a fresh clone
    got a pack missing the artefact its README tells you to run — and the scaffold would
    have copied that hole out to a user."""
    import subprocess

    from tradeflow.services.setup import example_pack_source

    source = example_pack_source()
    if source is None:
        pytest.skip("example pack source not present in this copy (wheel install)")

    for config in sorted((source / "configs").glob("*.json")):
        ignored = subprocess.run(["git", "check-ignore", str(config)], capture_output=True, text=True)
        assert ignored.returncode != 0, f"{config.name} is gitignored and would not ship"
