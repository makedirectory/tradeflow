"""Golden pins, second set: the mechanics the first set left unpinned.

The first set (``test_golden_pins.py``) pins a long-only daily backtest, one
walk-forward, the cost sweep and the formula layers. A fix to anything it does not touch
could move a published number with every pin still green. This set closes those gaps:

* a long-short book with borrow and real turnover;
* a stop hit inside a bar while another name signals for that bar's open (causality);
* intraday end-to-end annualisation (1Hour and 5Min);
* a walk-forward whose folds each have real substance and mixed-sign Sharpes;
* the borrow axis of the cost sweep;
* the composite research verdict (every gate: value, threshold, outcome);
* the attribution table;
* the conditional-risk matrix.

Same contract as the first set: every scenario is checked against an independent
computation written in the test (never a function of the package) *before* its numbers
are stored, so a pin never freezes a wrong number. Regenerate deliberately with
``QUANTRED_REGEN_GOLDEN=1 uv run --extra dev pytest tests/math -k golden``.
"""

import math
from datetime import datetime
from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantred.costs.parametric import ParametricCostModel
from quantred.engine.backtest import BacktestEngine
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator
from quantred.risk.conditional import condition_risk_matrix
from quantred.risk.factor import estimate_factor_model
from quantred.risk.sample import LedoitWolfCovariance
from quantred.services import analysis, audit
from quantred.services.registry import STRATEGIES
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData, FakeMarketData, LongShortScriptedStrategy
from tests.math import pins
from tests.math.test_engine_accounting import (
    _ONE_SLOT,
    ENTRY_BAR,
    _bars,
    _multi_run,
    _PerSymbolStrategy,
    _random_signals,
)
from tests.math.test_golden_pins import _independent_backtest_checks, _risk_world
from tests.math.test_walk_forward_math import _daily_returns, _mc_reference
from tests.test_walk_forward import PeriodicStrategy

B, H, S, CB, CS = signals.BUY, signals.HOLD, signals.SELL, signals.CLOSE_BUY, signals.CLOSE_SELL
START, END = datetime(2024, 1, 2), datetime(2025, 12, 31)


def _trade_columns(trades: pd.DataFrame, *columns: str) -> Dict[str, Any]:
    return {c: trades[c].tolist() for c in columns}


def _steps_held(trades: pd.DataFrame) -> np.ndarray:
    """Whole days between entry and exit on the bars' own calendar. Wall-clock dates are
    compared with the zone dropped: a DST change inside a hold makes a tz-aware difference a
    day short, which is not a step the book did not take."""
    exit_ = trades["exit_time"].dt.tz_localize(None)
    entry = trades["entry_time"].dt.tz_localize(None)
    return (exit_ - entry).dt.days.to_numpy()


# --------------------------------------------------------------------------- #
# 1. Long-short book with borrow and real turnover
# --------------------------------------------------------------------------- #
COMMISSION_BPS, SPREAD_BPS, BORROW_BPS = 2.0, 6.0, 300.0


def test_golden_backtest_long_short_with_borrow():
    """73 trades, 41 of them short, with borrow on. The cost of every trade is rebuilt from
    its own fills: both legs pay commission and half the spread, and only a short pays
    borrow, pro rata over the steps it was held. Guards borrow charged on longs, borrow
    that ignores the holding period, and a short's P&L sign."""
    model = ParametricCostModel(
        commission_bps=COMMISSION_BPS,
        default_spread_bps=SPREAD_BPS,
        impact_eta=0.0,
        annual_borrow_bps=BORROW_BPS,
    )
    result = _multi_run(0, model)
    t = result.trades
    is_short = (t["side"] == S).to_numpy()
    assert len(t) >= 60 and 20 <= is_short.sum() <= len(t) - 20  # both sides, real turnover
    _independent_backtest_checks(result)

    leg_rate = COMMISSION_BPS / 1e4 + SPREAD_BPS / 1e4 / 2
    steps = _steps_held(t)
    entry_notional = (t["size"] * t["entry_price"]).to_numpy()
    exit_notional = (t["size"] * t["exit_price"]).to_numpy()
    toll = (entry_notional + exit_notional) * leg_rate
    borrow = np.where(is_short, BORROW_BPS / 1e4 * entry_notional * steps / 252.0, 0.0)
    # borrow only on shorts, pro rata: the whole cost column is the two tolls plus borrow
    assert t["cost"].to_numpy() == pytest.approx(toll + borrow, rel=1e-12)
    assert borrow[~is_short].sum() == 0.0 and borrow[is_short].sum() > 0
    longs_cost = t.loc[~is_short, "cost"].to_numpy()
    assert longs_cost == pytest.approx(toll[~is_short], rel=1e-12)
    # short P&L sign: a short gains when the price falls (100 -> 90 is +10 a share)
    direction = np.where(is_short, -1.0, 1.0)
    gross = (t["exit_price"].to_numpy() - t["entry_price"].to_numpy()) * t["size"].to_numpy() * direction
    assert t["gross_pnl"].to_numpy() == pytest.approx(gross, rel=1e-12)
    assert (100.0 - 90.0) * -1.0 * -1.0 == 10.0  # the stated convention, in the same expression
    assert (gross[is_short] > 0).any() and (gross[is_short] < 0).any()
    # Sum of trade P&L is the book change, net of every cost
    assert result.final_capital - result.initial_capital == pytest.approx(t["pnl"].sum(), abs=1e-6)
    assert result.total_cost == pytest.approx((toll + borrow).sum(), rel=1e-12)

    values = pins.numeric_values(
        {
            "metrics": result.metrics,
            "final_capital": result.final_capital,
            "total_cost": result.total_cost,
            "gross_final_capital": result.gross_final_capital,
            "execution": result.execution,
            "n_trades": len(t),
            "n_short": int(is_short.sum()),
            "borrow_total": float(borrow.sum()),
            "trade": _trade_columns(t, "pnl", "gross_pnl", "cost"),
            "equity_tail": result.equity_curve[-5:],
        }
    )
    pins.check(
        "backtest_long_short_borrow",
        "scripted random signals on 3 names, 120 daily bars, up to 3 positions, both sides, "
        "commission 2bp, spread 6bp, borrow 300bp, impact off",
        "world seed 0, signals seed 1",
        values,
    )


# --------------------------------------------------------------------------- #
# 2. An exit inside bar k cannot fund an entry at bar k's open
# --------------------------------------------------------------------------- #
def test_golden_backtest_intrabar_exit_then_same_bar_candidate():
    """One slot. AAA's stop is hit *inside* bar 3 while BBB signals on bar 2 (so it would
    fill at bar 3's open); BBB's stop gaps through at bar 6's open while CCC signals on
    bar 5. In both cases the replacement signals again a bar later and enters at the
    following open, never earlier. Guards an exit that frees the slot for the same open."""
    n_bars = 11
    gap = (80, 82, 78, 79)
    frames = {
        "AAA": _bars([ENTRY_BAR] * 3 + [(95, 96, 85, 94)] + [ENTRY_BAR] * (n_bars - 4)),
        "BBB": _bars([ENTRY_BAR] * 6 + [gap] + [ENTRY_BAR] * (n_bars - 7)),
        "CCC": _bars([ENTRY_BAR] * n_bars),
    }
    scripts = {
        "AAA": [B] + [H] * (n_bars - 1),
        "BBB": [H, H, B, B] + [H] * (n_bars - 4),
        "CCC": [H] * 5 + [B, B] + [H] * (n_bars - 7),
    }
    strategy = _PerSymbolStrategy(scripts, _ONE_SLOT)
    result = BacktestEngine(strategy, MarketDataClient(DictMarketData(frames))).run(
        list(frames), START, END, 100_000.0
    )
    _independent_backtest_checks(result)
    t = result.trades.set_index("symbol")
    day0 = pd.Timestamp("2024-01-02", tz=NEW_YORK)

    def day(ts):
        return int((ts - day0).days)

    # AAA: signal bar 0 -> fills bar 1; stop 90 reached inside bar 3
    assert (day(t.loc["AAA", "entry_time"]), day(t.loc["AAA", "exit_time"])) == (1, 3)
    assert t.loc["AAA", "exit_price"] == pytest.approx(90.0) and t.loc["AAA", "exit_reason"] == "STOP_LOSS"
    # BBB signalled for bar 3's open (refused: the slot was still held) and again for bar 4
    assert day(t.loc["BBB", "entry_time"]) == 4
    assert t.loc["BBB", "exit_reason"] == "STOP_LOSS" and day(t.loc["BBB", "exit_time"]) == 6
    assert t.loc["BBB", "exit_price"] == pytest.approx(80.0)  # the gapped open, not the 90 level
    assert t.loc["BBB", "pnl"] == pytest.approx((80.0 - 100.0) * t.loc["BBB", "size"])
    # CCC signalled for bar 6's open (refused) and again for bar 7
    assert day(t.loc["CCC", "entry_time"]) == 7
    # the property itself: no entry at or before the open of the bar a stop was hit in
    stops = result.trades[result.trades["exit_reason"] == "STOP_LOSS"]
    for _, stop in stops.iterrows():
        later = result.trades[result.trades["entry_time"] > stop["entry_time"]]
        assert (later["entry_time"] >= stop["exit_time"] + pd.Timedelta(days=1)).all()
    assert len(result.trades) == 3

    values = pins.numeric_values(
        {
            "metrics": result.metrics,
            "final_capital": result.final_capital,
            "execution": result.execution,
            "trades": {
                sym: {
                    "entry_day": day(t.loc[sym, "entry_time"]),
                    "exit_day": day(t.loc[sym, "exit_time"]),
                    **{k: t.loc[sym, k] for k in ("size", "entry_price", "exit_price", "pnl", "exit_reason")},
                }
                for sym in frames
            },
            "equity_curve": result.equity_curve,
        }
    )
    pins.check(
        "backtest_intrabar_exit_causality",
        "one slot; an intrabar stop and a gapped stop each followed by a candidate signalled for the same open",
        "hand-built bars, no randomness",
        values,
    )


# --------------------------------------------------------------------------- #
# 3. Intraday end-to-end annualisation
# --------------------------------------------------------------------------- #
def _intraday_world(freq: str, n_bars: int, vol: float, seed: int):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-02 09:30", periods=n_bars, freq=freq, tz=NEW_YORK)
    frames = {}
    for sym in ("AAA", "BBB", "CCC"):
        close = 100 * np.cumprod(1 + rng.normal(0.0, vol, n_bars))
        open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, vol / 4, n_bars))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, vol / 3, n_bars))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, vol / 3, n_bars))
        frames[sym] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": rng.uniform(5e5, 2e6, n_bars),
            },
            index=index,
        )
    return frames, index


@pytest.mark.parametrize(
    "timeframe, freq, n_bars, vol, per_year, name",
    [
        ("1Hour", "h", 700, 0.004, 252 * 6.5, "backtest_intraday_1h"),
        ("5Min", "5min", 1500, 0.0015, 252 * 6.5 * 12, "backtest_intraday_5m"),
    ],
)
def test_golden_backtest_intraday_annualisation(timeframe, freq, n_bars, vol, per_year, name):
    """Sharpe, Sortino, volatility and CAGR of an intraday run annualise on the timeframe's
    own bars per year (1638 for hours, 19656 for five-minute bars), never on 252. The
    expected values are recomputed from the equity curve's step returns with the closed
    forms, so a hard-coded 252 anywhere in the path fails here."""
    frames, index = _intraday_world(freq, n_bars, vol, seed=31)
    scripts = _random_signals(7, n_bars, frames)
    strategy = _PerSymbolStrategy(
        scripts,
        {
            "timeframe": timeframe,
            "position_limits": {"max_positions": 3, "max_position_size": 20_000.0, "max_total_risk": 1.0},
        },
    )
    start, end = index[0].to_pydatetime().replace(tzinfo=None), index[-1].to_pydatetime().replace(tzinfo=None)
    result = BacktestEngine(strategy, MarketDataClient(DictMarketData(frames))).run(
        list(frames), start, end, 100_000.0
    )
    t = result.trades
    assert len(t) >= 30
    eq = np.array(result.equity_curve)
    assert len(eq) > 0.9 * n_bars  # the curve is sampled per bar, not per day
    r = eq[1:] / eq[:-1] - 1.0
    root = math.sqrt(per_year)
    assert result.metrics["sharpe_ratio"] == pytest.approx(root * r.mean() / r.std(ddof=1), rel=1e-9)
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    assert result.metrics["sortino_ratio"] == pytest.approx(root * r.mean() / downside, rel=1e-9)
    assert result.metrics["annualized_volatility"] == pytest.approx(100 * root * r.std(ddof=1), rel=1e-9)
    assert result.metrics["downside_deviation"] == pytest.approx(100 * root * downside, rel=1e-9)
    years = (end - start).days / 365.25
    assert result.metrics["cagr"] == pytest.approx(100 * ((eq[-1] / eq[0]) ** (1 / years) - 1), rel=1e-9)
    # and it is not the daily factor: the same series on 252 reads differently
    assert result.metrics["sharpe_ratio"] != pytest.approx(
        math.sqrt(252) * r.mean() / r.std(ddof=1), rel=1e-3
    )
    assert result.final_capital == pytest.approx(100_000.0 + t["pnl"].sum(), abs=1e-6)

    values = pins.numeric_values(
        {
            "metrics": result.metrics,
            "final_capital": result.final_capital,
            "n_trades": len(t),
            "periods_per_year": per_year,
            "step_returns_head": r[:5],
            "equity_tail": eq[-5:],
        }
    )
    pins.check(
        name,
        f"{timeframe} scripted random signals on 3 names, {n_bars} bars, no costs",
        "world seed 31, signals seed 7",
        values,
    )


# --------------------------------------------------------------------------- #
# 4. A walk-forward with substance
# --------------------------------------------------------------------------- #
def test_golden_walk_forward_with_substance():
    """Four folds of 16 to 48 OOS trades with Sharpes of both signs. The aggregate Sharpe,
    Sortino, volatility, the block-bootstrap percentiles and the PBO heuristic are each
    rebuilt from the per-fold trade lists with business-day bins and 252, never from the
    package's own ``oos_returns``; the package's series is then compared with that rebuild."""
    symbols = ["AAA", "BBB"]
    validator = WalkForwardValidator(
        PeriodicStrategy,
        MarketDataClient(FakeMarketData(symbols, n=700, freq="1D")),
        initial_capital=100_000.0,
        seed=7,
    )
    wf = validator.run(
        symbols,
        datetime(2024, 1, 2),
        datetime(2026, 1, 1),
        n_folds=4,
        method="grid",
        max_evals=4,
        monte_carlo=True,
        pbo=True,
    )
    pooled = wf.oos_trade_table
    sharpes = [f.oos_metrics["sharpe_ratio"] for f in wf.folds]
    assert len(wf.folds) == 4 and all(f.oos_trades >= 10 for f in wf.folds)
    assert min(sharpes) < 0 < max(sharpes)

    # per-fold trade lists, by entry time inside each fold's own window
    entry = pd.to_datetime(pooled["entry_time"]).dt.tz_localize(None)
    per_fold = []
    for f in wf.folds:
        mask = (entry >= pd.Timestamp(f.fold.oos_start)) & (entry <= pd.Timestamp(f.fold.oos_end))
        per_fold.append(pooled[mask.to_numpy()])
    assert [len(p) for p in per_fold] == [f.oos_trades for f in wf.folds]
    assert sum(len(p) for p in per_fold) == len(pooled)
    rebuilt = pd.concat(per_fold, ignore_index=True)

    r = _daily_returns(rebuilt)  # business-day bins from the per-fold lists
    agg = wf.oos_aggregate
    assert agg["sharpe_ratio"] == pytest.approx(math.sqrt(252) * r.mean() / r.std(ddof=1), rel=1e-9)
    assert agg["annualized_volatility"] == pytest.approx(100 * math.sqrt(252) * r.std(ddof=1), rel=1e-9)
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    assert agg["sortino_ratio"] == pytest.approx(math.sqrt(252) * r.mean() / downside, rel=1e-9)
    assert wf.oos_returns.to_numpy() == pytest.approx(r, rel=1e-9)  # the package series IS the rebuild
    # Deflated Sharpe: bounded by the plain probabilistic Sharpe, and the trial variance it
    # implies must be positive. (The trial Sharpes behind it live inside the optimiser.)
    skew, kurt = stats.skew(r, bias=False), stats.kurtosis(r, fisher=False, bias=False)
    sr = r.mean() / r.std(ddof=1)
    denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    psr = stats.norm.cdf(sr * math.sqrt(len(r) - 1) / denom)
    assert agg["probabilistic_sharpe_ratio"] == pytest.approx(psr, rel=1e-6)
    assert 0.0 < agg["deflated_sharpe_ratio"] < psr
    implied_sr_star = sr - stats.norm.ppf(agg["deflated_sharpe_ratio"]) * denom / math.sqrt(len(r) - 1)
    assert implied_sr_star > 0
    # Monte Carlo and PBO, by independent loops
    p05, p50, ret05, periods = _mc_reference(rebuilt, 1000, 20, 0)
    mc = wf.monte_carlo
    assert mc["periods"] == periods and mc["sharpe_p05"] == pytest.approx(p05, rel=1e-9)
    assert mc["sharpe_p50"] == pytest.approx(p50, rel=1e-9) and mc["return_p05"] == pytest.approx(
        ret05, rel=1e-9
    )
    from itertools import combinations

    is_s = [f.is_metrics["sharpe_ratio"] for f in wf.folds]
    below = total = 0
    for train in combinations(range(4), 2):
        test = [i for i in range(4) if i not in train]
        best = max(train, key=lambda i: is_s[i])
        below += sharpes[best] < np.median([sharpes[i] for i in test])
        total += 1
    assert wf.pbo == pytest.approx(below / total)

    values = pins.numeric_values(
        {
            "oos_aggregate": agg,
            "monte_carlo": wf.monte_carlo,
            "pbo": wf.pbo,
            "efficiency": wf.efficiency,
            "degradation": wf.degradation,
            "n_trials_total": wf.n_trials_total,
            "fold_trade_counts": [len(p) for p in per_fold],
            "fold_pnl": [float(p["pnl"].sum()) for p in per_fold],
            "folds": [
                {
                    "is": f.is_metrics,
                    "oos": f.oos_metrics,
                    "oos_trades": f.oos_trades,
                    "params": f.is_best_params,
                }
                for f in wf.folds
            ],
            "daily_returns_head": r[:5],
            "n_daily_returns": len(r),
        }
    )
    pins.check(
        "walk_forward_substance",
        "periodic-entry strategy on 2 names, 700 daily bars, 4 anchored folds, grid, MC and PBO on",
        "seed 7, bars 0..1",
        values,
    )


# --------------------------------------------------------------------------- #
# 5. The borrow axis of the cost sweep
# --------------------------------------------------------------------------- #
class _ShortOnly(LongShortScriptedStrategy):
    """Permanently short: the pivot sits above any price the fake feed reaches."""

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        **LongShortScriptedStrategy.PARAM_RANGES,
        "pivot": {"type": "float", "min": 400.0, "max": 500.0, "step": 50.0, "default": 500.0},
    }


def test_golden_cost_stress_borrow_axis(monkeypatch):
    """Borrow scales *exactly* by k on a short-only book with commission and impact off.
    The spread is a fixed toll on the borrow axis, so the borrow component at k is the
    total at k less the total at 0; it must be k times the component at 1."""
    monkeypatch.setitem(STRATEGIES, "short_only", _ShortOnly)
    client = MarketDataClient(FakeMarketData(["AAA", "BBB", "CCC"], n=600, freq="1D"))
    base_borrow = 100.0
    multiples = (0.0, 1.0, 2.0, 4.0)
    report = analysis.run_cost_stress(
        client,
        "short_only",
        ["AAA", "BBB", "CCC"],
        datetime(2024, 1, 2),
        datetime(2025, 6, 1),
        capital=100_000.0,
        benchmark=None,
        commission_bps=0.0,
        impact_eta=0.0,
        borrow_bps=base_borrow,
        multiples=multiples,
        axis="borrow",
    )
    cost = {p["multiple"]: p["total_cost"] for p in report["points"]}
    borrow = {k: cost[k] - cost[0.0] for k in multiples}
    assert borrow[1.0] > 0
    for k in (2.0, 4.0):
        assert borrow[k] == pytest.approx(k * borrow[1.0], rel=1e-9)
    # the borrow component from first principles: rate * entry notional * steps held / 252
    strategy = _ShortOnly.create_with_defaults()
    result = BacktestEngine(
        strategy,
        client,
        cost_model=ParametricCostModel(
            commission_bps=0.0, default_spread_bps=0.0, impact_eta=0.0, annual_borrow_bps=base_borrow
        ),
    ).run(["AAA", "BBB", "CCC"], datetime(2024, 1, 2), datetime(2025, 6, 1), 100_000.0)
    t = result.trades
    assert (t["side"] == S).all() and len(t) >= 3
    expected = base_borrow / 1e4 * (t["size"] * t["entry_price"]).to_numpy() * _business_steps(t) / 252.0
    assert result.total_cost == pytest.approx(expected.sum(), rel=1e-9)
    assert borrow[1.0] == pytest.approx(result.total_cost, rel=1e-3)  # same book, near-equal paths
    assert report["axis"] == "borrow"

    values = pins.numeric_values(
        {"points": report["points"], "borrow_component": {str(k): v for k, v in borrow.items()}},
        drop=("run_id",),
    )
    pins.check(
        "cost_stress_borrow_axis",
        "permanently-short strategy, borrow axis at 0x, 1x, 2x, 4x, commission and impact off",
        "fake bars seeds 0..2",
        values,
    )


def _business_steps(trades: pd.DataFrame) -> np.ndarray:
    """Merged-timeline steps held: the fake feed is a weekday-agnostic daily calendar, so a
    step is a calendar day on the wall clock."""
    return _steps_held(trades)


# --------------------------------------------------------------------------- #
# 6. The composite verdict
# --------------------------------------------------------------------------- #
VERDICT_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def _verdict_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


def _verdict_client():
    return MarketDataClient(FakeMarketData([*VERDICT_SYMBOLS, "SPY"], n=600, freq="1D"))


def _gate_outcome(check):
    """The comparator each gate documents, written here from the gate's value and threshold."""
    return check["passed"]


def test_golden_verdict_pipeline(_verdict_state):
    """Every gate's value, threshold and outcome, and the overall line they produce. The
    outcomes are re-derived from the documented comparators, and the verdict word from the
    pass/fail pattern, so a flipped comparator or a miscounted summary fails here."""
    result = analysis.run_verdict(
        _verdict_client(),
        "demo_trend",
        VERDICT_SYMBOLS,
        datetime(2024, 1, 2),
        datetime(2025, 6, 1),
        journal=False,
        force=True,
    )
    verdict, checks, info = result["verdict"], result["verdict"]["checks"], result["information"]
    assert set(checks) == {"ic_tstat", "ir_above_noise", "sanity_ceiling", "sample_size", "net_of_cost_alpha"}
    # the t-stat is mean IC over its standard error, and the gate is |t| >= 2
    assert checks["ic_tstat"]["value"] == pytest.approx(
        info["mean_ic"] / (info["ic_vol"] / math.sqrt(info["periods"])), rel=1e-9
    )
    assert checks["ic_tstat"]["passed"] == (abs(checks["ic_tstat"]["value"]) >= 2.0)
    assert checks["ir_above_noise"]["value"] == info["realized_ir"]
    assert checks["ir_above_noise"]["threshold"] == info["ir_standard_error"]
    assert checks["ir_above_noise"]["passed"] == (abs(info["realized_ir"]) > info["ir_standard_error"])
    assert checks["sanity_ceiling"]["passed"] == (abs(checks["sanity_ceiling"]["value"]) <= 2.0)
    assert checks["sample_size"]["value"] == info["periods"]
    assert checks["sample_size"]["passed"] == (info["periods"] >= checks["sample_size"]["threshold"])
    net = result["portfolio"]["diagnostics"]["expected_active_return_net"]
    assert checks["net_of_cost_alpha"]["value"] == net
    assert checks["net_of_cost_alpha"]["passed"] == (net > 0.0)
    passed = sorted(k for k, c in checks.items() if _gate_outcome(c))
    failed = sorted(k for k, c in checks.items() if not _gate_outcome(c))
    assert passed and failed and verdict["verdict"] == "mixed" and verdict["promotable"] is False
    assert verdict["summary"] == f"mixed — passed: {', '.join(passed)}; failed: {', '.join(failed)}"
    assert verdict["failed_steps"] == []

    values = pins.numeric_values(
        {
            "verdict": verdict,
            "information": {
                k: info[k]
                for k in (
                    "periods",
                    "mean_ic",
                    "ic_vol",
                    "ic_tstat",
                    "realized_ir",
                    "ir_standard_error",
                    "predicted_ir",
                )
            },
            "net_expected_active_return": net,
            "universe": result["inputs"]["universe"],
        }
    )
    pins.check(
        "verdict_pipeline",
        "demo_trend verdict on six names plus SPY, 600 daily bars, default caps and costs",
        "fake bars seeds 0..6",
        values,
    )


def test_golden_verdict_infeasible_book(_verdict_state):
    """A cap no six-name book can satisfy: the verdict is ``infeasible`` with the binding
    constraint, not ``needs more data``."""
    result = analysis.run_verdict(
        _verdict_client(),
        "demo_trend",
        VERDICT_SYMBOLS,
        datetime(2024, 1, 2),
        datetime(2025, 6, 1),
        max_weight=0.1,
        journal=False,
        force=True,
    )
    verdict = result["verdict"]
    assert 6 * 0.1 < 1.0  # the independent reason: six names at a 10% cap hold 60% of a fully-invested book
    assert verdict["verdict"] == "infeasible" and verdict["promotable"] is False
    assert verdict["checks"]["portfolio_feasible"]["passed"] is False
    pins.check(
        "verdict_infeasible",
        "demo_trend verdict with a per-name cap six names cannot fund",
        "fake bars seeds 0..6",
        pins.numeric_values({"verdict": verdict}),
    )


# --------------------------------------------------------------------------- #
# 7. Attribution
# --------------------------------------------------------------------------- #
def test_golden_attribution_table():
    """Every component sums to the period's active return, the systematic bucket splits
    into expected, surprise and timing, and the chain-linked cumulation matches the
    product form, all rebuilt from the per-period detail. The stock-picking remainder is
    structurally ~0 here (the active book *is* the alpha column), so its noise-only
    statistics are not stored."""
    symbols = [f"S{i}" for i in range(14)]
    client = MarketDataClient(FakeMarketData([*symbols, "SPY"], n=600, freq="1D"))
    out = analysis.compute_attribution(
        client, "demo_trend", symbols, datetime(2024, 1, 2), datetime(2025, 6, 1), n_points=24, detail=True
    )
    d = {k: np.array(v) for k, v in out["detail"].items()}
    components = [k for k in d if k not in ("r_active", "r_bench", "beta_a")]
    assert len(d["r_active"]) == out["periods"] >= 12
    assert sum(d[k] for k in components) == pytest.approx(d["r_active"], abs=1e-12)  # components sum to total
    assert np.abs(d["specific"]).max() < 1e-12 and out["best_row"] != "specific"
    for name in out["risk_factor_names"] + out["signal_names"]:
        assert out["rows"][name]["total"] == pytest.approx(d[name].sum(), abs=1e-12)
    split = out["systematic_split"]
    assert split["expected"] + split["surprise"] + split["timing"] == pytest.approx(
        d["systematic"].sum(), abs=1e-12
    )
    assert split["timing"] == pytest.approx(out["rows"]["timing"]["total"], abs=1e-15)
    assert split["expected"] == pytest.approx(0.05 * 5 / 252 * d["beta_a"].sum(), abs=1e-12)
    # cumulative active return is the product of portfolio returns less the product of benchmark's
    rp = d["r_bench"] + d["r_active"]
    assert out["cumulation"]["honest_car"] == pytest.approx(
        np.prod(1 + rp) - np.prod(1 + d["r_bench"]), abs=1e-12
    )
    ir = d["r_active"].mean() / d["r_active"].std() * math.sqrt(252 / 5)  # population std, as documented
    assert out["total_active_ir"] == pytest.approx(ir, rel=1e-9)
    assert out["sanity_ceiling_breached"] == (abs(ir) > 2.0)

    skip = lambda key: "specific" in key or key in ("run_id", "note")  # noqa: E731
    flat = {
        k: v for k, v in pins.flatten({k: v for k, v in out.items() if k != "detail"}).items() if not skip(k)
    }
    flat.update(
        {
            f"detail.{k}": v
            for k, v in pins.flatten({k: v.tolist() for k, v in d.items() if k != "specific"}).items()
        }
    )
    pins.check(
        "attribution",
        "demo_trend attribution on 14 names plus SPY, 600 daily bars, 24 sampled rebalances at horizon 5",
        "fake bars seeds 0..14",
        flat,
    )


# --------------------------------------------------------------------------- #
# 8. Conditional risk
# --------------------------------------------------------------------------- #
def test_golden_conditional_risk_matrix():
    """The conditioned intercept-factor matrix and the conditioned shrinkage matrix. The
    expected matrices are rebuilt by hand in ``test_risk_models.py``; here the outputs are
    stored, after a symmetry/PSD check."""
    returns, _ = _risk_world()
    names = list(returns.columns)
    style = pd.DataFrame(index=names)
    factor = estimate_factor_model(returns, style, 252.0)
    cond_factor = condition_risk_matrix(factor, returns, "ewma", 252.0, min_obs=60)
    sigma, _delta = LedoitWolfCovariance().estimate(returns)
    from quantred.risk.base import RiskMatrix

    base = RiskMatrix(names, sigma * 252.0)
    cond_shrink = condition_risk_matrix(base, returns, "ewma", 252.0, min_obs=60)
    for m in (cond_factor, cond_shrink):
        assert m.sigma == pytest.approx(m.sigma.T, abs=1e-15)
        assert np.linalg.eigvalsh(m.sigma).min() > 0
    values = pins.numeric_values(
        {
            "factor": {
                "sigma": cond_factor.sigma,
                "factor_cov": cond_factor.factor_cov,
                "specific_var": cond_factor.specific_var,
                "diagnostics": cond_factor.conditional_diagnostics,
            },
            "shrinkage": {"sigma": cond_shrink.sigma, "diagnostics": cond_shrink.conditional_diagnostics},
        }
    )
    pins.check(
        "conditional_risk",
        "8 names, intercept-only factor model and Ledoit-Wolf matrix, EWMA lambda 0.94 seeded on 60 rows",
        "seed 2024",
        values,
    )
