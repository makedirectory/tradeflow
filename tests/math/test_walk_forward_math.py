"""Math contract: the walk-forward validator's aggregate, deflation, resampling and
overfitting indicators.

The aggregate has no single simulation behind it, so it is rebuilt from closed-trade
P&L on trading days. Every reference here is computed from the same trade table with
plain numpy; none calls the package's own metric functions.

Related existing tests: ``tests/test_aggregate_annualisation.py``,
``tests/test_walk_forward.py``, ``tests/test_concat_trades.py``.

Defects this module guards:

* the aggregate was annualised on a calendar-day series and the Deflated Sharpe mixed a
  per-calendar-day trial variance with a per-trading-day Sharpe;
* the Monte Carlo "Sharpe" was ``sqrt(N_trades) * mean/std`` of per-trade P&L - a
  t-statistic that grew with trade count whatever the strategy earned per year;
* the PBO heuristic ranks folds, not configurations, and must be labelled and computed as
  such;
* two copies of the trade-concatenation helper.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import quantred
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator, _concat_trades
from tests.fakes import FakeMarketData
from tests.test_walk_forward import SYMBOLS, PeriodicStrategy

CAPITAL = 100_000.0
EULER = 0.5772156649015329


def _validator(**kw) -> WalkForwardValidator:
    return WalkForwardValidator(
        PeriodicStrategy,
        MarketDataClient(FakeMarketData(SYMBOLS, n=60, freq="1D")),
        initial_capital=CAPITAL,
        seed=1,
        **kw,
    )


def _trades(n_days=500, seed=3, per_day=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    exits = np.repeat(days.to_numpy(), per_day) + pd.Timedelta(hours=16)
    return pd.DataFrame({"exit_time": exits, "pnl": rng.normal(35.0, 380.0, n_days * per_day)})


def _daily_returns(trades):
    """Independent: business-day P&L (weekend exits fall on the Friday), equity, returns."""
    t = trades.copy()
    t["day"] = pd.to_datetime(t["exit_time"]).dt.normalize()
    t["day"] = t["day"].where(
        t["day"].dt.dayofweek < 5, t["day"] - pd.to_timedelta(t["day"].dt.dayofweek - 4, unit="D")
    )
    by_day = t.groupby("day")["pnl"].sum()
    days = pd.bdate_range(by_day.index.min(), by_day.index.max())
    pnl = by_day.reindex(days, fill_value=0.0).to_numpy()
    equity = CAPITAL + np.concatenate([[0.0], np.cumsum(pnl)])
    return equity[1:] / equity[:-1] - 1.0


# --------------------------------------------------------------------------- #
# Aggregate clock
# --------------------------------------------------------------------------- #
def test_aggregate_sharpe_and_volatility_are_trading_day_quantities():
    trades = _trades()
    r = _daily_returns(trades)
    out = _validator()._metrics_for_trades(trades, None, None, 1, None)
    assert out["sharpe_ratio"] == pytest.approx(math.sqrt(252) * r.mean() / r.std(ddof=1), rel=1e-9)
    assert out["annualized_volatility"] == pytest.approx(100 * math.sqrt(252) * r.std(ddof=1), rel=1e-9)
    assert out["skew"] == pytest.approx(stats.skew(r, bias=False), rel=1e-6)


def test_aggregate_deflated_sharpe_is_in_per_trading_day_units():
    """The trial Sharpes arrive annualised; the deflation needs the variance of their
    per-trading-day equivalents, and the aggregate's PSR is per trading day too."""
    trades = _trades(seed=4)
    r = _daily_returns(trades)
    annualised = [0.4, 1.1, -0.2, 0.9, 0.7, 1.3, 0.1, 0.6]
    var_trial = _validator()._per_period_sr_variance(annualised)
    assert var_trial == pytest.approx(np.var(np.array(annualised) / math.sqrt(252), ddof=1), rel=1e-12)

    n_trials = 40
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r, bias=False), stats.kurtosis(r, fisher=False, bias=False)
    sr_star = math.sqrt(var_trial) * (
        (1 - EULER) * stats.norm.ppf(1 - 1 / n_trials) + EULER * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    )
    z = (sr - sr_star) * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    got = _validator()._metrics_for_trades(trades, None, None, n_trials, var_trial)
    assert got["deflated_sharpe_ratio"] == pytest.approx(stats.norm.cdf(z), rel=1e-6)
    assert got["deflation_applied"] is True


def test_trial_variance_needs_two_trials():
    assert _validator()._per_period_sr_variance([1.0]) is None
    assert _validator()._per_period_sr_variance([]) is None


def test_two_trades_a_day_give_the_same_aggregate_as_their_daily_sum():
    one = _trades(300, 5, per_day=1)
    split = pd.concat([one.assign(pnl=one["pnl"] / 2), one.assign(pnl=one["pnl"] / 2)], ignore_index=True)
    a = _validator()._metrics_for_trades(one, None, None, 1, None)["sharpe_ratio"]
    b = _validator()._metrics_for_trades(split, None, None, 1, None)["sharpe_ratio"]
    assert b == pytest.approx(a, rel=1e-9)


# --------------------------------------------------------------------------- #
# Monte Carlo resampling
# --------------------------------------------------------------------------- #
def _mc_reference(trades, n_resamples, block, seed):
    r = _daily_returns(trades)
    equity = CAPITAL * np.concatenate([[1.0], np.cumprod(1 + r)])
    pnl = np.diff(equity)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(len(pnl) / block)
    sharpes, totals = [], []
    for _ in range(n_resamples):
        starts = rng.integers(0, len(pnl) - block + 1, size=n_blocks)
        sample = np.concatenate([pnl[s : s + block] for s in starts])[: len(pnl)]
        sharpes.append(math.sqrt(252) * sample.mean() / sample.std(ddof=1))
        totals.append(sample.sum())
    return np.percentile(sharpes, 5), np.percentile(sharpes, 50), np.percentile(totals, 5), len(pnl)


def test_monte_carlo_matches_an_independent_block_bootstrap():
    trades = _trades(260, 6)
    out = _validator().monte_carlo([trades], n_resamples=200, block=20, seed=9)
    p05, p50, ret05, periods = _mc_reference(trades, 200, 20, 9)
    assert out["periods"] == periods and out["samples"] == 200
    assert out["sharpe_p05"] == pytest.approx(p05, rel=1e-9)
    assert out["sharpe_p50"] == pytest.approx(p50, rel=1e-9)
    assert out["return_p05"] == pytest.approx(ret05, rel=1e-9)
    assert out["sharpe_p05"] <= out["sharpe_p50"]
    assert "trading days" in out["basis"]


def test_monte_carlo_sharpe_does_not_grow_with_trade_count():
    """Splitting each day's P&L across ten trades leaves daily P&L - and so the Sharpe
    distribution - unchanged; the per-trade t-statistic form grew by sqrt(10)."""
    one = _trades(260, 7)
    ten = pd.concat([one.assign(pnl=one["pnl"] / 10)] * 10, ignore_index=True)
    a = _validator().monte_carlo([one], n_resamples=100, block=20, seed=1)
    b = _validator().monte_carlo([ten], n_resamples=100, block=20, seed=1)
    assert b["sharpe_p50"] == pytest.approx(a["sharpe_p50"], rel=1e-9)


def test_monte_carlo_is_deterministic_per_seed_and_flags_short_series():
    trades = _trades(260, 8)
    v = _validator()
    assert v.monte_carlo([trades], 50, 20, 3) == v.monte_carlo([trades], 50, 20, 3)
    assert v.monte_carlo([trades], 50, 20, 3) != v.monte_carlo([trades], 50, 20, 4)
    assert v.monte_carlo([trades.head(10)], 50, 20, 3) == {"samples": 0, "periods": 10}
    assert v.monte_carlo([pd.DataFrame()], 50, 20, 3)["samples"] == 0


# --------------------------------------------------------------------------- #
# PBO heuristic
# --------------------------------------------------------------------------- #
def _folds(is_scores, oos_scores):
    return [
        SimpleNamespace(is_metrics={"sharpe_ratio": i}, oos_metrics={"sharpe_ratio": o})
        for i, o in zip(is_scores, oos_scores)
    ]


def test_pbo_known_answer_by_enumeration():
    """IS 1,2,3,4 against OOS 4,3,2,1. Of the six half-splits the in-sample best lands
    below the out-of-sample median of the other half in four: (0,3) 1<2.5, (1,2) 2<2.5,
    (1,3) 1<3, (2,3) 1<3.5; (0,1) 3 vs 1.5 and (0,2) 2 vs 2 are not below."""
    v = _validator()
    assert v.estimate_pbo([], _folds([1, 2, 3, 4], [4, 3, 2, 1]), "sharpe_ratio") == pytest.approx(4 / 6)


def test_pbo_is_lower_when_selection_persists_than_when_it_reverses_and_needs_four_folds():
    v = _validator()
    scores = [1, 2, 3, 4, 5, 6]
    persistent = v.estimate_pbo([], _folds(scores, scores), "sharpe_ratio")
    reversed_ = v.estimate_pbo([], _folds(scores, scores[::-1]), "sharpe_ratio")
    assert persistent == pytest.approx(0.2) and reversed_ == pytest.approx(
        0.8
    )  # 4 of 20 splits vs 16 of 20 (brute force)
    assert v.estimate_pbo([], _folds([1, 2, 3], [3, 2, 1]), "sharpe_ratio") is None


def test_pbo_matches_brute_force_on_random_scores():
    from itertools import combinations

    rng = np.random.default_rng(10)
    is_s, oos_s = rng.normal(size=6), rng.normal(size=6)
    below = total = 0
    for train in combinations(range(6), 3):
        test = [i for i in range(6) if i not in train]
        best = max(train, key=lambda i: is_s[i])
        below += oos_s[best] < np.median(oos_s[list(test)])
        total += 1
    assert _validator().estimate_pbo([], _folds(is_s, oos_s), "sharpe_ratio") == pytest.approx(below / total)


def test_pbo_infinite_scores_are_clamped_not_nan():
    out = _validator().estimate_pbo([], _folds([1, 2, 3, np.inf], [4, 3, 2, 1]), "sharpe_ratio")
    assert out is not None and 0.0 <= out <= 1.0


# --------------------------------------------------------------------------- #
# Folds, efficiency, concatenation
# --------------------------------------------------------------------------- #
def test_fold_windows_are_ordered_disjoint_and_leave_the_embargo_gap():
    from datetime import datetime, timedelta

    v = _validator()
    folds, holdout = v.build_folds(
        datetime(2024, 1, 2),
        datetime(2025, 6, 1),
        mode="anchored",
        n_folds=4,
        embargo_days=3,
        holdout_days=30,
    )
    assert len(folds) >= 3 and holdout == (datetime(2025, 5, 2), datetime(2025, 6, 1))
    for a, b in zip(folds, folds[1:]):
        assert a.oos_end < b.oos_start  # no bar belongs to two folds
        assert b.oos_start - a.oos_end <= timedelta(seconds=1)  # contiguous half-open windows
    for f in folds:
        assert f.oos_start - f.is_end == timedelta(days=3)  # the embargo
        assert f.is_start == folds[0].is_start  # anchored: the training window only grows
    assert folds[-1].oos_end < holdout[0]  # the holdout never re-enters the fold region


def test_walk_forward_efficiency_is_mean_oos_over_mean_is():
    v = _validator()
    folds = _folds([1.0, 2.0, 3.0], [0.5, 1.0, 2.5])
    assert v._efficiency(folds, "sharpe_ratio") == pytest.approx((4.0 / 3) / 2.0, rel=1e-12)
    assert v._efficiency(_folds([0.0, 0.0], [1.0, 1.0]), "sharpe_ratio") == 0.0


def test_concat_trades_keeps_none_empty_and_pooled_apart_and_orders_by_exit():
    a = pd.DataFrame({"exit_time": pd.to_datetime(["2024-03-01", "2024-01-01"]), "pnl": [1.0, 2.0]})
    b = pd.DataFrame({"exit_time": pd.to_datetime(["2024-02-01"]), "pnl": [3.0]})
    pooled = _concat_trades([a, None, pd.DataFrame(), b])
    assert pooled["pnl"].tolist() == [2.0, 3.0, 1.0] and pooled["pnl"].sum() == 6.0
    assert _concat_trades([None, None]) is None and _concat_trades([]) is None
    empty = _concat_trades([pd.DataFrame(), pd.DataFrame()])
    assert empty is not None and empty.empty


def test_there_is_one_trade_concatenation_helper_in_the_package():
    root = Path(quantred.__file__).parent
    defs = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_concat_trades"
    ]
    assert defs == ["optimization/walk_forward.py"]


def test_pbo_heuristic_is_formatted_as_a_fraction_not_a_clipped_probability():
    """The IS-best-fold rank-failure rate was formatted like a calibrated probability, so
    a tiny rate printed as "<0.001" and a high one as ">0.999". It is a share of folds."""
    from quantred.optimization.walk_forward import PROBABILITY_METRICS, format_gate_value

    assert "pbo" not in PROBABILITY_METRICS
    assert format_gate_value("pbo", 0.0001) == "0.000"
    assert format_gate_value("pbo", 0.9999) == "1.000"
    assert format_gate_value("pbo", 0.5) == "0.500"
    assert format_gate_value("deflated_sharpe", 0.0001) == "<0.001"  # a real probability still clips


def test_efficiency_is_not_a_ratio_of_two_losses():
    """In-sample -0.5 and out-of-sample -1.0 divide to +2: a fold that lost and then lost
    twice as much read as twice as efficient, and cleared a gate asking for 0.4. The ratio
    is only defined against a positive in-sample score."""
    from quantred.optimization.walk_forward import WalkForwardResult

    v = _validator()
    losing = _folds([-0.5, -0.5], [-1.0, -1.0])
    assert v._efficiency(losing, "sharpe_ratio") == 0.0
    result = WalkForwardResult(losing, {}, None, None, 0.0, {}, 2, "sharpe_ratio")
    assert result.median_efficiency() == 0.0
    # and a losing fold beside a real one no longer drags the median upward
    mixed = WalkForwardResult(_folds([-0.5, 2.0], [-1.0, 1.0]), {}, None, None, 0.0, {}, 2, "sharpe_ratio")
    assert mixed.median_efficiency() == pytest.approx(0.5, rel=1e-12)
