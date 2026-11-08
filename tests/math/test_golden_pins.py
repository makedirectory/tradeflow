"""Golden pins: end-to-end deterministic scenarios whose full numeric output is stored.

Each scenario first checks its output against the reference identities (so a wrong number
is never frozen), then compares every key with ``tests/math/golden/<scenario>.json``. See
``tests/math/pins.py`` for the comparison, the failure message and the regeneration path.

Tolerance is ``rel 1e-9 / abs 1e-12`` unless a scenario says otherwise and gives the
reason.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantred.alphas import combine, horizon
from quantred.alphas.base import Alpha
from quantred.analytics import attribution as attr
from quantred.analytics import information as info
from quantred.analytics import metrics as m
from quantred.costs import ParametricCostModel
from quantred.engine.backtest import ACCOUNTING_VERSION, BacktestEngine
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator
from quantred.portfolio import policy
from quantred.portfolio.optimizer import CostInputs, MeanVarianceOptimizer
from quantred.portfolio.posterior import black_litterman_from_ic
from quantred.risk.base import RiskMatrix, build_risk_matrix
from quantred.risk.conditional import ewma_variance_path
from quantred.risk.factor import estimate_factor_model
from quantred.risk.sample import LedoitWolfCovariance, SampleCovariance
from quantred.services import analysis
from quantred.services.registry import STRATEGIES
from quantred.strategies import signals
from tests.fakes import DictMarketData, FakeMarketData
from tests.math import pins
from tests.math.test_engine_accounting import ENTRY_BAR, _bars, _hold_world, _PerSymbolStrategy, _run
from tests.test_sizing_parity import CAPITAL as SIZING_CAPITAL
from tests.test_sizing_parity import _backtest_notionals, _live_notionals

SYMS = ["AAA", "BBB", "CCC"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)
AS_OF = datetime(2024, 6, 1)
START_GAP, END_GAP = datetime(2024, 1, 2), datetime(2024, 1, 31)
B, H = signals.BUY, signals.HOLD


def _client(extra=("SPY",)):
    return MarketDataClient(FakeMarketData([*SYMS, *extra], n=600, freq="1D"))


def _independent_backtest_checks(result):
    """Identities every pinned backtest must satisfy before its numbers are stored."""
    t = result.trades
    assert result.final_capital == pytest.approx(result.initial_capital + t["pnl"].sum(), abs=1e-6)
    assert result.equity_curve[-1] == pytest.approx(result.final_capital, abs=1e-6)
    eq = np.array(result.equity_curve)
    r = eq[1:] / eq[:-1] - 1
    assert result.metrics["sharpe_ratio"] == pytest.approx(
        math.sqrt(252) * r.mean() / r.std(ddof=1), rel=1e-9
    )
    assert result.metrics["max_drawdown"] == pytest.approx(
        100 * max((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)), rel=1e-9
    )
    assert result.total_cost == pytest.approx(t["cost"].sum(), rel=1e-12)


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def test_golden_backtest_daily_with_costs_and_benchmark():
    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    result = BacktestEngine(strategy, _client(), cost_model=ParametricCostModel()).run(
        SYMS, START, END, 100_000.0, benchmark="SPY"
    )
    _independent_backtest_checks(result)
    assert result.metrics["benchmark_available"] and len(result.trades) > 10
    values = pins.numeric_values(
        {
            "metrics": result.metrics,
            "final_capital": result.final_capital,
            "total_cost": result.total_cost,
            "gross_final_capital": result.gross_final_capital,
            "execution": result.execution,
            "n_trades": len(result.trades),
            "trade_pnl": result.trades["pnl"].tolist(),
            "equity_tail": result.equity_curve[-5:],
        }
    )
    pins.check(
        "backtest_daily",
        "demo_trend, 3 symbols, daily, default costs, SPY benchmark",
        "fake bars seeds 0..3",
        values,
    )


def test_golden_backtest_holding_the_benchmark():
    frame = _hold_world()
    overrides = {
        "stop_loss": 0.5,
        "take_profit": 50.0,
        "position_limits": {"max_positions": 1, "max_position_size": 1000.0, "max_total_risk": 1.0},
    }
    result = _run(
        {"AAA": frame, "BMK": frame.copy()},
        ["BUY"] + ["HOLD"] * 39,
        overrides,
        capital=1000.0,
        benchmark="BMK",
    )
    # the reference identities for this scenario, before anything is stored
    assert result.metrics["beta"] == pytest.approx(1.0, abs=1e-9)
    assert result.metrics["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert result.metrics["alpha"] == pytest.approx(0.0, abs=1e-12)
    _independent_backtest_checks(result)
    metrics = result.metrics
    pins.check(
        "backtest_hold_benchmark",
        "one long held to the end on the benchmark's own bars, whole capital",
        "seed 5",
        pins.numeric_values({"metrics": metrics, "final_capital": result.final_capital}),
    )


def test_golden_backtest_gapping_through_every_kind_of_exit():
    """A long stop, a short stop, a long target and a short target each gap through their
    level on a different bar. Guards the stop-fills-at-the-level defect, which a gap
    exposes: the exit price must be the open that gapped past the level, never the level."""
    # 10 shares entered at 100: stop and target are 10% either side (90/110 long, 110/90 short)
    cases = {  # symbol -> (entry signal, bar index of the gap, gap bar, expected exit, reason)
        "AAA": (B, 2, (80, 82, 78, 79), 80.0, "STOP_LOSS"),
        "BBB": (signals.SELL, 3, (125, 126, 122, 124), 125.0, "STOP_LOSS"),
        "CCC": (B, 4, (115, 118, 114, 116), 115.0, "TAKE_PROFIT"),
        "DDD": (signals.SELL, 5, (85, 86, 83, 84), 85.0, "TAKE_PROFIT"),
    }
    n_bars = 8
    frames, scripts = {}, {}
    for sym, (entry, at, gap, _, _) in cases.items():
        rows = [ENTRY_BAR] * n_bars
        rows[at] = gap
        frames[sym] = _bars(rows)
        scripts[sym] = [entry] + [H] * (n_bars - 1)
    strategy = _PerSymbolStrategy(
        scripts,
        {"position_limits": {"max_positions": 4, "max_position_size": 1_000.0, "max_total_risk": 1.0}},
    )
    client = MarketDataClient(DictMarketData(frames))
    result = BacktestEngine(strategy, client).run(list(frames), START_GAP, END_GAP, 100_000.0)
    _independent_backtest_checks(result)
    # the reference answer, by arithmetic, for each of the four exits
    trades = result.trades.set_index("symbol")
    assert len(trades) == 4
    for sym, (entry, _, _, price, reason) in cases.items():
        trade = trades.loc[sym]
        direction = 1.0 if entry == B else -1.0
        assert trade["exit_reason"] == reason and trade["exit_price"] == pytest.approx(price)
        assert trade["pnl"] == pytest.approx((price - 100.0) * trade["size"] * direction)
    pins.check(
        "backtest_gap_through",
        "four resting exits (long/short stop, long/short target) each gapped through, scripted bars",
        "hand-built bars, no randomness",
        pins.numeric_values(
            {
                "metrics": result.metrics,
                "final_capital": result.final_capital,
                "trades": {
                    sym: {k: trades.loc[sym][k] for k in ("size", "entry_price", "exit_price", "pnl")}
                    for sym in cases
                },
                "equity_curve": result.equity_curve,
            }
        ),
    )


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def test_golden_walk_forward_four_folds():
    validator = WalkForwardValidator(
        STRATEGIES["demo_trend"], _client(), initial_capital=100_000.0, seed=42, benchmark="SPY"
    )
    wf = validator.run(SYMS, START, END, n_folds=4, method="grid", max_evals=4, monte_carlo=True, pbo=True)
    assert len(wf.folds) >= 3 and wf.pbo is not None and wf.monte_carlo["samples"] > 0
    agg = wf.oos_aggregate
    pooled = wf.oos_trade_table
    # the aggregate Sharpe is a trading-day Sharpe of the pooled OOS P&L
    r = wf.oos_returns.to_numpy()
    assert agg["sharpe_ratio"] == pytest.approx(math.sqrt(252) * r.mean() / r.std(ddof=1), rel=1e-9)
    assert len(pooled) == wf.total_oos_trades()
    assert wf.monte_carlo["sharpe_p05"] <= wf.monte_carlo["sharpe_p50"]
    values = pins.numeric_values(
        {
            "oos_aggregate": agg,
            "monte_carlo": wf.monte_carlo,
            "pbo": wf.pbo,
            "efficiency": wf.efficiency,
            "degradation": wf.degradation,
            "n_trials_total": wf.n_trials_total,
            "folds": [
                {
                    "is": f.is_metrics,
                    "oos": f.oos_metrics,
                    "oos_trades": f.oos_trades,
                    "params": f.is_best_params,
                }
                for f in wf.folds
            ],
            "median_oos_sharpe": wf.median_oos("sharpe_ratio"),
            "median_efficiency": wf.median_efficiency(),
        }
    )
    pins.check(
        "walk_forward",
        "demo_trend walk-forward, 4 anchored folds, grid, MC and PBO on",
        "seed 42, bars 0..3",
        values,
    )


def test_golden_cost_stress_sweep():
    report = analysis.run_cost_stress(
        _client(),
        "demo_trend",
        SYMS,
        START,
        END,
        capital=100_000.0,
        impact_eta=0.0,
        borrow_bps=0.0,
        multiples=(1.0, 2.0, 4.0),
    )
    cost = {p["multiple"]: p["total_cost"] for p in report["points"]}
    assert cost[2.0] / cost[1.0] == pytest.approx(2.0, rel=0.05)  # linear terms only
    assert cost[4.0] / cost[1.0] == pytest.approx(4.0, rel=0.08)
    values = pins.numeric_values(
        {"points": report["points"], "survives": report.get("survives_multiple", None)}, drop=("run_id",)
    )
    pins.check(
        "cost_stress",
        "demo_trend cost stress at 1x, 2x, 4x, impact and borrow off",
        "fake bars seeds 0..3",
        values,
    )


# --------------------------------------------------------------------------- #
# Risk models
# --------------------------------------------------------------------------- #
def _risk_world():
    rng = np.random.default_rng(2024)
    t, n = 400, 8
    names = [f"R{i}" for i in range(n)]
    market = rng.normal(0.0002, 0.009, (t, 1))
    style = rng.normal(size=(n, 2))
    style = (style - style.mean(0)) / style.std(0)
    factor = rng.normal(0, 0.004, (t, 2))
    returns = market * rng.uniform(0.7, 1.3, n) + factor @ style.T + rng.normal(0, 0.008, (t, n))
    return pd.DataFrame(returns, columns=names), pd.DataFrame(
        style, index=names, columns=["momentum", "size"]
    )


def test_golden_risk_models():
    returns, style = _risk_world()
    factor = estimate_factor_model(returns, style, 252.0)
    lw_sigma, delta = LedoitWolfCovariance().estimate(returns)
    sample_sigma, _ = SampleCovariance().estimate(returns)
    ew = np.full(8, 1 / 8)

    def mean_corr(sigma):
        d = np.sqrt(np.diag(sigma))
        c = sigma / np.outer(d, d)
        return (c.sum() - 8) / 56

    # identities: the factor/specific split adds up; shrinkage keeps variances; the sample
    # annualises to np.cov
    assert factor.factor_variance(ew) + factor.specific_variance(ew) == pytest.approx(
        factor.variance(ew), rel=1e-10
    )
    assert np.diag(lw_sigma) == pytest.approx(np.diag(sample_sigma), rel=1e-12)
    assert sample_sigma == pytest.approx(np.cov(returns.to_numpy(), rowvar=False, bias=True), rel=1e-10)
    assert mean_corr(factor.sigma) == pytest.approx(
        mean_corr(sample_sigma), abs=0.08
    )  # one world, two estimators

    bars = {c: pd.DataFrame({"close": 100 * np.cumprod(1 + returns[c].to_numpy())}) for c in returns}
    annual = build_risk_matrix(LedoitWolfCovariance(), bars, 252.0, min_obs=60)
    values = pins.numeric_values(
        {
            "factor": {
                "factor_names": factor.factor_names,
                "factor_cov": factor.factor_cov,
                "specific_var": factor.specific_var,
                "ew_vol": factor.volatility(ew),
                "ew_factor_variance": factor.factor_variance(ew),
                "ew_specific_variance": factor.specific_variance(ew),
                "mean_corr": mean_corr(factor.sigma),
            },
            "shrinkage": {
                "delta": delta,
                "ew_vol": math.sqrt(ew @ (lw_sigma * 252) @ ew),
                "mean_corr": mean_corr(lw_sigma),
                "sigma_row0": (lw_sigma * 252)[0],
                "annualised_delta": annual.shrinkage,
            },
            "sample": {
                "ew_vol": math.sqrt(ew @ (sample_sigma * 252) @ ew),
                "mean_corr": mean_corr(sample_sigma),
                "sigma_row0": (sample_sigma * 252)[0],
            },
            "ewma_variance": ewma_variance_path(returns, 0.94, 60).to_numpy(),
        }
    )
    pins.check(
        "risk_models",
        "8 names, one market + 2 style factors + noise; factor, Ledoit-Wolf, sample",
        "seed 2024",
        values,
    )


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def test_golden_optimizer_runs():
    rng = np.random.default_rng(77)
    n = 8
    syms = [f"N{i}" for i in range(n)]
    loadings = rng.normal(0, 0.2, (n, 3))
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.02, 0.06, n))
    alpha = rng.normal(0.0, 0.03, n)
    alpha[0], alpha[1] = 0.12, 0.10
    risk = RiskMatrix(syms, sigma)
    alphas = [Alpha(s, float(a), AS_OF, 0.2, 0.05, 0.0) for s, a in zip(syms, alpha)]
    ir_star = math.sqrt(alpha @ np.linalg.solve(sigma, alpha))

    capped = MeanVarianceOptimizer(max_weight=0.2).optimize(alphas, risk, risk_aversion=2.0)
    w = np.array([capped.weights.get(s, 0.0) for s in syms])
    d = capped.diagnostics
    # reference identities: the covariance-metric TC, tc * IR* = predicted IR, the cap binds
    assert d["transfer_coefficient"] == pytest.approx(
        alpha @ w / (ir_star * math.sqrt(w @ sigma @ w)), rel=1e-9
    )
    assert d["transfer_coefficient"] * ir_star == pytest.approx(d["predicted_ir"], rel=1e-9)
    assert w.max() == pytest.approx(0.2, abs=1e-6) and w.sum() == pytest.approx(1.0, abs=1e-8)

    cost = MeanVarianceOptimizer(max_weight=0.3).optimize(
        alphas,
        risk,
        target_te=0.04,
        current_weights={s: 1 / n for s in syms},
        cost_model=ParametricCostModel(),
        cost_inputs=CostInputs(
            spread={s: 0.0006 for s in syms},
            adv_dollar={s: 3e6 for s in syms},
            daily_vol={s: 0.02 for s in syms},
        ),
        capital=500_000.0,
        holding_period_years=1 / 12,
    )
    assert cost.diagnostics["risk_aversion"] == pytest.approx(ir_star / (2 * 0.04), rel=1e-12)
    bench = MeanVarianceOptimizer(max_weight=1.0).optimize(
        alphas, risk, risk_aversion=4.0, benchmark_weights={s: 1 / n for s in syms}
    )
    # the unconstrained active book carries no benchmark beta (alpha was neutralised); a
    # binding long-only floor is what moves the held book off it
    beta_vec = sigma @ np.full(n, 1 / n) / (np.full(n, 1 / n) @ sigma @ np.full(n, 1 / n))
    unconstrained_active = np.array([bench.unconstrained_weights[s] for s in syms]) - 1 / n
    assert beta_vec @ unconstrained_active == pytest.approx(0.0, abs=1e-9)

    def pack(res):
        return {
            "weights": res.weights,
            "unconstrained": res.unconstrained_weights,
            "diagnostics": res.diagnostics,
        }

    values = pins.numeric_values(
        {"capped": pack(capped), "cost_aware": pack(cost), "benchmark": pack(bench), "ir_star": ir_star}
    )
    # The projection is an iterated bisection with a 1e-10 stopping tolerance, so the stored
    # weights are only stable to ~1e-8 across BLAS builds; diagnostics inherit that.
    pins.check(
        "optimizer",
        "8 names, 3-factor covariance; capped long-only, cost-aware at target TE, benchmark-aware",
        "seed 77",
        values,
        loose={"capped": (1e-6, 1e-9), "cost_aware": (1e-6, 1e-9), "benchmark": (1e-6, 1e-9)},
    )


# --------------------------------------------------------------------------- #
# Two-clock sizing and the posterior / trading-rate closed forms
# --------------------------------------------------------------------------- #
def test_golden_two_clock_sizing_and_closed_forms():
    backtest, live = _backtest_notionals(), _live_notionals(capital=SIZING_CAPITAL)
    assert backtest == live == [2500.0, 1850.0, 1400.0]  # worked by hand in test_sizing_math.py
    kappa = policy.derive_kappa(1.0, 0.04, 0.5)
    risk = RiskMatrix(
        ["A", "B", "C"], np.array([[0.09, 0.02, 0.01], [0.02, 0.04, 0.015], [0.01, 0.015, 0.0625]])
    )
    post = black_litterman_from_ic({"A": 0.05, "C": -0.03}, risk, ic=0.06, t_eff=120.0)
    # the calibration identity holds for a single view; here two views just must be finite
    assert all(np.isfinite(v) for v in post.mu_post.values())
    values = pins.numeric_values(
        {
            "backtest_notionals": backtest,
            "live_notionals": live,
            "kappa": kappa,
            "discount_phi_0_3": policy.discount_factor(kappa, 0.3),
            "trading_half_life": policy.trading_half_life(kappa),
            "posterior": post.mu_post,
            "view_omega": post.omega,
            "tau": post.tau,
        }
    )
    pins.check(
        "sizing_and_closed_forms",
        "three-entry sizing sequence on both clocks; kappa; Black-Litterman",
        "seed 0 / closed form",
        values,
    )


# --------------------------------------------------------------------------- #
# Information, breadth and return-series statistics
# --------------------------------------------------------------------------- #
def test_golden_information_and_series_report():
    rng = np.random.default_rng(99)
    ics = list(rng.normal(0.04, 0.08, 36))
    market = pd.Series(rng.normal(0, 0.01, 500))
    names = pd.DataFrame(
        {f"S{i}": market * rng.uniform(0.8, 1.2) + rng.normal(0, 0.01, 500) for i in range(6)}
    )
    rho = info.mean_residual_correlation(names, market)
    breadth = info.effective_breadth(6, 12, rho)
    stats_ = info.ic_stats(ics)
    # identities
    assert breadth["br_eff"] == pytest.approx(72 / (1 + 5 * rho), rel=1e-12)
    assert stats_["ic_tstat"] == pytest.approx(
        stats_["mean_ic"] / (stats_["ic_vol"] / 6), rel=1e-12
    )  # sqrt(36)
    decay = horizon.fit_decay({n: 0.08 * 0.85**n for n in range(7)})
    assert decay["delta"] == pytest.approx(0.85, rel=1e-9)
    omega = np.array([[1, 0.4, 0.1], [0.4, 1, 0.3], [0.1, 0.3, 1]])
    w_sig = combine.combination_weights(np.array([0.05, 0.04, 0.02]), omega, ridge=0.0)
    assert w_sig == pytest.approx(np.linalg.solve(omega, [0.05, 0.04, 0.02]), rel=1e-12)

    series = rng.standard_t(5, 750) * 0.01 + 0.0004
    bench = series * 0.6 + rng.normal(0, 0.006, 750)
    alpha_, beta_, r2 = m.alpha_beta(series, bench)
    assert beta_ == pytest.approx(np.polyfit(bench, series, 1)[0], rel=1e-9)
    eq = 100_000 * np.cumprod(1 + series)
    series_report = {
        "sharpe": m.sharpe_ratio(series),
        "sortino": m.sortino_ratio(series),
        "downside_deviation": m.downside_deviation(series),
        "volatility": m.annualized_volatility(series),
        "skew": m.skewness(series),
        "kurtosis": m.kurtosis(series),
        "var95": m.value_at_risk(series),
        "cvar95": m.conditional_var(series),
        "tail_ratio": m.tail_ratio(series),
        "omega": m.omega_ratio(series),
        "psr": m.probabilistic_sharpe_ratio(series),
        "dsr_25_trials": m.deflated_sharpe_ratio(series, 25, 4e-4),
        "expected_max_sharpe_25": m.expected_max_sharpe(4e-4, 25),
        "max_drawdown": m.max_drawdown(eq),
        "drawdown_duration": m.max_drawdown_duration(eq),
        "ulcer": m.ulcer_index(eq),
        "alpha": alpha_,
        "beta": beta_,
        "r_squared": r2,
        "information_ratio": m.information_ratio(series, bench),
    }
    assert all(math.isfinite(v) for v in series_report.values())
    idx = [f"S{i}" for i in range(30)]
    x = pd.DataFrame(rng.normal(size=(30, 2)), index=idx, columns=["m", "mom"])
    period = attr.attribute_period(
        pd.Series(rng.normal(0, 0.03, 30), index=idx),
        x,
        pd.Series(rng.normal(0, 0.03, 30), index=idx),
        pd.Series(rng.uniform(0.8, 1.2, 30), index=idx),
        0.01,
    )
    assert abs(period.residual) < 1e-14
    values = pins.numeric_values(
        {
            "ic_stats": stats_,
            "breadth": {**breadth, "predicted_ir": info.predicted_ir(stats_["mean_ic"], breadth["br_eff"])},
            "ir_standard_error_3y": info.ir_standard_error(0.5, 3.0),
            "decay": decay,
            "blend": horizon.blend_weights(0.8, 0.5),
            "combination_weights": w_sig,
            "combined_ic": combine.combined_ic(np.array([0.05, 0.04, 0.02]), omega),
            "series": series_report,
            "attribution": period.__dict__,
        }
    )
    pins.check(
        "information_report",
        "fixed IC history, residual-correlation breadth, decay, blend, series statistics, attribution",
        "seed 99",
        values,
    )


# --------------------------------------------------------------------------- #
# The pins themselves
# --------------------------------------------------------------------------- #
def test_every_golden_file_was_generated_under_the_current_accounting_version():
    """A bump of ACCOUNTING_VERSION must be followed by a deliberate regeneration."""
    files = sorted(pins.GOLDEN_DIR.glob("*.json"))
    assert files, "no golden files: generate them with QUANTRED_REGEN_GOLDEN=1"
    for path in files:
        header = json.loads(path.read_text())["provenance"]
        assert header["accounting_version"] == ACCOUNTING_VERSION, (
            f"{path.name}: generated under {header['accounting_version']}, code is at {ACCOUNTING_VERSION}. "
            + pins.CHANGED_MESSAGE
        )
        assert {"scenario", "seed", "generated_by"} <= set(header)


def test_a_changed_pin_fails_with_old_and_new_values_and_the_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "GOLDEN_DIR", tmp_path)
    monkeypatch.setenv(pins.REGEN_ENV, "1")
    pins.check("probe", "probe scenario", 0, {"a": 1.0, "b": 2.0, "c": True, "d": float("inf")})
    monkeypatch.delenv(pins.REGEN_ENV)
    pins.check("probe", "probe scenario", 0, {"a": 1.0 + 1e-12, "b": 2.0, "c": True, "d": float("inf")})
    with pytest.raises(pytest.fail.Exception) as err:
        pins.check("probe", "probe scenario", 0, {"a": 1.0, "b": 2.0001, "c": False, "e": 5})
    text = str(err.value)
    assert "b: 2.0 -> 2.0001" in text and "c: True -> False" in text
    assert "d: {'nonfinite': 'inf'} -> (missing)" in text and "e: (new) -> 5" in text
    assert "a pinned number changed: confirm the new value is correct, bump ACCOUNTING_VERSION" in text
    assert "regenerate deliberately" in text and "a:" not in text.replace("a pinned", "")


def test_a_golden_file_from_another_accounting_version_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "GOLDEN_DIR", tmp_path)
    monkeypatch.delenv(pins.REGEN_ENV, raising=False)
    (tmp_path / "old.json").write_text(
        json.dumps({"provenance": {"accounting_version": ACCOUNTING_VERSION - 1}, "values": {"a": 1.0}})
    )
    with pytest.raises(AssertionError, match="regenerate deliberately"):
        pins.check("old", "s", 0, {"a": 1.0})
