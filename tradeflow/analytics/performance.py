"""Backtest performance accounting - i.e. the part that delivers the bad news.

Turns a table of completed trades plus an equity curve into a metrics dict. This
logic lives here - not inside ``Strategy`` or the engine - so strategies stay
about signals and the engine stays about orchestration (separation of concerns).

The metric *formulas* all live in :mod:`tradeflow.analytics.metrics`; this module only composes them over the trade table
and equity curve.

Caveat: ``build_equity_curve`` accumulates *closed-trade* P&L
resampled to calendar-daily, so intra-trade (mark-to-market) drawdown is not
captured. ``max_drawdown``/``ulcer_index``/volatility therefore understate true
risk during long holds. Consumers (and the metrics glossary) must not over-trust
these as mark-to-market figures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tradeflow.analytics import metrics as m

#: Below this many trades a result is statistically low-power.
LOW_SAMPLE_TRADES = 30

#: The metric keys always present in a results dict (handy for optimizers/reports).
#: Existing keys keep their name, meaning and order (backward compatibility);
#: new keys are appended by tier.
METRIC_KEYS = (
    # --- original headline set ---
    "total_return",
    "buy_hold_return",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "total_trades",
    "win_rate",
    "profit_factor",
    "avg_win",
    "avg_loss",
    "largest_win",
    "largest_loss",
    # --- Tier 1: required for honest evaluation ---
    "cagr",
    "annualized_volatility",
    "max_drawdown_duration",
    "exposure",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    # --- Tier 2: valuable ---
    "recovery_factor",
    "ulcer_index",
    "martin_ratio",
    "sterling_ratio",
    "var_95",
    "var_99",
    "cvar_95",
    "expectancy",
    "payoff_ratio",
    "gain_to_pain_ratio",
    "kelly_criterion",
    "sqn",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "avg_trade_duration",
    "mae_pct",
    "mfe_pct",
    "alpha",
    "beta",
    "r_squared",
    "treynor_ratio",
    "information_ratio",
    "benchmark_buy_hold_return",
    # --- Tier 3: nice-to-have ---
    "skew",
    "kurtosis",
    "downside_deviation",
    "tail_ratio",
    "omega_ratio",
    "best_period",
    "worst_period",
    "turnover",
)

#: Non-numeric flags also always present in a results dict.
#:
#: ``treynor_available`` is separate from ``benchmark_available`` because the two fail
#: independently: a benchmark can be present and Treynor still meaningless, when the
#: book's beta is too near zero to divide by. Without the flag a suppressed ratio and a
#: genuine zero are the same number.
FLAG_KEYS = ("benchmark_available", "low_sample", "treynor_available")


#: Thresholds for the executability verdict. Judgment calls, not evidence-derived:
#: they mark where an executed book stops resembling the one that was validated, and
#: they are exposed so a deployment can set its own. Deliberately **separate** from the
#: promotion gates - those ask whether the edge was real, this asks whether the book
#: can be traded at this capital, and collapsing the two would make one number mean two
#: things and silently redefine `promotable` for every trial already recorded.
DEFAULT_EXECUTION_LIMITS: Dict[str, float] = {
    "max_rounding_drag_pct": 10.0,  # executed notional this far below intended
    "max_unfillable_pct": 5.0,  # this share of intended entries never opened
    "max_cost_share_of_gross_pct": 40.0,  # transaction cost as a share of gross profit
}


def execution_verdict(
    execution: Optional[Dict[str, Any]], limits: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Whether the book the engine actually traded resembles the one it was asked for.

    Whole-share rounding is invisible and scales with how small the account is: the
    same config that drags 0.3% at $100,000 drags 5% at $4,000 and 22% at $500, and
    nothing in the result said so. This turns that into a check with its numbers shown,
    in the house style - every threshold beside its value, no single reassuring summary.

    Returns ``executable: None`` when there is nothing to judge (no entries attempted),
    which is not the same as passing and must not be rendered as one.
    """
    checks: Dict[str, Any] = {}
    if not execution or not (
        execution.get("positions_filled", 0)
        or execution.get("positions_rounded_to_zero", 0)
        or execution.get("positions_below_min_notional", 0)
    ):
        return {"executable": None, "checks": checks, "reason": "no entries were attempted"}

    limit = {**DEFAULT_EXECUTION_LIMITS, **(limits or {})}
    drag = float(execution.get("rounding_drag_pct", 0.0))
    unfillable = float(execution.get("unfillable_pct", 0.0))
    checks["rounding_drag"] = {
        "value": drag,
        "threshold": limit["max_rounding_drag_pct"],
        "passed": drag <= limit["max_rounding_drag_pct"],
        "note": "share rounding removed this much of the intended notional at this capital",
    }
    checks["unfillable_entries"] = {
        "value": unfillable,
        "threshold": limit["max_unfillable_pct"],
        "passed": unfillable <= limit["max_unfillable_pct"],
        "note": "this share of intended entries could not be opened at all",
    }
    # Against *gross profit*, not capital: the same cost is unremarkable against a
    # large gross return and fatal against a small one, and the two denominators
    # disagree most exactly when the answer matters. Skipped rather than guessed when
    # the strategy did not make money gross - there is no edge for cost to eat, and a
    # ratio against a non-positive denominator would be arithmetic rather than a fact.
    # Breadth: not "is the cap small" - a strategy that concentrates in the best five
    # of sixty is a legitimate design - but "was the cap ever chosen". A one-position
    # book selected from many candidates is a different strategy from a many-position
    # one, and 1 is what every shipped strategy declares by default, so this is
    # overwhelmingly a setting nobody changed rather than a decision anybody made.
    max_positions = execution.get("max_positions")
    universe_size = int(execution.get("universe_size") or 0)
    if max_positions is not None and universe_size > 1:
        checks["book_breadth"] = {
            "value": float(max_positions),
            "threshold": 2.0,
            "passed": float(max_positions) > 1,
            "note": (
                f"the book holds at most {max_positions} of {universe_size} candidates - "
                "max_positions is 1, the shipped default; set it to the book you intend"
            ),
        }

    gross_profit = float(execution.get("gross_profit", 0.0))
    if gross_profit > 0:
        cost_share = float(execution.get("total_cost", 0.0)) / gross_profit * 100.0
        checks["cost_share_of_gross"] = {
            "value": cost_share,
            "threshold": limit["max_cost_share_of_gross_pct"],
            "passed": cost_share <= limit["max_cost_share_of_gross_pct"],
            "note": "transaction cost ate this share of the gross profit",
        }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "executable": not failed,
        "checks": checks,
        "reason": "" if not failed else f"{', '.join(failed)} beyond limit at this capital",
    }


#: Prerequisites checked *before* promotion, beside `promotable` rather than inside it.
#:
#: `promotable` stays statistical - median OOS Sharpe, efficiency, drawdown ratio,
#: deflated Sharpe - and keeps meaning for every trial already recorded. These are the
#: questions asked of a candidate that has already cleared those: does the edge survive
#: worse cost assumptions, and does it still look like skill once the whole family of
#: trials is priced in?
#:
#: `min_family_trials` is the interesting one. A family test over two series is
#: arithmetic rather than evidence, so the check does not run at all below the floor -
#: and reports that it did not, rather than passing by default. That is also what makes
#: this safe to add: a campaign with a thin family is told the gate is unevaluated, not
#: told it passed.
DEFAULT_PREREQUISITES: Dict[str, float] = {
    "min_cost_stress_multiple": 3.0,
    "max_family_p": 0.05,
    "min_family_trials": 10,
}


def promotion_prerequisites(
    *,
    cost_stress: Optional[Dict[str, Any]] = None,
    bootstrap: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Checks a candidate should clear before paper, reported beside ``promotable``.

    Every check carries ``evaluated``. A check whose input is absent is *not* a pass:
    a cost curve nobody ran and a family too thin to test are both "unknown", and
    rendering unknown as green is the failure this whole section exists to avoid.

    ``ready`` is ``None`` when nothing could be evaluated, ``False`` when an evaluated
    check failed, ``True`` only when every evaluated check passed - and it never
    implies the unevaluated ones would have.
    """
    limit = {**DEFAULT_PREREQUISITES, **(limits or {})}
    checks: Dict[str, Any] = {}

    survives = (cost_stress or {}).get("survives_to_multiple")
    checks["cost_stress"] = {
        "evaluated": survives is not None,
        "value": float(survives) if survives is not None else None,
        "threshold": limit["min_cost_stress_multiple"],
        "passed": survives is not None and float(survives) >= limit["min_cost_stress_multiple"],
        "note": "the edge survives this multiple of its own assumed cost",
    }

    family = (bootstrap or {}).get("family") or {}
    n_used = int(family.get("n_used") or 0)
    family_p = family.get("family_p")
    enough_family = bool(family.get("available")) and n_used >= limit["min_family_trials"]
    checks["family_bootstrap"] = {
        "evaluated": enough_family and family_p is not None,
        "value": float(family_p) if family_p is not None else None,
        "threshold": limit["max_family_p"],
        "passed": enough_family and family_p is not None and float(family_p) <= limit["max_family_p"],
        "n_used": n_used,
        "note": (
            f"needs {int(limit['min_family_trials'])} usable return-series trials to mean "
            f"anything; {n_used} available"
            if not enough_family
            else "still notable once every trial the campaign tried is priced in"
        ),
    }

    evaluated = [check for check in checks.values() if check["evaluated"]]
    ready = None if not evaluated else all(check["passed"] for check in evaluated)
    return {"ready": ready, "checks": checks, "evaluated": len(evaluated), "total": len(checks)}


def empty_metrics() -> Dict[str, float]:
    """A zeroed metrics dict, returned when no trades occurred."""
    base = {key: 0.0 for key in METRIC_KEYS}
    base["benchmark_available"] = False
    base["low_sample"] = True
    base["treynor_available"] = False
    return base


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    equity_curve: List[float],
    initial_capital: float,
    final_capital: float,
    market_data: Dict[str, Dict[str, float]],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    periods_per_year: int = m.TRADING_DAYS_PER_YEAR,
    benchmark_returns: Optional[pd.Series] = None,
    n_trials: int = 1,
    var_of_trial_sr: Optional[float] = None,
) -> Dict[str, float]:
    """Compute headline performance metrics for a completed backtest.

    Args:
        trades_df: One row per closed trade; must include a ``pnl`` column (and,
            for excursion metrics, ``mae_pct``/``mfe_pct``/``entry_time``/``exit_time``).
        equity_curve: Account equity sampled over the backtest (daily-resampled).
        initial_capital / final_capital: Capital at the start / end.
        market_data: ``{symbol: {"first_open", "last_close"}}`` for buy & hold.
        start / end: Backtest window, used for CAGR and exposure. When absent,
            elapsed time is inferred from the equity-curve length.
        periods_per_year: Annualization factor for the equity-curve returns
            (the curve is daily, so the default 252 is correct; pass a different
            value only for a non-daily series).
        benchmark_returns: Optional return series for alpha/beta/IR. When ``None``
            those keys are ``0.0`` and ``benchmark_available`` is ``False``.
        n_trials: Number of configs evaluated to produce this result (drives the
            Deflated Sharpe). ``1`` for a standalone backtest.
        var_of_trial_sr: Variance of per-period Sharpe across those trials, if
            known (passed by the optimizer/walk-forward); otherwise estimated.

    Returns:
        A flat ``{metric_name: value}`` dict (see :data:`METRIC_KEYS`/:data:`FLAG_KEYS`).
    """
    if trades_df.empty:
        return empty_metrics()

    pnl = trades_df["pnl"]
    returns = m.returns_from_equity(equity_curve)

    winning = trades_df[pnl > 0]["pnl"]
    losing = trades_df[pnl < 0]["pnl"]

    total_return_pct = (final_capital / initial_capital - 1) * 100 if initial_capital else 0.0
    net_profit = final_capital - initial_capital
    drawdown_frac = m.max_drawdown(equity_curve)
    drawdown_pct = drawdown_frac * 100
    drawdown_dollars = _max_drawdown_dollars(equity_curve)
    years = _elapsed_years(start, end, equity_curve)
    cagr_frac = m.cagr(equity_curve, years)

    has_benchmark = benchmark_returns is not None and len(benchmark_returns) > 0
    benchmark_buy_hold = 0.0
    if has_benchmark:
        alpha, beta, r_squared = m.alpha_beta(returns, benchmark_returns)
        info_ratio = m.information_ratio(returns, benchmark_returns, periods_per_year)
        treynor = m.treynor_ratio(returns, beta, periods_per_year)
        # Compounded over exactly the steps the strategy was measured on, rather than
        # the benchmark's own first/last close: the two differ whenever the curve does
        # not start on the benchmark's first bar, and the comparison is only fair over
        # one window.
        benchmark_buy_hold = float(((1.0 + pd.Series(benchmark_returns).dropna()).prod() - 1.0) * 100)
    else:
        alpha = beta = r_squared = info_ratio = treynor = 0.0

    result = {
        # --- original headline set ---
        "total_return": total_return_pct,
        "buy_hold_return": _buy_hold_return(market_data),
        "benchmark_buy_hold_return": benchmark_buy_hold,
        "sharpe_ratio": m.sharpe_ratio(returns, periods_per_year),
        "sortino_ratio": m.sortino_ratio(returns, periods_per_year),
        "calmar_ratio": m.calmar_ratio(cagr_frac, drawdown_frac),
        "max_drawdown": drawdown_pct,
        "total_trades": int(len(trades_df)),
        "win_rate": m.win_rate(pnl) * 100,
        "profit_factor": m.profit_factor(pnl),
        "avg_win": float(winning.mean()) if not winning.empty else 0.0,
        "avg_loss": float(abs(losing.mean())) if not losing.empty else 0.0,
        "largest_win": float(winning.max()) if not winning.empty else 0.0,
        "largest_loss": float(abs(losing.min())) if not losing.empty else 0.0,
        # --- Tier 1 ---
        "cagr": cagr_frac * 100,
        "annualized_volatility": m.annualized_volatility(returns, periods_per_year) * 100,
        "max_drawdown_duration": m.max_drawdown_duration(equity_curve),
        "exposure": _exposure(trades_df, start, end) * 100,
        "probabilistic_sharpe_ratio": m.probabilistic_sharpe_ratio(returns),
        "deflated_sharpe_ratio": m.deflated_sharpe_ratio(returns, n_trials, var_of_trial_sr),
        # --- Tier 2 ---
        "recovery_factor": m.recovery_factor(net_profit, drawdown_dollars),
        "ulcer_index": m.ulcer_index(equity_curve),
        "martin_ratio": m.martin_ratio(cagr_frac, m.ulcer_index(equity_curve)),
        "sterling_ratio": m.sterling_ratio(cagr_frac, _average_drawdown(equity_curve)),
        "var_95": m.value_at_risk(returns, 0.95) * 100,
        "var_99": m.value_at_risk(returns, 0.99) * 100,
        "cvar_95": m.conditional_var(returns, 0.95) * 100,
        "expectancy": m.expectancy(pnl),
        "payoff_ratio": m.payoff_ratio(pnl),
        "gain_to_pain_ratio": m.gain_to_pain_ratio(pnl),
        "kelly_criterion": m.kelly_criterion(pnl),
        "sqn": m.system_quality_number(pnl),
        "max_consecutive_wins": m.consecutive(pnl, winning=True),
        "max_consecutive_losses": m.consecutive(pnl, winning=False),
        "avg_trade_duration": _avg_trade_duration(trades_df),
        "mae_pct": float(trades_df["mae_pct"].mean()) if "mae_pct" in trades_df else 0.0,
        "mfe_pct": float(trades_df["mfe_pct"].mean()) if "mfe_pct" in trades_df else 0.0,
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "treynor_ratio": treynor,
        "information_ratio": info_ratio,
        # --- Tier 3 ---
        "skew": m.skewness(returns),
        "kurtosis": m.kurtosis(returns),
        "downside_deviation": m.downside_deviation(returns, periods_per_year) * 100,
        "tail_ratio": m.tail_ratio(returns),
        "omega_ratio": m.omega_ratio(returns),
        "best_period": float(returns.max() * 100) if not returns.empty else 0.0,
        "worst_period": float(returns.min() * 100) if not returns.empty else 0.0,
        "turnover": _turnover(trades_df, initial_capital),
        # --- flags ---
        "benchmark_available": bool(has_benchmark),
        "treynor_available": bool(has_benchmark and abs(beta) >= m.MIN_ABS_BETA_FOR_TREYNOR),
        "low_sample": bool(len(trades_df) < LOW_SAMPLE_TRADES),
    }
    return result


def build_equity_curve(trades_df: pd.DataFrame, initial_capital: float) -> List[float]:
    """Build a daily equity curve by accumulating trade P&L at exit time."""
    dated = build_dated_equity_curve(trades_df, initial_capital)
    return dated.tolist() if not dated.empty else [initial_capital]


def build_dated_equity_curve(trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Like :func:`build_equity_curve`, but keeps the ``DatetimeIndex``.

    ``build_equity_curve`` returns a plain ``List[float]`` because nothing on its
    existing call paths ever needed the dates back. The per-trial OOS
    return series does: the Reality Check inner-joins trials on a common calendar
    (`src.store.trials.TrialStore.returns_panel`), which needs real dates, not
    just an ordered sequence.
    """
    if trades_df.empty or "exit_time" not in trades_df:
        return pd.Series(dtype="float64")
    daily_pnl = trades_df.set_index("exit_time")["pnl"].resample("D").sum().fillna(0)
    if daily_pnl.empty:
        return pd.Series(dtype="float64")
    equity = initial_capital + daily_pnl.cumsum()
    first_day = equity.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series([initial_capital], index=[first_day]), equity])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _elapsed_years(start: Optional[datetime], end: Optional[datetime], equity_curve: List[float]) -> float:
    """Elapsed calendar years for CAGR - from the window if known, else the curve."""
    if start is not None and end is not None:
        days = (end - start).days
        return days / 365.25 if days > 0 else 0.0
    # Fall back to the daily-resampled curve length (one point per calendar day).
    return max(len(equity_curve) - 1, 0) / 365.25


def _exposure(trades_df: pd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> float:
    """Fraction of the window with at least one open position (union of holds)."""
    if trades_df.empty or "entry_time" not in trades_df or "exit_time" not in trades_df:
        return 0.0
    if start is None or end is None:
        return 0.0
    total = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    if total <= 0:
        return 0.0

    intervals = sorted(
        (pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(trades_df["entry_time"], trades_df["exit_time"])
    )
    # Normalize timezone so subtraction against tz-naive start/end is valid.
    covered = 0.0
    cur_start, cur_end = None, None
    for a, b in intervals:
        a, b = a.tz_localize(None) if a.tzinfo else a, b.tz_localize(None) if b.tzinfo else b
        if cur_end is None:
            cur_start, cur_end = a, b
        elif a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            covered += (cur_end - cur_start).total_seconds()
            cur_start, cur_end = a, b
    if cur_end is not None:
        covered += (cur_end - cur_start).total_seconds()
    return min(covered / total, 1.0)


def _max_drawdown_dollars(equity_curve: List[float]) -> float:
    """Largest peak-to-trough decline of the equity curve in account currency."""
    equity = pd.Series(equity_curve, dtype="float64")
    if equity.empty:
        return 0.0
    return float((equity.cummax() - equity).max())


def _average_drawdown(equity_curve: List[float]) -> float:
    """Mean depth of the underwater periods (fraction), for the Sterling ratio."""
    dd = m.drawdown_series(equity_curve)
    underwater = dd[dd < 0]
    return float(abs(underwater.mean())) if not underwater.empty else 0.0


def _avg_trade_duration(trades_df: pd.DataFrame) -> float:
    """Mean holding period in hours."""
    if "entry_time" not in trades_df or "exit_time" not in trades_df:
        return 0.0
    durations = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])
    seconds = durations.dt.total_seconds()
    return float(seconds.mean() / 3600.0) if not seconds.empty else 0.0


def _turnover(trades_df: pd.DataFrame, initial_capital: float) -> float:
    """Traded notional / capital - a proxy for fee/slippage drag."""
    if "entry_price" not in trades_df or "size" not in trades_df or not initial_capital:
        return 0.0
    notional = (trades_df["entry_price"] * trades_df["size"]).sum()
    return float(notional / initial_capital)


def _buy_hold_return(market_data: Dict[str, Dict[str, float]]) -> float:
    """Average buy & hold return (%) across the traded symbols."""
    per_symbol = [
        (data["last_close"] / data["first_open"] - 1) * 100
        for data in market_data.values()
        if data.get("first_open")
    ]
    return float(np.mean(per_symbol)) if per_symbol else 0.0
