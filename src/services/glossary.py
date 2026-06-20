"""Metric glossary.

Definitions + pitfalls for every metric the analytics layer reports, written for
an LLM reader so it doesn't over-trust in-sample numbers. Surfaces the two traps
that matter most: the equity-curve fidelity caveat and the
multiple-testing correction the Deflated Sharpe applies.
"""

from typing import Any, Dict

from src.analytics.performance import FLAG_KEYS, METRIC_KEYS

#: Cross-cutting caveats the agent should keep in mind for *all* metrics.
GLOBAL_CAVEATS = [
    "Equity curve is built from CLOSED-trade P&L resampled to daily: "
    "intra-trade (mark-to-market) drawdown is invisible, so max_drawdown / ulcer_index / "
    "volatility UNDERSTATE true risk during long holds.",
    "In-sample metrics from run_optimization are NOT evidence of edge - they are the "
    "by-product of selecting the best of many configs. Validate with run_walk_forward, "
    "which scores out-of-sample and applies the Deflated Sharpe.",
    "Any metric computed on fewer than ~30 trades is low-power (see the low_sample flag). "
    "A great Sharpe on 4 trades is noise.",
]

_DEFS: Dict[str, str] = {
    "total_return": "Final/initial capital - 1, in %. Period-dependent; use cagr to compare windows of different length.",
    "buy_hold_return": "Average %% return of simply holding the traded symbols over the window (the baseline to beat).",
    "sharpe_ratio": "Annualised mean/std of returns (rf=0). The headline ratio. Inflated by short samples and fat tails - read probabilistic_sharpe_ratio alongside it.",
    "sortino_ratio": "Like Sharpe but only penalises downside deviation.",
    "calmar_ratio": "CAGR / max drawdown. Reward per unit of worst peak-to-trough loss.",
    "max_drawdown": "Largest peak-to-trough equity decline, %% (positive). UNDERSTATED - closed-trade curve only.",
    "total_trades": "Number of closed trades. The sample-size guard; <30 => low_sample.",
    "win_rate": "%% of trades with positive P&L. High win rate can still lose money if losers are large (see payoff_ratio).",
    "profit_factor": "Gross profit / gross loss. >1 is profitable; inf means no losing trades (often a tiny sample).",
    "avg_win": "Mean P&L of winning trades ($).",
    "avg_loss": "Mean P&L of losing trades, magnitude ($).",
    "largest_win": "Best single trade ($).",
    "largest_loss": "Worst single trade, magnitude ($).",
    "cagr": "Compound annual growth rate, %%. The cross-window comparison number (annualised).",
    "annualized_volatility": "Std of returns scaled to a year, %%.",
    "max_drawdown_duration": "Longest run of consecutive periods spent underwater (below a prior peak).",
    "exposure": "%% of the window with at least one open position. A Sharpe earned at 5%% exposure != one at 95%%.",
    "probabilistic_sharpe_ratio": "P(true Sharpe > 0) given sample length, skew and kurtosis. Corrects the Sharpe for short, fat-tailed samples. 0..1.",
    "deflated_sharpe_ratio": "Probabilistic Sharpe against the EXPECTED-BEST Sharpe of n_trials configs. The anti-overfitting metric: the more configs tried, the higher the bar. 0..1; >0.5 is encouraging.",
    "recovery_factor": "Net profit / max drawdown ($). Absolute-dollar cousin of Calmar.",
    "ulcer_index": "RMS drawdown depth (%%) - penalises deep AND long drawdowns. Lower is better.",
    "martin_ratio": "CAGR / ulcer index (Ulcer Performance Index).",
    "sterling_ratio": "CAGR / average drawdown.",
    "var_95": "Historical 95%% Value-at-Risk of period returns (%%, positive loss).",
    "var_99": "Historical 99%% Value-at-Risk (%%).",
    "cvar_95": "Expected shortfall: mean loss in the worst 5%% tail (%%).",
    "expectancy": "Mean P&L per trade ($).",
    "payoff_ratio": "Average win / average loss (magnitudes).",
    "gain_to_pain_ratio": "Sum of P&L / sum of absolute losses.",
    "kelly_criterion": "Theoretical optimal capital fraction from win rate and payoff. Usually scaled down in practice.",
    "sqn": "System Quality Number (Van Tharp): sqrt(n)*mean(pnl)/std(pnl). Expectancy quality adjusted for sample size.",
    "max_consecutive_wins": "Longest winning streak.",
    "max_consecutive_losses": "Longest losing streak - drives position-sizing survivability.",
    "avg_trade_duration": "Mean holding period in hours.",
    "mae_pct": "Average Max Adverse Excursion: worst drawdown DURING the hold, %%. Diagnoses stop placement / entry timing.",
    "mfe_pct": "Average Max Favorable Excursion: best runup during the hold, %%. Diagnoses exit timing.",
    "alpha": "Per-period OLS intercept of strategy vs benchmark returns (0 if no benchmark supplied).",
    "beta": "OLS slope vs benchmark - market sensitivity (0 if no benchmark).",
    "r_squared": "Fraction of strategy variance explained by the benchmark.",
    "treynor_ratio": "Annualised excess return per unit of beta (needs a benchmark).",
    "information_ratio": "Annualised active return / tracking error vs benchmark.",
    "skew": "Skewness of returns. Negative => occasional large losses.",
    "kurtosis": "Excess kurtosis. High => fat tails (more extreme moves than normal).",
    "downside_deviation": "Annualised std of negative returns only (%%).",
    "tail_ratio": "abs(95th pct) / abs(5th pct) of returns. >1 => right tail dominates.",
    "omega_ratio": "Probability-weighted gains / losses about a threshold.",
    "best_period": "Best single-period return (%%).",
    "worst_period": "Worst single-period return (%%).",
    "turnover": "Traded notional / capital - a proxy for fee/slippage drag.",
}

_FLAG_DEFS: Dict[str, str] = {
    "benchmark_available": "Whether a benchmark series was supplied; if False, alpha/beta/treynor/information_ratio are 0.",
    "low_sample": "True when total_trades < 30 - treat all ratios with caution.",
}

_PITFALLS: Dict[str, str] = {
    "sharpe_ratio": "Most likely to mislead: high in-sample after optimization is expected, not impressive. Compare to deflated_sharpe_ratio and OOS.",
    "profit_factor": "inf / very high usually means too few losing trades to be meaningful.",
    "max_drawdown": "Understated by the closed-trade equity curve; true intra-trade drawdown is worse.",
    "calmar_ratio": "Sensitive to a single drawdown event and to window length.",
    "deflated_sharpe_ratio": "Requires an honest n_trials. The research agent must accumulate trials across the whole session, or this lies.",
}


def metrics_glossary() -> Dict[str, Any]:
    """Definition + pitfalls of every reported metric, plus global caveats."""
    metrics = {}
    for key in METRIC_KEYS:
        entry: Dict[str, str] = {"definition": _DEFS.get(key, "(no description)")}
        if key in _PITFALLS:
            entry["pitfall"] = _PITFALLS[key]
        metrics[key] = entry
    flags = {key: {"definition": _FLAG_DEFS.get(key, "")} for key in FLAG_KEYS}
    return {"metrics": metrics, "flags": flags, "global_caveats": GLOBAL_CAVEATS}
