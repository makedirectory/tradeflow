"""Human-readable rendering of backtest results.

Kept separate from metric *computation* so the numbers can be consumed
programmatically (e.g. by the optimizer) without any formatting concerns.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# (metric key, label, format spec), grouped into report sections.
_SECTIONS = [
    (
        "Returns",
        [
            ("total_return", "Total Return", "{:.2f}%"),
            ("cagr", "CAGR", "{:.2f}%"),
            ("buy_hold_return", "Buy & Hold Return", "{:.2f}%"),
            ("annualized_volatility", "Annualized Volatility", "{:.2f}%"),
        ],
    ),
    (
        "Risk-adjusted",
        [
            ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
            ("probabilistic_sharpe_ratio", "  Probabilistic Sharpe", "{:.2f}"),
            ("deflated_sharpe_ratio", "  Deflated Sharpe", "{:.2f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
            ("calmar_ratio", "Calmar Ratio", "{:.2f}"),
            ("martin_ratio", "Martin Ratio (UPI)", "{:.2f}"),
            ("information_ratio", "Information Ratio", "{:.2f}"),
            ("treynor_ratio", "Treynor Ratio", "{:.2f}"),
        ],
    ),
    (
        "Drawdown & tail",
        [
            ("max_drawdown", "Max Drawdown", "{:.2f}%"),
            ("max_drawdown_duration", "Max DD Duration (days)", "{:.0f}"),
            ("ulcer_index", "Ulcer Index", "{:.2f}"),
            ("recovery_factor", "Recovery Factor", "{:.2f}"),
            ("var_95", "VaR (95%)", "{:.2f}%"),
            ("cvar_95", "CVaR (95%)", "{:.2f}%"),
            ("var_99", "VaR (99%)", "{:.2f}%"),
            ("exposure", "Time in Market", "{:.1f}%"),
        ],
    ),
    (
        "Trades",
        [
            ("total_trades", "Total Trades", "{:.0f}"),
            ("win_rate", "Win Rate", "{:.2f}%"),
            ("profit_factor", "Profit Factor", "{:.2f}"),
            ("payoff_ratio", "Payoff Ratio", "{:.2f}"),
            ("expectancy", "Expectancy", "${:.2f}"),
            ("sqn", "System Quality (SQN)", "{:.2f}"),
            ("kelly_criterion", "Kelly Fraction", "{:.2f}"),
            ("max_consecutive_wins", "Max Consecutive Wins", "{:.0f}"),
            ("max_consecutive_losses", "Max Consecutive Losses", "{:.0f}"),
            ("avg_trade_duration", "Avg Trade Duration (h)", "{:.2f}"),
            ("mae_pct", "Avg MAE", "{:.2f}%"),
            ("mfe_pct", "Avg MFE", "{:.2f}%"),
            ("avg_win", "Average Win", "${:.2f}"),
            ("avg_loss", "Average Loss", "${:.2f}"),
            ("largest_win", "Largest Win", "${:.2f}"),
            ("largest_loss", "Largest Loss", "${:.2f}"),
        ],
    ),
    (
        "Benchmark-relative",
        [
            ("alpha", "Alpha (per period)", "{:.5f}"),
            ("beta", "Beta", "{:.2f}"),
            ("r_squared", "R-squared", "{:.2f}"),
        ],
    ),
]

# Flat view for backward compatibility / simple consumers.
_ROWS = [row for _, rows in _SECTIONS for row in rows]


def format_backtest_report(
    metrics: Dict[str, float],
    initial_capital: float,
    final_capital: float,
    title: str = "Backtest Results",
) -> str:
    """Render metrics as an aligned, fixed-width text block, grouped by section."""
    lines = [f"=== {title} ===", f"{'Capital':28}${initial_capital:,.2f} -> ${final_capital:,.2f}"]
    if metrics.get("low_sample"):
        lines.append(f"{'(!) low sample':28}fewer than {30} trades - treat ratios with caution")
    if not metrics.get("benchmark_available", True):
        lines.append(f"{'(i) no benchmark':28}alpha/beta/information-ratio unavailable")
    for section, rows in _SECTIONS:
        section_lines = []
        for key, label, fmt in rows:
            if key in metrics:
                value = fmt.format(metrics[key]) if metrics[key] != float("inf") else "inf"
                section_lines.append(f"{label:28}{value}")
        if section_lines:
            lines.append(f"--- {section} ---")
            lines.extend(section_lines)
    lines.append("=" * (len(title) + 8))
    return "\n".join(lines)


def log_backtest_report(metrics: Dict[str, float], initial_capital: float, final_capital: float) -> None:
    """Log the rendered report at INFO."""
    logger.info("\n%s", format_backtest_report(metrics, initial_capital, final_capital))


def _age_str(ts: Optional[str]) -> str:
    """A human ``"N days"``/``"N hours"`` age for an ISO timestamp; ``"unknown"``
    when it can't be parsed - never let a formatting failure block the notice."""
    if not ts:
        return "unknown age"
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - when
    except ValueError:
        return "unknown age"
    seconds = delta.total_seconds()
    if seconds < 0:
        return "0 days"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def format_cached_notice(row: Dict[str, Any], *, current_accounting: Optional[int] = None) -> str:
    """A prominent, unmistakable "this is a memo, not a fresh verification" banner.

    Every command that can serve a memoized trial (``backtest``/``optimize``/
    ``walkforward``, CLI and MCP alike) renders this immediately above its report
    so the reused case looks visually distinct everywhere, not just wherever it
    was first implemented. Always names the original run's id and timestamp/age -
    never let a stale number pass as freshly checked.
    """
    ts = row.get("ts")
    lines = [
        f"=== REUSED — trial {row.get('id', '?')} from {ts or 'unknown time'} ({_age_str(ts)} old) ===",
        "Not re-run: an identical trial already exists in the trial store. Pass --force/--rerun to re-verify.",
    ]
    row_accounting = row.get("accounting")
    if current_accounting is not None and row_accounting is not None and row_accounting != current_accounting:
        lines.append(
            f"(!) stored under accounting v{row_accounting}, engine is v{current_accounting} — "
            "its metrics are NOT comparable to a fresh run"
        )
    lines.append(
        "(i) reused on 'same requested inputs' only — not yet guaranteed identical underlying data "
        "(no data-vintage stamp yet); a rare vendor correction/backfill since the original run could differ"
    )
    return "\n".join(lines)
