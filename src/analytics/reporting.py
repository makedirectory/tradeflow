"""Human-readable rendering of backtest results.

Kept separate from metric *computation* so the numbers can be consumed
programmatically (e.g. by the optimizer) without any formatting concerns.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# (metric key, label, format spec)
_ROWS = [
    ("total_return", "Total Return", "{:.2f}%"),
    ("buy_hold_return", "Buy & Hold Return", "{:.2f}%"),
    ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
    ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
    ("calmar_ratio", "Calmar Ratio", "{:.2f}"),
    ("max_drawdown", "Max Drawdown", "{:.2f}%"),
    ("total_trades", "Total Trades", "{:.0f}"),
    ("win_rate", "Win Rate", "{:.2f}%"),
    ("profit_factor", "Profit Factor", "{:.2f}"),
    ("avg_win", "Average Win", "${:.2f}"),
    ("avg_loss", "Average Loss", "${:.2f}"),
    ("largest_win", "Largest Win", "${:.2f}"),
    ("largest_loss", "Largest Loss", "${:.2f}"),
]


def format_backtest_report(
    metrics: Dict[str, float],
    initial_capital: float,
    final_capital: float,
    title: str = "Backtest Results",
) -> str:
    """Render metrics as an aligned, fixed-width text block."""
    lines = [f"=== {title} ===", f"{'Capital':24}${initial_capital:,.2f} -> ${final_capital:,.2f}"]
    for key, label, fmt in _ROWS:
        if key in metrics:
            value = fmt.format(metrics[key]) if metrics[key] != float("inf") else "inf"
            lines.append(f"{label:24}{value}")
    lines.append("=" * (len(title) + 8))
    return "\n".join(lines)


def log_backtest_report(metrics: Dict[str, float], initial_capital: float, final_capital: float) -> None:
    """Log the rendered report at INFO."""
    logger.info("\n%s", format_backtest_report(metrics, initial_capital, final_capital))
