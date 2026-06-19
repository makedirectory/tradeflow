"""Backtest performance accounting.

Turns a table of completed trades plus an equity curve into a metrics dict. This
logic lives here - not inside ``Strategy`` or the engine - so strategies stay
about signals and the engine stays about orchestration (separation of concerns).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.analytics import metrics as m

#: The metric keys always present in a results dict (handy for optimizers/reports).
METRIC_KEYS = (
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
)


def empty_metrics() -> Dict[str, float]:
    """A zeroed metrics dict, returned when no trades occurred."""
    return {key: 0.0 for key in METRIC_KEYS}


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    equity_curve: List[float],
    initial_capital: float,
    final_capital: float,
    market_data: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute headline performance metrics for a completed backtest.

    Args:
        trades_df: One row per closed trade; must include a ``pnl`` column.
        equity_curve: Account equity sampled over the backtest.
        initial_capital / final_capital: Capital at the start / end.
        market_data: ``{symbol: {"first_open", "last_close"}}`` for buy & hold.

    Returns:
        A flat ``{metric_name: value}`` dict (see :data:`METRIC_KEYS`).
    """
    if trades_df.empty:
        return empty_metrics()

    pnl = trades_df["pnl"]
    returns = m.returns_from_equity(equity_curve)

    winning = trades_df[pnl > 0]["pnl"]
    losing = trades_df[pnl < 0]["pnl"]

    total_return_pct = (final_capital / initial_capital - 1) * 100 if initial_capital else 0.0
    drawdown_pct = m.max_drawdown(equity_curve) * 100

    return {
        "total_return": total_return_pct,
        "buy_hold_return": _buy_hold_return(market_data),
        "sharpe_ratio": m.sharpe_ratio(returns),
        "sortino_ratio": m.sortino_ratio(returns),
        "calmar_ratio": m.calmar_ratio(total_return_pct, drawdown_pct),
        "max_drawdown": drawdown_pct,
        "total_trades": int(len(trades_df)),
        "win_rate": m.win_rate(pnl) * 100,
        "profit_factor": m.profit_factor(pnl),
        "avg_win": float(winning.mean()) if not winning.empty else 0.0,
        "avg_loss": float(abs(losing.mean())) if not losing.empty else 0.0,
        "largest_win": float(winning.max()) if not winning.empty else 0.0,
        "largest_loss": float(abs(losing.min())) if not losing.empty else 0.0,
    }


def build_equity_curve(trades_df: pd.DataFrame, initial_capital: float) -> List[float]:
    """Build a daily equity curve by accumulating trade P&L at exit time."""
    if trades_df.empty:
        return [initial_capital]

    daily_pnl = trades_df.set_index("exit_time")["pnl"].resample("D").sum().fillna(0)
    equity = initial_capital + daily_pnl.cumsum()
    return [initial_capital, *equity.tolist()]


def _buy_hold_return(market_data: Dict[str, Dict[str, float]]) -> float:
    """Average buy & hold return (%) across the traded symbols."""
    per_symbol = [
        (data["last_close"] / data["first_open"] - 1) * 100
        for data in market_data.values()
        if data.get("first_open")
    ]
    return float(np.mean(per_symbol)) if per_symbol else 0.0
