"""Backtest engine.

Orchestrates a vectorised-fetch / bar-by-bar-simulate / aggregate pipeline:

    data (marketdata) -> signals (strategy) -> fills (here) -> metrics (analytics)

The engine owns *only* the trade simulation and the wiring between layers. It
holds no indicator logic (strategy) and no metric formulas (analytics).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analytics import performance
from src.brokers.base import AccountSnapshot
from src.execution.sizing import PositionSizer, RiskBasedSizer
from src.marketdata.client import MarketDataClient
from src.strategies import signals
from src.strategies.base import Strategy
from src.utils.numeric import round_quantity

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything produced by a backtest run."""

    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: List[float]
    initial_capital: float
    final_capital: float
    start: datetime
    end: datetime
    strategy_config: Dict[str, Any]


class BacktestEngine:
    """Runs a strategy over historical data and reports performance."""

    def __init__(self, strategy: Strategy, data_client: MarketDataClient, sizer: Optional[PositionSizer] = None):
        self.strategy = strategy
        self.data_client = data_client
        # Same sizing abstraction as live execution; defaults to the strategy's
        # own risk-based sizing so behaviour is unchanged unless a sizer is given.
        self.sizer = sizer or RiskBasedSizer(strategy)

    def run(
        self, symbols: List[str], start: datetime, end: datetime, initial_capital: float
    ) -> BacktestResult:
        """Backtest ``symbols`` over ``[start, end]`` starting from ``initial_capital``."""
        self.strategy.initialize()

        data = self.data_client.get_bars(symbols, self.strategy.config["timeframe"], start, end)

        # ``available`` is realised cash used to gate new entries; it carries
        # across symbols so the run can't spend the same dollar twice.
        self._available = initial_capital
        market_data: Dict[str, Dict[str, float]] = {}
        all_trades: List[Dict[str, Any]] = []

        for symbol, bars in data.items():
            if bars.empty:
                continue
            market_data[symbol] = {"first_open": bars["open"].iloc[0], "last_close": bars["close"].iloc[-1]}
            try:
                processed = self.strategy.process_data(bars)
                signal_map = self.strategy.generate_signals(processed)
                all_trades.extend(self._simulate_symbol(symbol, processed, signal_map))
            except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't abort the run
                logger.error("Error backtesting %s: %s", symbol, exc, exc_info=True)

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

        final_capital = initial_capital + (trades_df["pnl"].sum() if not trades_df.empty else 0.0)
        equity_curve = performance.build_equity_curve(trades_df, initial_capital)
        metrics = performance.compute_backtest_metrics(
            trades_df, equity_curve, initial_capital, final_capital, market_data
        )

        return BacktestResult(
            metrics=metrics,
            trades=trades_df,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            final_capital=final_capital,
            start=start,
            end=end,
            strategy_config=dict(self.strategy.config),
        )

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def _simulate_symbol(
        self, symbol: str, data: pd.DataFrame, signal_map: Dict[Any, str]
    ) -> List[Dict[str, Any]]:
        """Replay one symbol's bars, opening/closing a single position at a time."""
        if data.empty:
            return []

        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        timestamps = data.index.to_numpy()

        signal_series = (
            pd.Series(signal_map).reindex(data.index, fill_value=signals.HOLD)
            if signal_map
            else pd.Series(signals.HOLD, index=data.index)
        )
        bar_signals = signal_series.to_numpy()

        trades: List[Dict[str, Any]] = []
        position: Optional[Dict[str, Any]] = None

        for i in range(len(timestamps)):
            signal = bar_signals[i]

            if position is not None:
                closed = self._maybe_close(position, signal, opens[i], highs[i], lows[i], timestamps[i])
                if closed is not None:
                    trades.append(closed)
                    self._available += closed["pnl"]
                    position = None
                    continue

            if position is None and signal in signals.ENTRY_SIGNALS:
                position = self._maybe_open(symbol, signal, opens[i], timestamps[i])

        # Force-close anything still open at the final bar.
        if position is not None:
            trades.append(self._close(position, data["close"].iloc[-1], timestamps[-1], "END_OF_PERIOD"))
            self._available += trades[-1]["pnl"]

        return trades

    def _maybe_open(self, symbol: str, signal: str, price: float, timestamp) -> Optional[Dict[str, Any]]:
        # Present realised cash as an account snapshot so the same PositionSizer
        # used in live execution can size backtest entries.
        account = AccountSnapshot(
            cash=self._available, equity=self._available,
            buying_power=self._available, portfolio_value=self._available,
        )
        size = round_quantity(self.sizer.size(symbol, price, account))
        if size <= 0 or size * price > self._available:
            return None

        stop_pct = self.strategy.config["stop_loss"]
        take_pct = self.strategy.config["take_profit"]
        if signal == signals.BUY:
            stop, take = price * (1 - stop_pct), price * (1 + take_pct)
        else:
            stop, take = price * (1 + stop_pct), price * (1 - take_pct)

        return {
            "symbol": symbol,
            "side": signal,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp,
            "stop_loss": stop,
            "take_profit": take,
        }

    def _maybe_close(
        self, position: Dict[str, Any], signal: str, price_open, high, low, timestamp
    ) -> Optional[Dict[str, Any]]:
        """Return a closed-trade record if an exit triggers this bar, else None."""
        if position["side"] == signals.BUY:
            if low <= position["stop_loss"]:
                return self._close(position, position["stop_loss"], timestamp, "STOP_LOSS")
            if high >= position["take_profit"]:
                return self._close(position, position["take_profit"], timestamp, "TAKE_PROFIT")
            if signal in (signals.SELL, signals.CLOSE_BUY):
                return self._close(position, price_open, timestamp, "SIGNAL")
        else:  # short
            if high >= position["stop_loss"]:
                return self._close(position, position["stop_loss"], timestamp, "STOP_LOSS")
            if low <= position["take_profit"]:
                return self._close(position, position["take_profit"], timestamp, "TAKE_PROFIT")
            if signal in (signals.BUY, signals.CLOSE_SELL):
                return self._close(position, price_open, timestamp, "SIGNAL")
        return None

    @staticmethod
    def _close(position: Dict[str, Any], exit_price: float, timestamp, reason: str) -> Dict[str, Any]:
        direction = 1 if position["side"] == signals.BUY else -1
        pnl = (exit_price - position["entry_price"]) * position["size"] * direction
        return {
            "symbol": position["symbol"],
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": position["size"],
            "pnl": pnl,
            "exit_reason": reason,
        }
