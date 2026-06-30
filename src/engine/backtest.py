"""Backtest engine - where strategies go to find out the truth about themselves.

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
from src.costs.base import CostModel, Trade
from src.execution.sizing import PositionSizer, RiskBasedSizer
from src.marketdata.client import MarketDataClient
from src.marketdata.timeframe import Timeframe
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
    total_cost: float = 0.0  # total transaction cost charged across all trades
    gross_final_capital: float = 0.0  # final capital before cost (for the haircut attribution)


class BacktestEngine:
    """Runs a strategy over historical data and reports performance."""

    #: Trailing windows for the as-of liquidity inputs the cost model needs.
    ADV_WINDOW = 20
    VOL_WINDOW = 20

    def __init__(
        self,
        strategy: Strategy,
        data_client: MarketDataClient,
        sizer: Optional[PositionSizer] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.strategy = strategy
        self.data_client = data_client
        # Same sizing abstraction as live execution; defaults to the strategy's
        # own risk-based sizing so behaviour is unchanged unless a sizer is given.
        self.sizer = sizer or RiskBasedSizer(strategy)
        # When set, every simulated fill is charged so metrics are net of cost.
        # None = gross (the engine is a mechanism; the service layer defaults to net).
        self.cost_model = cost_model

    def run(
        self, symbols: List[str], start: datetime, end: datetime, initial_capital: float
    ) -> BacktestResult:
        """Backtest ``symbols`` over ``[start, end]`` starting from ``initial_capital``.

        Loads the full ``{symbol: bars}`` panel up front. For broad universes / long
        intraday histories that don't fit in RAM, :meth:`run_streaming` is equivalent
        but bounded by one symbol at a time.
        """
        self.strategy.initialize()
        timeframe = self.strategy.config["timeframe"]
        self._periods_per_year = Timeframe.parse(timeframe).periods_per_year()
        data = self.data_client.get_bars(symbols, timeframe, start, end)
        return self._simulate(((s, b) for s, b in data.items()), start, end, initial_capital)

    def run_streaming(
        self,
        source,
        symbols: List[str],
        start: datetime,
        end: datetime,
        initial_capital: float,
        lookback_days: int = 3650,
    ) -> BacktestResult:
        """Backtest from a :class:`~src.data.scan.BarSource`, **one symbol at a time**.

        The per-symbol simulation is independent and cash carries across symbols in
        order, so this is identical to :meth:`run` on the same data — but peak memory is
        bounded by a single symbol's history (each frame is retired before the next is
        scanned), not the whole panel. The leakage guard lives in the source's scan.
        """
        self.strategy.initialize()
        timeframe = self.strategy.config["timeframe"]
        self._periods_per_year = Timeframe.parse(timeframe).periods_per_year()

        def per_symbol():
            for symbol in symbols:
                scanned = source.scan([symbol], timeframe, end, lookback_days)
                frame = scanned.get(symbol)
                if frame is not None and not frame.empty:
                    yield symbol, frame  # retired after this iteration → bounded memory

        return self._simulate(per_symbol(), start, end, initial_capital)

    def _simulate(self, symbol_bars, start, end, initial_capital: float) -> BacktestResult:
        """Run the per-symbol simulation over an iterable of ``(symbol, bars)`` and assemble."""
        # ``available`` is realised cash used to gate new entries; it carries across
        # symbols (in order) so the run can't spend the same dollar twice.
        self._available = initial_capital
        market_data: Dict[str, Dict[str, float]] = {}
        all_trades: List[Dict[str, Any]] = []

        for symbol, bars in symbol_bars:
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

        net_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
        total_cost = float(trades_df["cost"].sum()) if "cost" in trades_df else 0.0
        final_capital = initial_capital + net_pnl
        equity_curve = performance.build_equity_curve(trades_df, initial_capital)
        metrics = performance.compute_backtest_metrics(
            trades_df, equity_curve, initial_capital, final_capital, market_data, start=start, end=end
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
            total_cost=total_cost,
            gross_final_capital=final_capital + total_cost,
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
        # As-of liquidity inputs for the cost model (trailing, so no look-ahead).
        if self.cost_model is not None and "volume" in data:
            adv = data["volume"].rolling(self.ADV_WINDOW).mean().fillna(0.0).to_numpy()
            vol = data["close"].pct_change().rolling(self.VOL_WINDOW).std().fillna(0.0).to_numpy()
        else:
            adv = vol = None

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
                # Track the worst/best price seen while open, for MAE/MFE.
                position["lowest"] = min(position["lowest"], lows[i])
                position["highest"] = max(position["highest"], highs[i])
                exit_cost = self._trade_cost(symbol, position["size"], opens[i], adv, vol, i) + self._carry(
                    position, i
                )
                closed = self._maybe_close(
                    position, signal, opens[i], highs[i], lows[i], timestamps[i], exit_cost
                )
                if closed is not None:
                    trades.append(closed)
                    self._available += closed["pnl"]
                    position = None
                    continue

            if position is None and signal in signals.ENTRY_SIGNALS:
                position = self._maybe_open(symbol, signal, opens[i], timestamps[i])
                if position is not None:
                    position["entry_bar"] = i
                    position["entry_cost"] = self._trade_cost(symbol, position["size"], opens[i], adv, vol, i)

        # Force-close anything still open at the final bar.
        if position is not None:
            last = len(timestamps) - 1
            final_cost = self._trade_cost(
                symbol, position["size"], data["close"].iloc[-1], adv, vol, -1
            ) + self._carry(position, last)
            trades.append(
                self._close(position, data["close"].iloc[-1], timestamps[-1], "END_OF_PERIOD", final_cost)
            )
            self._available += trades[-1]["pnl"]

        return trades

    def _trade_cost(self, symbol, shares, price, adv, vol, i) -> float:
        """Dollar cost of trading ``shares`` at bar ``i`` (0 when no cost model)."""
        if self.cost_model is None or adv is None or shares == 0:
            return 0.0
        trade = Trade(symbol=symbol, shares=shares, price=price, adv=float(adv[i]), daily_vol=float(vol[i]))
        return self.cost_model.cost(trade).total

    def _carry(self, position: Dict[str, Any], exit_bar: int) -> float:
        """Financing (borrow) cost of holding ``position`` until ``exit_bar`` - shorts only."""
        if self.cost_model is None:
            return 0.0
        held_years = max(exit_bar - position.get("entry_bar", exit_bar), 0) / self._periods_per_year
        notional = position["size"] * position["entry_price"]
        return self.cost_model.carry_cost(notional, position["side"] == signals.SELL, held_years)

    def _maybe_open(self, symbol: str, signal: str, price: float, timestamp) -> Optional[Dict[str, Any]]:
        # Present realised cash as an account snapshot so the same PositionSizer
        # used in live execution can size backtest entries.
        account = AccountSnapshot(
            cash=self._available,
            equity=self._available,
            buying_power=self._available,
            portfolio_value=self._available,
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
            # Running extremes while the position is open, seeded at entry.
            "lowest": price,
            "highest": price,
        }

    def _maybe_close(
        self, position: Dict[str, Any], signal: str, price_open, high, low, timestamp, exit_cost: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """Return a closed-trade record if an exit triggers this bar, else None."""
        if position["side"] == signals.BUY:
            if low <= position["stop_loss"]:
                return self._close(position, position["stop_loss"], timestamp, "STOP_LOSS", exit_cost)
            if high >= position["take_profit"]:
                return self._close(position, position["take_profit"], timestamp, "TAKE_PROFIT", exit_cost)
            if signal in (signals.SELL, signals.CLOSE_BUY):
                return self._close(position, price_open, timestamp, "SIGNAL", exit_cost)
        else:  # short
            if high >= position["stop_loss"]:
                return self._close(position, position["stop_loss"], timestamp, "STOP_LOSS", exit_cost)
            if low <= position["take_profit"]:
                return self._close(position, position["take_profit"], timestamp, "TAKE_PROFIT", exit_cost)
            if signal in (signals.BUY, signals.CLOSE_SELL):
                return self._close(position, price_open, timestamp, "SIGNAL", exit_cost)
        return None

    @staticmethod
    def _close(
        position: Dict[str, Any], exit_price: float, timestamp, reason: str, exit_cost: float = 0.0
    ) -> Dict[str, Any]:
        direction = 1 if position["side"] == signals.BUY else -1
        entry = position["entry_price"]
        gross_pnl = (exit_price - entry) * position["size"] * direction
        # Cost is charged on BOTH legs (entry + exit); net pnl is what the book keeps.
        cost = position.get("entry_cost", 0.0) + exit_cost
        pnl = gross_pnl - cost
        # Max adverse / favorable excursion as fractions of entry price. For a long
        # the adverse extreme is the low and the favorable extreme is the high; for
        # a short the roles swap.
        low, high = position.get("lowest", entry), position.get("highest", entry)
        if direction == 1:
            mae_pct, mfe_pct = (entry - low) / entry, (high - entry) / entry
        else:
            mae_pct, mfe_pct = (high - entry) / entry, (entry - low) / entry
        return {
            "symbol": position["symbol"],
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "entry_price": entry,
            "exit_price": exit_price,
            "size": position["size"],
            "pnl": pnl,
            "gross_pnl": gross_pnl,
            "cost": cost,
            "exit_reason": reason,
            "mae_pct": float(max(mae_pct, 0.0)) * 100.0,
            "mfe_pct": float(max(mfe_pct, 0.0)) * 100.0,
        }
