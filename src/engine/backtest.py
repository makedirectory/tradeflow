"""Backtest engine - where strategies go to find out the truth about themselves.

Orchestrates a vectorized-fetch / bar-by-bar-simulate / aggregate pipeline:

    data (marketdata) -> signals (strategy) -> fills (here) -> metrics (analytics)

The engine owns *only* the trade simulation and the wiring between layers. It
holds no indicator logic (strategy) and no metric formulas (analytics).

**One clock, one capital pool.** Every symbol is simulated on a single merged
timeline against shared cash, so positions compete for capital exactly as they
would live: within a bar, exits settle before entries, and when more entry
signals arrive than the book can fund they are admitted in conviction order
(the strategy's own score, ties broken by symbol for determinism).

This is what makes absolute metrics mean anything. Simulating each symbol
independently over its whole history - the engine's original shape - summed N
full-capital single-name backtests onto one capital base, so return and Sharpe
both scaled with universe size. See spec 025.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics import metrics as m
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

#: How the engine accounts for capital. Bump this whenever a change makes results
#: incommensurable with earlier ones, so a stored record can never be compared to a
#: newer one as though the two measured the same thing.
#:
#: * **1** — pre-spec-025. Each symbol was simulated independently over its whole
#:   history against the full capital base, and the equity curve accumulated realized
#:   P&L at exit, resampled to calendar days. Absolute return and Sharpe scaled with
#:   universe size, and position limits were per-symbol.
#: * **2** — spec 025. One merged timeline, one shared capital pool, portfolio-level
#:   position limits, and a per-bar mark-to-market equity curve.
#:
#: Records written before this field existed carry no version; absence means 1.
ACCOUNTING_VERSION = 2


class BacktestError(RuntimeError):
    """A backtest could not be simulated at all (as opposed to finding no edge).

    Raised when every symbol in the universe failed, which almost always means a
    broken strategy or an unconstructable config. Distinguishing this from a
    genuine zero-trade result is what stops a crash being scored as "no edge".
    """


def _align_tz(when, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Coerce ``when`` to ``index``'s tz-awareness so the two can be compared.

    Callers pass plain datetimes while bar indices are usually exchange-local, and
    pandas refuses to compare tz-naive with tz-aware.
    """
    ts = pd.Timestamp(when)
    tz = getattr(index, "tz", None)
    if tz is not None and ts.tz is None:
        return ts.tz_localize(tz)
    if tz is None and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


#: Fallback when a strategy declares no ``position_limits`` (mirrors the base class).
DEFAULT_POSITION_LIMITS: Dict[str, float] = {
    "max_positions": 5,
    "max_position_size": 1500.0,
    "max_total_risk": 0.05,
}


@dataclass
class _Panel:
    """One symbol's arrays, aligned to the merged timeline.

    ``rows`` maps each master-timeline step to this symbol's row index, or -1 when
    the symbol has no bar then (a listing gap, a halt, a shorter history). -1 means
    "not tradeable at t" - never "carry the last price forward".
    """

    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    timestamps: np.ndarray
    sig: np.ndarray
    score: np.ndarray
    rows: np.ndarray
    adv: Optional[np.ndarray] = None
    vol: Optional[np.ndarray] = None
    last_timestamp: Any = None


@dataclass
class _Book:
    """Mutable portfolio state carried across the merged timeline."""

    cash: float
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def open_risk(self) -> float:
        """Sum of per-position risk (notional x stop distance) across the book."""
        return sum(p["risk"] for p in self.positions.values())

    def market_value(self) -> float:
        """Reserved notional plus unrealized gross P&L, marked at last seen price."""
        total = 0.0
        for p in self.positions.values():
            direction = 1 if p["side"] == signals.BUY else -1
            total += p["notional"] + (p["last_price"] - p["entry_price"]) * p["size"] * direction
        return total

    def equity(self) -> float:
        return self.cash + self.market_value()


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
        # own risk-based sizing so behavior is unchanged unless a sizer is given.
        self.sizer = sizer or RiskBasedSizer(strategy)
        # When set, every simulated fill is charged so metrics are net of cost.
        # None = gross (the engine is a mechanism; the service layer defaults to net).
        self.cost_model = cost_model

    def run(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        initial_capital: float,
        trade_from: Optional[datetime] = None,
    ) -> BacktestResult:
        """Backtest ``symbols`` over ``[start, end]`` starting from ``initial_capital``.

        Loads the full ``{symbol: bars}`` panel up front. For broad universes / long
        intraday histories that don't fit in RAM, :meth:`run_streaming` is equivalent
        but bounded by one symbol at a time.

        ``trade_from`` separates *warmup* from *trading*: bars before it feed the
        indicators but open no positions, and the equity curve starts there. This is
        what lets an out-of-sample window be measured on its own portfolio curve
        rather than reconstructed from a filtered list of trades.
        """
        self.strategy.initialize()
        timeframe = self.strategy.config["timeframe"]
        self._periods_per_year = Timeframe.parse(timeframe).periods_per_year()
        data = self.data_client.get_bars(symbols, timeframe, start, end)
        return self._simulate(
            ((s, b) for s, b in data.items()), start, end, initial_capital, trade_from
        )

    def run_streaming(
        self,
        source,
        symbols: List[str],
        start: datetime,
        end: datetime,
        initial_capital: float,
        lookback_days: int = 3650,
        trade_from: Optional[datetime] = None,
    ) -> BacktestResult:
        """Backtest from a :class:`~src.data.scan.BarSource`, **fetching** one symbol at a time.

        Identical to :meth:`run` on the same data; the leakage guard lives in the
        source's scan.

        Memory note: this used to be bounded by a single symbol's history, because each
        symbol was simulated independently and its frame retired immediately. Portfolio
        accounting removed that property — positions compete for one capital pool, which
        cannot be resolved without every symbol's bars resident. The saving here is now
        in the *fetch* path (scan one symbol at a time rather than the whole panel at
        once), not in peak simulation memory. For universes that genuinely do not fit,
        the out-of-core substrate is the right tool.
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

        return self._simulate(per_symbol(), start, end, initial_capital, trade_from)

    def _simulate(
        self, symbol_bars, start, end, initial_capital: float, trade_from=None
    ) -> BacktestResult:
        """Simulate the whole universe on one clock against one capital pool."""
        panels, market_data, master = self._prepare(symbol_bars)
        all_trades, equity_curve = self._replay(panels, master, initial_capital, trade_from)

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

        net_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
        total_cost = float(trades_df["cost"].sum()) if "cost" in trades_df else 0.0
        final_capital = initial_capital + net_pnl
        metrics = performance.compute_backtest_metrics(
            trades_df,
            equity_curve,
            initial_capital,
            final_capital,
            market_data,
            start=start,
            end=end,
            # The curve is now sampled per *bar*, so it must be annualized on the
            # strategy's own timeframe rather than the daily default.
            periods_per_year=getattr(self, "_periods_per_year", None) or m.TRADING_DAYS_PER_YEAR,
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
    def _prepare(self, symbol_bars):
        """Compute signals/scores per symbol and align every symbol to one timeline.

        Portfolio accounting needs the whole panel resident: a shared capital pool
        cannot be simulated one symbol at a time.
        """
        frames: Dict[str, pd.DataFrame] = {}
        market_data: Dict[str, Dict[str, float]] = {}
        attempted = 0
        failures: List[str] = []

        for symbol, bars in symbol_bars:
            if bars.empty:
                continue
            attempted += 1
            market_data[symbol] = {"first_open": bars["open"].iloc[0], "last_close": bars["close"].iloc[-1]}
            try:
                processed = self.strategy.process_data(bars)
                if processed is None or processed.empty:
                    continue
                frames[symbol] = processed
            except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't abort the run
                failures.append(f"{symbol}: {exc}")
                logger.error("Error preparing %s: %s", symbol, exc, exc_info=True)

        # A run where *every* symbol raised is a broken strategy, not a flat one.
        # Reporting it as a zero-trade result would let a config error masquerade as
        # "no edge" - the one failure mode a validation engine must never have.
        if attempted and len(failures) == attempted:
            raise BacktestError(
                f"backtest failed for all {attempted} symbol(s); first error - {failures[0]}"
            )
        if not frames:
            return {}, market_data, pd.DatetimeIndex([])

        master = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
        panels: Dict[str, _Panel] = {}
        for symbol in sorted(frames):  # sorted: iteration order must not affect results
            data = frames[symbol]
            try:
                panels[symbol] = self._panel_for(symbol, data, master)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}: {exc}")
                logger.error("Error backtesting %s: %s", symbol, exc, exc_info=True)

        if attempted and len(failures) == attempted:
            raise BacktestError(
                f"backtest failed for all {attempted} symbol(s); first error - {failures[0]}"
            )
        return panels, market_data, master

    def _panel_for(self, symbol: str, data: pd.DataFrame, master: pd.DatetimeIndex) -> _Panel:
        """Build one symbol's master-aligned arrays."""
        signal_map = self.strategy.generate_signals(data)
        sig = (
            pd.Series(signal_map).reindex(data.index, fill_value=signals.HOLD).to_numpy()
            if signal_map
            else np.full(len(data), signals.HOLD, dtype=object)
        )
        try:
            scores = self.strategy.calculate_scores(data)
            score = pd.Series(scores).reindex(data.index).fillna(0.0).to_numpy(dtype=float)
        except Exception:  # noqa: BLE001 - ranking falls back to symbol order
            logger.debug("No usable score for %s; ranking on symbol order", symbol, exc_info=True)
            score = np.zeros(len(data), dtype=float)

        if self.cost_model is not None and "volume" in data:
            adv = data["volume"].rolling(self.ADV_WINDOW).mean().fillna(0.0).to_numpy()
            vol = data["close"].pct_change().rolling(self.VOL_WINDOW).std().fillna(0.0).to_numpy()
        else:
            adv = vol = None

        return _Panel(
            opens=data["open"].to_numpy(),
            timestamps=data.index.to_numpy(),
            highs=data["high"].to_numpy(),
            lows=data["low"].to_numpy(),
            closes=data["close"].to_numpy(),
            sig=sig,
            score=score,
            rows=data.index.get_indexer(master),
            adv=adv,
            vol=vol,
            last_timestamp=data.index[-1],
        )

    def _replay(self, panels: Dict[str, _Panel], master, initial_capital: float, trade_from=None):
        """Walk the merged timeline once: mark, exit, rank, admit, record."""
        book = _Book(cash=initial_capital)
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [initial_capital]
        if not panels:
            return trades, equity_curve

        cutoff = _align_tz(trade_from, master) if trade_from is not None else None
        limits = {**DEFAULT_POSITION_LIMITS, **(self.strategy.config.get("position_limits") or {})}
        max_positions = limits["max_positions"]
        max_total_risk = limits["max_total_risk"]
        n_steps = len(master)
        order = sorted(panels)

        for k in range(n_steps):
            # 1. Mark open positions to market and track excursion extremes.
            for symbol in order:
                pos = book.positions.get(symbol)
                if pos is None:
                    continue
                i = panels[symbol].rows[k]
                if i < 0:  # no bar for this name right now
                    continue
                panel = panels[symbol]
                pos["last_price"] = panel.closes[i]
                pos["lowest"] = min(pos["lowest"], panel.lows[i])
                pos["highest"] = max(pos["highest"], panel.highs[i])

            # 2. Exits first, so the capital they free is reusable this same bar.
            for symbol in order:
                pos = book.positions.get(symbol)
                if pos is None:
                    continue
                panel = panels[symbol]
                i = panel.rows[k]
                if i < 0:
                    continue
                exit_cost = self._trade_cost(
                    symbol, pos["size"], panel.opens[i], panel.adv, panel.vol, i
                ) + self._carry(pos, k)
                closed = self._maybe_close(
                    pos, panel.sig[i], panel.opens[i], panel.highs[i], panel.lows[i], panel.timestamps[i], exit_cost
                )
                if closed is not None:
                    trades.append(closed)
                    book.cash += pos["notional"] + closed["gross_pnl"] - exit_cost
                    del book.positions[symbol]

            # 3. Collect this bar's entry candidates across the whole universe.
            #    Before the cutoff, bars only warm up indicators - no position opens.
            candidates: List[Tuple[float, str, int]] = []
            trading = cutoff is None or master[k] >= cutoff
            for symbol in order if trading else ():
                if symbol in book.positions:
                    continue
                panel = panels[symbol]
                i = panel.rows[k]
                if i < 0:
                    continue
                if panel.sig[i] in signals.ENTRY_SIGNALS:
                    candidates.append((abs(float(panel.score[i])), symbol, i))

            # 4. Admit in conviction order while limits and cash allow. Ranking by the
            #    strategy's own score keeps one source of truth; the symbol tie-break
            #    makes a tie deterministic rather than dict-order dependent.
            if candidates:
                candidates.sort(key=lambda c: (-c[0], c[1]))
                equity_now = book.equity()
                for _, symbol, i in candidates:
                    if len(book.positions) >= max_positions:
                        break
                    panel = panels[symbol]
                    pos = self._open_position(
                        symbol, panel.sig[i], panel.opens[i], panel.timestamps[i], book, equity_now, max_total_risk
                    )
                    if pos is None:
                        continue
                    pos["entry_k"] = k
                    pos["entry_cost"] = self._trade_cost(
                        symbol, pos["size"], panel.opens[i], panel.adv, panel.vol, i
                    )
                    book.cash -= pos["notional"] + pos["entry_cost"]
                    book.positions[symbol] = pos

            if trading:
                equity_curve.append(book.equity())

        # Force-close whatever is still open, each at its own last bar.
        for symbol in order:
            pos = book.positions.get(symbol)
            if pos is None:
                continue
            panel = panels[symbol]
            price = panel.closes[-1]
            final_cost = self._trade_cost(
                symbol, pos["size"], price, panel.adv, panel.vol, -1
            ) + self._carry(pos, n_steps - 1)
            closed = self._close(pos, price, panel.last_timestamp, "END_OF_PERIOD", final_cost)
            trades.append(closed)
            book.cash += pos["notional"] + closed["gross_pnl"] - final_cost
            del book.positions[symbol]
        if len(equity_curve) > 1:
            equity_curve[-1] = book.equity()

        return trades, equity_curve

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
        held_years = max(exit_bar - position.get("entry_k", exit_bar), 0) / self._periods_per_year
        notional = position["size"] * position["entry_price"]
        return self.cost_model.carry_cost(notional, position["side"] == signals.SELL, held_years)

    def _open_position(
        self,
        symbol: str,
        signal: str,
        price: float,
        timestamp,
        book: _Book,
        equity: float,
        max_total_risk: float,
    ) -> Optional[Dict[str, Any]]:
        """Size and admit one candidate, or return None if the book cannot fund it."""
        # Free cash is the buying power; equity is the whole book. Sizing against
        # cash is what makes positions actually compete for the same dollars.
        account = AccountSnapshot(
            cash=book.cash,
            equity=equity,
            buying_power=book.cash,
            portfolio_value=equity,
        )
        size = round_quantity(self.sizer.size(symbol, price, account))
        if size <= 0 or size * price > book.cash:
            return None

        stop_pct = self.strategy.config["stop_loss"]
        take_pct = self.strategy.config["take_profit"]
        # Portfolio-level risk budget. calculate_position_size caps a *single*
        # position against max_total_risk; nothing capped the book as a whole, so
        # "max_total_risk" was per-symbol in practice.
        risk = size * price * stop_pct
        if max_total_risk and book.open_risk() + risk > equity * max_total_risk:
            return None

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
            "notional": size * price,
            "risk": risk,
            "last_price": price,
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
