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
both scaled with universe size.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tradeflow.analytics import metrics as m
from tradeflow.analytics import performance
from tradeflow.brokers.base import AccountSnapshot
from tradeflow.costs.base import CostModel, Trade
from tradeflow.execution.sizing import PositionSizer, RiskBasedSizer
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.numeric import round_quantity

logger = logging.getLogger(__name__)

#: How the engine accounts for capital. Bump this whenever a change makes results
#: incommensurable with earlier ones, so a stored record can never be compared to a
#: newer one as though the two measured the same thing.
#:
#: * **1** — the original model. Each symbol was simulated independently over its whole
#:   history against the full capital base, and the equity curve accumulated realized
#:   P&L at exit, resampled to calendar days. Absolute return and Sharpe scaled with
#:   universe size, and position limits were per-symbol.
#: * **2** — one merged timeline, one shared capital pool, portfolio-level position limits,
#:   and a per-bar mark-to-market equity curve. Per-step quantities (the equity curve, short
#:   carry) were annualized at the strategy's single-symbol timeframe rate, which understates
#:   the merged timeline's sampling rate whenever symbols don't share one bar grid.
#: * **3** — the current model. Per-step quantities annualize on the merged timeline's own
#:   rate. Identical to 2 for a universe whose symbols share a grid; corrects inflated
#:   Sharpe/volatility and understated carry when they don't.
#:
#: Records written before this field existed carry no version; absence means 1.
ACCOUNTING_VERSION = 3


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
    #: This symbol's own last valid master-timeline step (where ``rows >= 0``) -
    #: distinct from the merged panel's last step when this symbol's history is
    #: shorter than the universe's. Force-close must age a position against this,
    #: not the master timeline's length, or a short-history symbol gets charged
    #: carry cost as if held through bars it never actually traded.
    last_step: int = -1


@dataclass
class _Execution:
    """What the sizer asked for versus what could actually be traded.

    Whole-share rounding is silent, and at small capital it is not a rounding error:
    the sizer asks for 0.4 shares of a $500 name, the engine floors it to zero, and the
    equity curve is correct while the reason for it is invisible. These counters make
    the difference between the intended book and the executed one a number rather than
    an inference.

    Filled and requested notional are accumulated over *opened* positions only, so the
    ratio is pure rounding drag; positions lost outright are counted separately, since
    a book that cannot open a name at all is a different problem from one that opens a
    slightly smaller one.
    """

    requested_notional: float = 0.0
    filled_notional: float = 0.0
    filled: int = 0
    rounded_to_zero: int = 0
    below_min_notional: int = 0

    def drag_pct(self) -> float:
        """Requested notional that whole-share rounding removed, as a percentage."""
        if self.requested_notional <= 0:
            return 0.0
        return float((1.0 - self.filled_notional / self.requested_notional) * 100.0)

    def unfillable_pct(self) -> float:
        """Share of intended entries that could not be opened at all, as a percentage.

        Separate from the drag: a book that opens every name slightly smaller is a
        different proposition from one that silently never opens a quarter of them.
        """
        attempted = self.filled + self.rounded_to_zero + self.below_min_notional
        if attempted <= 0:
            return 0.0
        return float((self.rounded_to_zero + self.below_min_notional) / attempted * 100.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_notional": self.requested_notional,
            "filled_notional": self.filled_notional,
            "rounding_drag_pct": self.drag_pct(),
            "positions_filled": self.filled,
            "positions_rounded_to_zero": self.rounded_to_zero,
            "positions_below_min_notional": self.below_min_notional,
            "unfillable_pct": self.unfillable_pct(),
        }


@dataclass
class _Book:
    """Mutable portfolio state carried across the merged timeline."""

    cash: float
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution: _Execution = field(default_factory=_Execution)

    def open_risk(self) -> float:
        """Sum of per-position risk (notional x stop distance) across the book."""
        return sum(p["risk"] for p in self.positions.values())

    def gross_exposure(self) -> float:
        """Marked gross notional across the book, shorts counted by magnitude.

        Deliberately not :meth:`market_value`, which is an accounting number -
        reserved notional plus unrealized P&L under full cash collateral. This is
        the market exposure the book is actually carrying.
        """
        return sum(p["size"] * p["last_price"] for p in self.positions.values())

    def market_value(self) -> float:
        """Reserved notional plus unrealized gross P&L, marked at last seen price."""
        total = 0.0
        for p in self.positions.values():
            direction = 1 if p["side"] == signals.BUY else -1
            total += p["notional"] + (p["last_price"] - p["entry_price"]) * p["size"] * direction
        return total

    def equity(self) -> float:
        return self.cash + self.market_value()


def _leg_report(leg_curves, trades, initial_capital: float) -> Dict[str, Any]:
    """What each side of a long/short book actually did, separately.

    A near-zero net beta is the headline of any market-neutral result, and it has two
    completely different causes: genuinely small exposure on both sides, or a large long
    beta cancelling a large short one. Those are the same number and opposite risks, and
    nothing in a net-level report tells them apart.

    Curves are realized-plus-unrealized per side, sampled on the equity curve's own
    steps, so the per-leg volatility, drawdown and beta describe the position as it was
    held rather than only as it was closed.

    Diagnostic only: no threshold, no verdict. Make the risk visible first and decide
    afterwards whether any of it deserves to gate something.
    """
    report: Dict[str, Any] = {}
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    for side, curve in leg_curves.items():
        name = "long" if side == signals.BUY else "short"
        if trades_df.empty or "side" not in trades_df:
            leg_trades = pd.DataFrame()
        else:
            leg_trades = trades_df[trades_df["side"] == side]
        if len(curve) < 2 and leg_trades.empty:
            continue
        # As a fraction of capital, so the two legs are on one scale and the pair adds
        # up to something a reader can check against the headline return.
        series = pd.Series(curve, dtype="float64") / initial_capital if initial_capital else pd.Series(curve)
        report[name] = {
            "pnl": float(leg_trades["pnl"].sum()) if not leg_trades.empty else 0.0,
            "return_pct": float(series.iloc[-1] * 100.0),
            "volatility_pct": float(series.diff().std() * 100.0),
            "max_drawdown_pct": float(abs(m.max_drawdown(list(1.0 + series))) * 100.0),
            "trades": int(len(leg_trades)),
            "cost": float(leg_trades["cost"].sum()) if "cost" in leg_trades else 0.0,
            "_returns": series.diff().dropna(),  # for beta, dropped before serialization
        }
    return report


def _leg_betas(legs: Dict[str, Any], benchmark_returns) -> None:
    """Add each leg's beta against the benchmark, in place.

    Separate from :func:`_leg_report` because the benchmark is aligned later - and
    because a leg beta is exactly the number that distinguishes "no exposure" from
    "two exposures cancelling", so it is worth its own step rather than a side effect.
    """
    for leg in legs.values():
        leg_returns = leg.pop("_returns", None)
        if benchmark_returns is None or leg_returns is None or len(leg_returns) < 2:
            leg["beta"] = None
            leg["benchmark_correlation"] = None
            continue
        paired = pd.concat(
            [pd.Series(leg_returns.to_numpy()), pd.Series(pd.Series(benchmark_returns).to_numpy())],
            axis=1,
            keys=["leg", "bench"],
        ).dropna()
        if len(paired) < 2 or paired["bench"].std() == 0 or paired["leg"].std() == 0:
            leg["beta"] = None
            leg["benchmark_correlation"] = None
            continue
        leg["beta"] = float(paired["leg"].cov(paired["bench"]) / paired["bench"].var())
        leg["benchmark_correlation"] = float(paired["leg"].corr(paired["bench"]))


def _benchmark_returns(closes, equity_times) -> Optional[pd.Series]:
    """Benchmark returns sampled at the equity curve's own instants.

    Positional, not date-indexed, because that is the only alignment the metrics can
    use: the equity curve reaches them as a bare list of floats, so its returns carry
    a RangeIndex and a date-indexed benchmark would join to nothing at all - silently,
    producing an empty regression rather than an error.

    Reindexed onto those instants and forward-filled, so a benchmark that does not
    trade on one of the universe's bars carries its last close rather than dropping
    the step and shifting every later pairing by one.
    """
    if closes is None or closes.empty or len(equity_times) < 2:
        return None
    stamps = pd.DatetimeIndex([t for t in equity_times[1:]])
    aligned = closes.reindex(closes.index.union(stamps)).ffill().reindex(stamps)
    returns = aligned.pct_change()
    if returns.notna().sum() < 2:
        return None
    # Drop the index: the pairing is by position against the equity curve's own steps.
    return pd.Series(returns.to_numpy(), dtype="float64")


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
    #: What the sizer asked for versus what was actually tradeable - see :class:`_Execution`.
    execution: Dict[str, Any] = field(default_factory=dict)
    #: Per-side realized performance for a long/short book - see :func:`_leg_report`.
    #: Empty for a long-only run, which has nothing to decompose.
    legs: Dict[str, Any] = field(default_factory=dict)
    #: The benchmark returns this run scored against, already aligned positionally to
    #: ``equity_curve``. Exposed so a caller that recomputes metrics on *this* curve
    #: (the walk-forward does, to fold in its own trial counts) can reuse the alignment
    #: instead of deriving a second one - two alignments of the same series is how the
    #: two quietly stop matching.
    benchmark_returns: Optional[Any] = None


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
        benchmark: Optional[str] = None,
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
        bench = self._benchmark_closes(benchmark, timeframe, start, end)
        return self._simulate(
            ((s, b) for s, b in data.items()), start, end, initial_capital, trade_from, bench
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
        """Backtest from a :class:`~tradeflow.data.scan.BarSource`, **fetching** one symbol at a time.

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

    def _benchmark_closes(self, benchmark: Optional[str], timeframe, start, end):
        """The benchmark's close series, or ``None`` when there is nothing usable.

        A benchmark that cannot be fetched degrades to no benchmark rather than
        failing the run - but it says so, because silently scoring alpha against
        nothing is exactly the reading this whole path got wrong before.
        """
        if not benchmark:
            return None
        try:
            bars = self.data_client.get_bars([benchmark], timeframe, start, end)
        except Exception as exc:  # noqa: BLE001 - a missing benchmark is not a failed run
            logger.warning("Benchmark %s unavailable (%s); alpha/beta/IR will be unavailable", benchmark, exc)
            return None
        frame = bars.get(benchmark)
        if frame is None or frame.empty or "close" not in frame:
            logger.warning("Benchmark %s returned no bars; alpha/beta/IR will be unavailable", benchmark)
            return None
        return frame["close"]

    def _simulate(
        self, symbol_bars, start, end, initial_capital: float, trade_from=None, benchmark_closes=None
    ) -> BacktestResult:
        """Simulate the whole universe on one clock against one capital pool."""
        panels, market_data, master = self._prepare(symbol_bars)
        # A merged-timeline step is the unit for *both* carry accrual and the equity
        # curve, so both have to annualize on the merged timeline's own rate.
        self._step_periods_per_year = self._step_rate(panels, master)
        all_trades, equity_curve, equity_times, execution, legs = self._replay(
            panels, master, initial_capital, trade_from
        )

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

        aligned_benchmark = _benchmark_returns(benchmark_closes, equity_times)
        _leg_betas(legs, aligned_benchmark)
        net_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
        total_cost = float(trades_df["cost"].sum()) if "cost" in trades_df else 0.0
        final_capital = initial_capital + net_pnl
        # Cost belongs beside the fill diagnostics: both answer "what did executing
        # this actually take out of the book". Gross profit is the honest denominator -
        # 3% of capital in cost is fine against 40% gross and fatal against 4%.
        execution["total_cost"] = total_cost
        execution["gross_profit"] = float(net_pnl + total_cost)
        # The shape of the book that was actually validated. All three shipped
        # strategies declare max_positions=1, so a run over a scanned universe of
        # sixty names can validate a one-position book without ever saying so.
        limits = self.strategy.position_limits()
        execution["max_positions"] = limits.get("max_positions")
        execution["universe_size"] = len(market_data)
        execution["symbols_traded"] = int(trades_df["symbol"].nunique()) if not trades_df.empty else 0
        metrics = performance.compute_backtest_metrics(
            trades_df,
            equity_curve,
            initial_capital,
            final_capital,
            market_data,
            start=start,
            end=end,
            # The curve is sampled per merged-timeline step, so it annualizes on that
            # timeline's rate — not the daily default, and not the raw timeframe.
            periods_per_year=self._step_periods_per_year,
            benchmark_returns=aligned_benchmark,
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
            execution=execution,
            benchmark_returns=aligned_benchmark,
            legs=legs,
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
            raise BacktestError(f"backtest failed for all {attempted} symbol(s); first error - {failures[0]}")
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
            raise BacktestError(f"backtest failed for all {attempted} symbol(s); first error - {failures[0]}")
        return panels, market_data, master

    def _step_rate(self, panels: Dict[str, _Panel], master: pd.DatetimeIndex) -> float:
        """Steps-per-year of the merged timeline, for annualizing per-step quantities.

        The strategy's timeframe gives the rate of a *single* symbol's bars. The
        merged timeline is the union of every symbol's timestamps, so it is at least
        as dense and is strictly denser whenever symbols don't share one grid —
        halts, differing listing calendars, a mixed-venue universe. Annualizing a
        per-step series at the single-symbol rate would then understate the sampling
        frequency, inflating Sharpe and volatility by ``sqrt(density)`` and
        understating short carry by ``density``.

        Correcting by the observed density keeps an aligned universe (the common
        case) exactly at its timeframe rate, since the ratio is then 1.
        """
        base = getattr(self, "_periods_per_year", None) or m.TRADING_DAYS_PER_YEAR
        if not panels or len(master) < 2:
            return float(base)
        # The densest symbol is the best available estimate of one symbol's own grid;
        # any symbol with a shorter history would overstate the density.
        densest = max(len(p.opens) for p in panels.values())
        if densest < 2:
            return float(base)
        return float(base) * (len(master) - 1) / (densest - 1)

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

        rows = data.index.get_indexer(master)
        return _Panel(
            opens=data["open"].to_numpy(),
            timestamps=data.index.to_numpy(),
            highs=data["high"].to_numpy(),
            lows=data["low"].to_numpy(),
            closes=data["close"].to_numpy(),
            sig=sig,
            score=score,
            rows=rows,
            adv=adv,
            vol=vol,
            last_timestamp=data.index[-1],
            last_step=int(np.nonzero(rows >= 0)[0][-1]),
        )

    def _replay(self, panels: Dict[str, _Panel], master, initial_capital: float, trade_from=None):
        """Walk the merged timeline once: mark, exit, rank, admit, record."""
        book = _Book(cash=initial_capital)
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [initial_capital]
        # One timestamp per recorded equity point. The curve is a bare list of floats
        # by the time metrics see it, so alpha/beta against a date-indexed benchmark
        # can only be computed if something remembers which instants those floats
        # belong to - and nothing did.
        equity_times: List[Any] = [None]
        # Realized-plus-unrealized P&L per side, sampled on the same steps as the equity
        # curve. Built here rather than from the trade list because a curve
        # reconstructed from closed trades only sees P&L at exit - which is precisely
        # what volatility, drawdown and beta are most distorted by.
        realized_by_side: Dict[str, float] = {signals.BUY: 0.0, signals.SELL: 0.0}
        leg_curves: Dict[str, List[float]] = {signals.BUY: [0.0], signals.SELL: [0.0]}
        if not panels:
            return (
                trades,
                equity_curve,
                equity_times,
                book.execution.as_dict(),
                _leg_report(leg_curves, trades, initial_capital),
            )

        cutoff = _align_tz(trade_from, master) if trade_from is not None else None
        limits = self.strategy.position_limits()
        max_positions = limits["max_positions"]
        max_total_risk = limits["max_total_risk"]
        max_gross_exposure = limits.get("max_gross_exposure")
        min_notional = limits.get("min_notional")
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
                    pos,
                    panel.sig[i],
                    panel.opens[i],
                    panel.highs[i],
                    panel.lows[i],
                    panel.timestamps[i],
                    exit_cost,
                )
                if closed is not None:
                    trades.append(closed)
                    realized_by_side[pos["side"]] += closed["pnl"]
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
                        symbol,
                        panel.sig[i],
                        panel.opens[i],
                        panel.timestamps[i],
                        book,
                        equity_now,
                        max_total_risk,
                        max_gross_exposure,
                        min_notional,
                        panel.adv,
                        panel.vol,
                        i,
                    )
                    if pos is None:
                        continue
                    pos["entry_k"] = k
                    book.cash -= pos["notional"] + pos["entry_cost"]
                    book.positions[symbol] = pos

            if trading:
                equity_curve.append(book.equity())
                equity_times.append(master[k])
                unrealized = {signals.BUY: 0.0, signals.SELL: 0.0}
                for pos in book.positions.values():
                    direction = 1 if pos["side"] == signals.BUY else -1
                    unrealized[pos["side"]] += (
                        (pos["last_price"] - pos["entry_price"]) * pos["size"] * direction
                    )
                for side, curve in leg_curves.items():
                    curve.append(realized_by_side[side] + unrealized[side])

        # Force-close whatever is still open, each at its own last bar.
        for symbol in order:
            pos = book.positions.get(symbol)
            if pos is None:
                continue
            panel = panels[symbol]
            price = panel.closes[-1]
            final_cost = self._trade_cost(symbol, pos["size"], price, panel.adv, panel.vol, -1) + self._carry(
                pos, panel.last_step
            )
            closed = self._close(pos, price, panel.last_timestamp, "END_OF_PERIOD", final_cost)
            trades.append(closed)
            book.cash += pos["notional"] + closed["gross_pnl"] - final_cost
            del book.positions[symbol]
        if len(equity_curve) > 1:
            equity_curve[-1] = book.equity()

        return (
            trades,
            equity_curve,
            equity_times,
            book.execution.as_dict(),
            _leg_report(leg_curves, trades, initial_capital),
        )

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
        # entry_k/exit_bar are merged-timeline steps, so the divisor must be the
        # merged timeline's rate rather than a single symbol's timeframe rate.
        rate = getattr(self, "_step_periods_per_year", None) or self._periods_per_year
        held_years = max(exit_bar - position.get("entry_k", exit_bar), 0) / rate
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
        max_gross_exposure: Optional[float],
        min_notional: Optional[float],
        adv: Optional[np.ndarray],
        vol: Optional[np.ndarray],
        i: int,
    ) -> Optional[Dict[str, Any]]:
        """Size and admit one candidate, or return None if the book cannot fund it.

        **Shorts are fully cash-collateralized.** Opening debits the whole notional
        regardless of side, rather than crediting short proceeds against margin as a
        real margin account would. This is deliberate and conservative: it charges a
        short the same buying power as the equivalent long, so the book can never
        take on leverage the engine isn't modelling. The consequence to keep in mind
        is that short capacity is understated, and a long-short configuration is
        therefore compared against a long-only one on slightly unequal footing.
        Entry and exit are symmetric, so realized P&L is unaffected either way.
        """
        # Free cash is the buying power; equity is the whole book. Sizing against
        # cash is what makes positions actually compete for the same dollars.
        account = AccountSnapshot(
            cash=book.cash,
            equity=equity,
            buying_power=book.cash,
            portfolio_value=equity,
        )
        # Requested before rounding, so the gap between intent and fill is measurable
        # rather than inferred. Whole shares: the live path floors too, and the two
        # clocks must agree about what is fillable.
        requested = self.sizer.size(symbol, price, account)
        size = round_quantity(requested)
        if size <= 0:
            if requested > 0:
                book.execution.rounded_to_zero += 1
            return None
        if min_notional and size * price < min_notional:
            # A venue floor is an execution fact, not a preference: an order below it
            # would be refused, so filling it here would validate a book that could not
            # be traded.
            book.execution.below_min_notional += 1
            return None
        # The affordability check must include what this fill will cost to enter,
        # not just its notional - otherwise a maximally-sized position can push
        # cash negative once commission/spread/impact are subtracted.
        entry_cost = self._trade_cost(symbol, size, price, adv, vol, i)
        if size * price + entry_cost > book.cash:
            return None

        stop_pct = self.strategy.config["stop_loss"]
        take_pct = self.strategy.config["take_profit"]
        # Portfolio-level risk budget. calculate_position_size caps a *single*
        # position against max_total_risk; nothing capped the book as a whole, so
        # "max_total_risk" was per-symbol in practice.
        #
        # What it bounds is loss-at-stop, not deployed notional: at a 0.5% stop a 5%
        # budget admits ten times equity in notional before this gate binds. It is
        # not a gross exposure cap, and max_gross_exposure below is the one that is.
        risk = size * price * stop_pct
        if max_total_risk and book.open_risk() + risk > equity * max_total_risk:
            return None

        # Gross notional cap, off unless configured. Free cash already holds the book
        # near 1x on its own (shorts are fully collateralized, per the docstring), so
        # this binds when a config wants to sit deliberately below that.
        if max_gross_exposure and book.gross_exposure() + size * price > equity * max_gross_exposure:
            return None

        if signal == signals.BUY:
            stop, take = price * (1 - stop_pct), price * (1 + take_pct)
        else:
            stop, take = price * (1 + stop_pct), price * (1 - take_pct)

        # Counted here rather than at the sizing call, because everything between the
        # two can still decline the entry - and a position the book refused to fund is
        # not evidence about whether rounding was affordable.
        book.execution.requested_notional += requested * price
        book.execution.filled_notional += size * price
        book.execution.filled += 1
        return {
            "symbol": symbol,
            "side": signal,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp,
            "stop_loss": stop,
            "take_profit": take,
            "notional": size * price,
            "entry_cost": entry_cost,
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
