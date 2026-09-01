"""Base class for trading strategies - the part where you encode your beautiful
theory about the market, which the backtester then evaluates without mercy.

A :class:`Strategy` is responsible for exactly three things:

1. **Indicators** - turn raw OHLCV into the columns it needs (:meth:`process_data`).
2. **Conviction** - turn those columns into a single continuous **score** per bar
   (:meth:`calculate_scores`): signed (``+`` bullish, ``-`` bearish), with
   magnitude = strength. This is the one source of truth for the strategy's view.
   The discrete ``BUY/SELL/HOLD`` the trade clock consumes is *derived* from the
   score by the base class (:meth:`generate_signals`); the cross-sectional alpha
   layer reads the same score. There is no parallel discrete-signal path.
3. **Sizing & risk** - decide how big a position should be and reject signals
   that conflict with the current book.

Deliberately *not* a strategy's job (separation of concerns):

* fetching data            -> ``tradeflow/marketdata``
* placing orders           -> ``tradeflow/execution``
* simulating fills / P&L   -> ``tradeflow/engine``
* computing performance    -> ``tradeflow/analytics``

The same strategy object is driven unchanged by the backtest and live engines.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from tradeflow.strategies import signals
from tradeflow.utils.timeutils import match_index_tz

logger = logging.getLogger(__name__)

#: Whether a live strategy opens a position implied by the score but whose entry edge
#: it never saw — a crossing lost to a rejected bar, a dropped stream, a restart, or
#: the warm-up history. On by default: a trend-follower started mid-trend should hold
#: the trend, not sit flat until the next crossing. Set ``reaffirm_entries=False`` in a
#: strategy's config to wait for a fresh edge instead. Exits ignore this entirely.
REAFFIRM_ENTRIES_DEFAULT = True


@dataclass(frozen=True)
class ScoreThresholds:
    """Hysteresis bands that turn a continuous score into discrete entries/exits.

    Semantics (long side): enter long when ``score > enter_long``; stay long while
    ``score > exit_long``; exit when ``score <= exit_long``. The short side mirrors
    it. Defaults are all 0 - pure sign: long while score > 0. A strategy with
    asymmetric entry/exit levels (e.g. enter oversold, exit overbought) overrides
    :meth:`Strategy.signal_thresholds`.
    """

    enter_long: float = 0.0
    exit_long: float = 0.0
    enter_short: float = 0.0
    exit_short: float = 0.0


#: Portfolio limits applied when a strategy declares no ``position_limits``.
#:
#: The two fractions measure different things and read alike:
#:
#: * ``max_total_risk`` is a **risk budget** - the fraction of equity the book gives
#:   up if every open position stops out. A position contributes
#:   ``notional x stop_loss``, so a tight stop buys a lot of notional for very
#:   little budget. It bounds loss-at-stop, not how much is deployed.
#: * ``max_gross_exposure`` is a **notional cap** - marked gross notional over
#:   equity, shorts counted by magnitude. ``None`` (the default) leaves free cash as
#:   the only bound on deployed notional.
#: * ``min_notional`` is an **execution floor** in dollars, not a fraction: a venue or
#:   broker minimum below which an order would be refused. ``None`` (the default) keeps
#:   the historical behaviour of assuming any positive size is fillable, which stops
#:   being a safe assumption as capital shrinks - at $4,000 across a wide universe the
#:   sizer routinely asks for positions a real account could not open.
#:
#: They live here rather than in the backtest engine because both clocks enforce
#: them, and the trade clock cannot import the engine.
DEFAULT_POSITION_LIMITS: Dict[str, Any] = {
    "max_positions": 5,
    "max_position_size": 1500.0,
    "max_total_risk": 0.05,
    "max_gross_exposure": None,
    #: Ceiling on |long - short| as a fraction of equity. Distinct from
    #: ``max_gross_exposure``, which bounds long + short: a book can sit inside a gross
    #: cap while being entirely one-directional, so a long/short strategy needs both.
    "max_net_exposure": None,
    "min_notional": None,
}


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    #: Subclasses declare tunable parameters here for validation/optimization.
    PARAM_RANGES: Dict[str, Dict[str, Any]] = {}

    #: Whether the strategy may hold short positions. Long-only strategies never
    #: emit SELL entries; a negative score simply means "flat" (exit any long).
    LONG_ONLY: bool = True

    @classmethod
    def create_with_defaults(cls) -> "Strategy":
        """Construct an instance using each parameter's declared default value.

        The single construction path used by the CLI, MCP server, and research
        agent, so every entry point builds a strategy the same way.
        """
        config = {param: spec["default"] for param, spec in cls.PARAM_RANGES.items()}
        return cls(config)

    def __init__(self, config: Dict[str, Any]):
        """Initialize with a configuration dict.

        Recognized keys (strategy-specific keys may be added freely):
            risk_per_trade: fraction of capital risked per trade
            stop_loss / take_profit: fractional distances from entry price
            position_limits: optional {max_positions, max_position_size,
                max_total_risk, max_gross_exposure}. max_total_risk is a
                stop-weighted risk budget (loss if everything stops out), not a
                cap on deployed notional; max_gross_exposure is the notional cap.
                Both are enforced across the book by the backtest engine.
            reaffirm_entries: live-only; open a position the score implies even when
                its entry edge was missed (default True). See
                :data:`REAFFIRM_ENTRIES_DEFAULT`.
        """
        self.config = config

        # symbol -> open position details, used for live signal validation.
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Rolling per-symbol buffers, used only in live/real-time mode.
        self.real_time_data: Dict[str, pd.DataFrame] = {}
        self.last_processed_data: Dict[str, pd.DataFrame] = {}
        self.max_buffer_size = 1000

        # Cache the lookback the indicators need so the data layer can warm up.
        self.config["required_lookback_periods"] = self.calculate_required_lookback()

        self._validate_parameters()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def _validate_parameters(self) -> None:
        """Coerce and range-check config values declared in ``PARAM_RANGES``."""
        for param, value in self.config.items():
            spec = self.PARAM_RANGES.get(param)
            if spec is None:
                continue

            param_type = spec["type"]
            try:
                if param_type == "int":
                    value = int(value)
                elif param_type == "float":
                    value = float(value)
                self.config[param] = value
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{param}' value {value!r} cannot be converted to {param_type}")

            # A spec without min/max is a *pinned* parameter: typed and required, but
            # not searched (see ParameterSpace.searchable). There is no range to check.
            if "min" in spec and "max" in spec and not (spec["min"] <= value <= spec["max"]):
                raise ValueError(
                    f"Parameter '{param}' value {value} is outside valid range [{spec['min']}, {spec['max']}]"
                )

    # ------------------------------------------------------------------ #
    # Abstract hooks implemented by concrete strategies
    # ------------------------------------------------------------------ #
    @abstractmethod
    def initialize(self) -> None:
        """Validate parameters / set up indicators before trading begins."""

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return ``data`` enriched with the indicator columns the strategy needs."""

    @abstractmethod
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Return a continuous, signed conviction score per bar of processed ``data``.

        Positive = bullish, negative = bearish, magnitude = strength; NaN where
        indicators aren't yet valid. This is the strategy's single source of truth:
        the discrete trade signal is derived from it (:meth:`generate_signals`) and
        the cross-sectional alpha layer scales the same score.
        """

    @abstractmethod
    def calculate_required_lookback(self) -> int:
        """Number of bars required before indicators/signals are valid."""

    # ------------------------------------------------------------------ #
    # Signal derivation (shared - never overridden by a strategy)
    # ------------------------------------------------------------------ #
    def signal_thresholds(self) -> ScoreThresholds:
        """The hysteresis bands used to derive discrete signals from the score.

        Defaults to pure sign (enter long when score > 0). Strategies with
        asymmetric entry/exit levels override this.
        """
        return ScoreThresholds()

    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        """Derive the discrete ``BUY/SELL/HOLD`` stream from :meth:`calculate_scores`.

        Walks the score series once, tracking the desired position direction with
        hysteresis: a fresh entry emits ``BUY``/``SELL``; returning to flat emits
        ``CLOSE_BUY``/``CLOSE_SELL``; a direct flip closes first (the opposite entry
        follows on the next bar). While a direction is held the bar emits ``HOLD``,
        so entries are edge-triggered - re-affirmation is left to the engine, which
        dedupes against the open position.
        """
        return self._walk_scores(data)[0]

    def _walk_scores(self, data: pd.DataFrame) -> tuple:
        """Return ``(signal per bar, the direction the score implies at the last bar)``.

        The direction is the part edges alone cannot express. An edge says *change*;
        the direction says *what should be true now*, which is the only thing that can
        be compared against what is actually held.
        """
        if data.empty:
            return {}, 0

        scores = self.calculate_scores(data)
        thresholds = self.signal_thresholds()
        long_only = self.LONG_ONLY

        out: Dict[Any, str] = {}
        state = 0  # desired direction implied by the score: +1 long, -1 short, 0 flat
        values = scores.to_numpy()
        index = scores.index
        for i in range(len(values)):
            score = values[i]
            if score != score:  # NaN - indicators not yet valid
                out[index[i]] = signals.HOLD
                continue
            target = self._desired_direction(score, state, thresholds, long_only)
            signal, state = self._transition(state, target)
            out[index[i]] = signal
        return out, state

    @staticmethod
    def _desired_direction(score: float, state: int, th: ScoreThresholds, long_only: bool) -> int:
        """Target direction for this bar given the score and the current direction."""
        # Hold an existing position until its exit band trips.
        if state == 1 and score > th.exit_long:
            return 1
        if state == -1 and score < th.exit_short:
            return -1
        # Flat, or just released: look for a fresh entry.
        if score > th.enter_long:
            return 1
        if not long_only and score < th.enter_short:
            return -1
        return 0

    @staticmethod
    def _transition(state: int, target: int) -> tuple:
        """Map a (current -> target) direction change to a signal + the new state."""
        if target == state:
            return signals.HOLD, state
        if state == 0:  # entering from flat
            return (signals.BUY, 1) if target == 1 else (signals.SELL, -1)
        # Leaving a position (exit to flat, or a flip): always close first and go
        # flat; a flip re-enters on the next bar once state is 0.
        return (signals.CLOSE_BUY if state == 1 else signals.CLOSE_SELL), 0

    # ------------------------------------------------------------------ #
    # Position sizing & risk
    # ------------------------------------------------------------------ #
    def position_limits(self) -> Dict[str, Any]:
        """Configured portfolio limits merged over :data:`DEFAULT_POSITION_LIMITS`.

        One accessor so the backtest engine and the live trader read the same merged
        dict, and so a config that sets only some of the keys gets defaults for the
        rest rather than a ``KeyError`` at the first entry.
        """
        return {**DEFAULT_POSITION_LIMITS, **(self.config.get("position_limits") or {})}

    def calculate_position_size(self, capital: float, price: float, risk_factor: float = 1.0) -> float:
        """Size a position from risk-per-trade, capped by configured limits.

        The result is the smallest of three constraints: the risk-per-trade
        target, the max notional per position, and the max total portfolio risk.

        That third one is applied to *one* position in isolation, because sizing
        has no view of the open book - it answers "could this position alone consume
        the whole risk budget?". Enforcing the budget across the book is the
        backtest engine's job on the research clock and the live trader's on the
        trade clock; both read :meth:`position_limits`. ``max_gross_exposure`` is
        absent here for the same reason, and more so: a cap on total deployed
        notional says nothing about what any single position may take.
        """
        risk_amount = capital * self.config["risk_per_trade"] * risk_factor
        stop_loss_pct = self.config["stop_loss"]
        target_size = risk_amount / (price * stop_loss_pct)

        limits = self.position_limits()
        max_size_by_notional = limits["max_position_size"] / price
        max_size_by_total_risk = (capital * limits["max_total_risk"]) / (price * stop_loss_pct)

        return min(target_size, max_size_by_notional, max_size_by_total_risk)

    def check_exit_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Flag stop-loss / take-profit exits for currently open positions.

        Returns a DataFrame aligned to ``data.index`` with ``exit_signal``,
        ``exit_reason`` and ``signal_type`` columns.
        """
        exits = pd.DataFrame(index=data.index)
        exits["exit_signal"] = False
        exits["exit_reason"] = ""
        exits["signal_type"] = signals.HOLD

        for position in self.positions.values():
            side = position.get("side")
            if side == signals.BUY:
                stop_hit = data["close"] <= position["stop_loss"]
                take_hit = data["close"] >= position["take_profit"]
                close_signal = signals.CLOSE_BUY
            elif side == signals.SELL:
                stop_hit = data["close"] >= position["stop_loss"]
                take_hit = data["close"] <= position["take_profit"]
                close_signal = signals.CLOSE_SELL
            else:
                continue

            for mask, reason in ((stop_hit, "STOP_LOSS"), (take_hit, "TAKE_PROFIT")):
                exits.loc[mask, "exit_signal"] = True
                exits.loc[mask, "exit_reason"] = reason
                exits.loc[mask, "signal_type"] = close_signal

        return exits

    def validate_signal(self, signal: str, symbol: str, price: float) -> bool:
        """Reject signals that conflict with the current position book."""
        if signal in signals.EXIT_SIGNALS:
            position = self.positions.get(symbol)
            if position is None:
                return False
            return (signal == signals.CLOSE_BUY and position["side"] == signals.BUY) or (
                signal == signals.CLOSE_SELL and position["side"] == signals.SELL
            )

        # Don't stack a second position on the same side of an existing one.
        existing = self.positions.get(symbol)
        if existing and existing["side"] == signal:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Real-time processing (live mode)
    # ------------------------------------------------------------------ #
    def _reaffirm(self, symbol: str, signal: str, direction: int) -> str:
        """Re-state an edge that was missed, by comparing intent against the book.

        Signals are edge-triggered: an entry is emitted on the bar the score crosses
        and never again. Live, that edge can be missed - a bar rejected by the quality
        guard, a dropped stream, a restart, or simply a crossing that happened inside
        the warm-up history. Afterwards the score still says "should be long" while the
        bar emits ``HOLD`` forever, and the position is never opened. The mirror case
        is worse: a missed exit leaves a real position that nothing will close.

        So live mode compares the direction the score implies against the position book
        (which :class:`~tradeflow.execution.live_trader.LiveTrader` keeps synced with
        broker truth) and re-states the difference. Where an edge says *change*, this
        says *what should be true now* - the loop converges on the intended book
        instead of depending on having caught one specific bar.

        This is live-only. The backtest walks the same scores through
        :meth:`generate_signals`, where its book is derived from those very signals and
        so can never disagree - the two paths differ only in the case where the live
        one has fallen behind.

        A flip closes first and re-enters on the next bar, exactly as
        :meth:`_transition` does, rather than reversing in one step.

        **Entries are gated; exits never are.** ``reaffirm_entries=False`` makes the
        strategy wait for a fresh crossing rather than opening a position on a signal
        whose edge it did not witness - most visibly, an engine started into an
        established trend stays flat until the next entry. That is a legitimate
        preference about what a strategy trades. Declining to *close* a position the
        strategy no longer wants is not a preference, it is a stuck position, so the
        exit side stays unconditional whatever the flag says.
        """
        if signal != signals.HOLD:
            return signal

        desired = {1: signals.BUY, -1: signals.SELL}.get(direction)
        held = self.positions.get(symbol)
        held_side = held.get("side") if held else None

        if desired == held_side:  # book already matches intent (including both flat)
            return signals.HOLD
        if held_side is not None:  # holding something we should not be: close it
            return signals.CLOSE_BUY if held_side == signals.BUY else signals.CLOSE_SELL
        if not self.config.get("reaffirm_entries", REAFFIRM_ENTRIES_DEFAULT):
            return signals.HOLD
        return desired or signals.HOLD

    def process_bar(self, symbol: str, bar: Dict[str, float], timestamp: datetime) -> Optional[str]:
        """Fold one streamed OHLCV bar into the rolling buffer and emit a signal.

        ``bar`` must contain ``open``/``high``/``low``/``close``/``volume``. Returns
        the latest signal once enough history has accumulated, otherwise ``None``.
        Never raises - real-time processing must not break the stream.
        """
        try:
            buffer = self.real_time_data.get(symbol)
            if buffer is None:
                buffer = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                buffer.index.name = "timestamp"

            row = {key: bar[key] for key in ("open", "high", "low", "close", "volume")}
            # Warm-up history arrives localized; a streamed bar may not be. One naive
            # timestamp in an otherwise aware index makes every subsequent comparison
            # raise, and the guard below would turn that into permanent silence.
            timestamp = match_index_tz(timestamp, buffer.index)
            buffer = pd.concat([buffer, pd.DataFrame([row], index=[timestamp])])
            if len(buffer) > self.max_buffer_size:
                buffer = buffer.tail(self.max_buffer_size)
            self.real_time_data[symbol] = buffer

            if len(buffer) < self.config.get("required_lookback_periods", 20):
                return None

            processed = self.process_data(buffer.copy())
            self.last_processed_data[symbol] = processed

            emitted, direction = self._walk_scores(processed)
            latest = self._reaffirm(symbol, self._latest_signal(emitted), direction)
            return latest if self.validate_signal(latest, symbol, bar["close"]) else signals.HOLD
        except Exception:  # noqa: BLE001 - never break the stream
            # Swallowing keeps the stream alive, but a strategy that raises on every
            # bar emits nothing and looks exactly like one with no opinion. Say so.
            logger.warning("Discarding bar for %s: the strategy raised", symbol, exc_info=True)
            return None

    def process_real_time_data(
        self, symbol: str, price: float, volume: float, timestamp: datetime
    ) -> Optional[str]:
        """Tick-style convenience wrapper: treat a single price as a flat bar."""
        flat_bar = {"open": price, "high": price, "low": price, "close": price, "volume": volume}
        return self.process_bar(symbol, flat_bar, timestamp)

    def warm_up(self, symbol: str, processed_history: pd.DataFrame) -> None:
        """Seed a symbol's real-time buffer with pre-processed historical bars."""
        if not processed_history.empty:
            self.real_time_data[symbol] = processed_history.tail(self.max_buffer_size).copy()

    @staticmethod
    def _latest_signal(generated) -> str:
        """Extract the most recent signal from a dict/Series/list of signals."""
        if isinstance(generated, dict) and generated:
            return generated[max(generated.keys())]
        if hasattr(generated, "iloc") and len(generated) > 0:
            return generated.iloc[-1]
        if isinstance(generated, (list, tuple)) and generated:
            return generated[-1]
        return signals.HOLD

    def get_real_time_buffer(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.real_time_data.get(symbol)

    def clear_real_time_buffer(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self.real_time_data.clear()
            self.last_processed_data.clear()
        else:
            self.real_time_data.pop(symbol, None)
            self.last_processed_data.pop(symbol, None)
