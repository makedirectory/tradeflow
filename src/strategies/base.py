"""Base class for trading strategies - the part where you encode your beautiful
theory about the market, which the backtester then evaluates without mercy.

A :class:`Strategy` is responsible for exactly three things:

1. **Indicators** - turn raw OHLCV into the columns it needs (:meth:`process_data`).
2. **Signals** - turn those columns into BUY/SELL/HOLD decisions
   (:meth:`generate_signals`).
3. **Sizing & risk** - decide how big a position should be and reject signals
   that conflict with the current book.

Deliberately *not* a strategy's job (separation of concerns):

* fetching data            -> ``src/marketdata``
* placing orders           -> ``src/execution``
* simulating fills / P&L   -> ``src/engine``
* computing performance    -> ``src/analytics``

The same strategy object is driven unchanged by the backtest and live engines.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.strategies import signals


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    #: Subclasses declare tunable parameters here for validation/optimization.
    PARAM_RANGES: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create_with_defaults(cls) -> "Strategy":
        """Construct an instance using each parameter's declared default value.

        The single construction path used by the CLI, MCP server, and research
        agent, so every entry point builds a strategy the same way.
        """
        config = {param: spec["default"] for param, spec in cls.PARAM_RANGES.items()}
        return cls(config)

    def __init__(self, config: Dict[str, Any]):
        """Initialise with a configuration dict.

        Recognised keys (strategy-specific keys may be added freely):
            risk_per_trade: fraction of capital risked per trade
            stop_loss / take_profit: fractional distances from entry price
            position_limits: optional {max_positions, max_position_size,
                max_total_risk}
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

            if not (spec["min"] <= value <= spec["max"]):
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
    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        """Map each timestamp in processed ``data`` to a signal string."""

    @abstractmethod
    def calculate_required_lookback(self) -> int:
        """Number of bars required before indicators/signals are valid."""

    # ------------------------------------------------------------------ #
    # Position sizing & risk
    # ------------------------------------------------------------------ #
    def calculate_position_size(self, capital: float, price: float, risk_factor: float = 1.0) -> float:
        """Size a position from risk-per-trade, capped by configured limits.

        The result is the smallest of three constraints: the risk-per-trade
        target, the max notional per position, and the max total portfolio risk.
        """
        risk_amount = capital * self.config["risk_per_trade"] * risk_factor
        stop_loss_pct = self.config["stop_loss"]
        target_size = risk_amount / (price * stop_loss_pct)

        limits = self.config.get(
            "position_limits",
            {"max_positions": 5, "max_position_size": 1500.0, "max_total_risk": 0.05},
        )
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
            buffer = pd.concat([buffer, pd.DataFrame([row], index=[timestamp])])
            if len(buffer) > self.max_buffer_size:
                buffer = buffer.tail(self.max_buffer_size)
            self.real_time_data[symbol] = buffer

            if len(buffer) < self.config.get("required_lookback_periods", 20):
                return None

            processed = self.process_data(buffer.copy())
            self.last_processed_data[symbol] = processed

            latest = self._latest_signal(self.generate_signals(processed))
            return latest if self.validate_signal(latest, symbol, bar["close"]) else signals.HOLD
        except Exception:  # noqa: BLE001 - never break the stream
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
