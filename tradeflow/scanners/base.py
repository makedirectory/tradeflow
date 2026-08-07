"""Base class for universe scanners - the bouncers deciding which symbols are even
worth a strategy's attention today.

A scanner answers a different question than a strategy: *which symbols are worth
trading right now?* It scores each symbol's recent bars and emits a per-symbol
scan signal (``SCANNER_BUY`` / ``SCANNER_SELL`` / ``SCANNER_HOLD``).

Scanners operate on a single symbol's OHLCV frame at a time (the
:class:`~tradeflow.scanners.symbol_scanner.SymbolScanner` iterates the universe), which
keeps the interface simple and avoids MultiIndex bookkeeping.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from tradeflow.analytics import metrics

# Scan-signal vocabulary (distinct from trade signals so the two never mix).
SCANNER_BUY = "SCANNER_BUY"
SCANNER_SELL = "SCANNER_SELL"
SCANNER_HOLD = "SCANNER_HOLD"

_ACTIONABLE = frozenset({SCANNER_BUY, SCANNER_SELL})


class ScannerStrategy(ABC):
    """Abstract base for scanner strategies."""

    SCANNER_BUY = SCANNER_BUY
    SCANNER_SELL = SCANNER_SELL

    #: Tunable parameters (with min/max/step) for the optimizer.
    PARAM_RANGES: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)

    # ------------------------------------------------------------------ #
    # Abstract hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def initialize(self) -> None:
        """Optional setup before scanning."""

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return one symbol's OHLCV frame enriched with scan indicators."""

    @abstractmethod
    def generate_signals_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a frame aligned to ``data.index`` with a ``signal`` column.

        Should also include a numeric ``signal_strength`` column for ranking.
        """

    # ------------------------------------------------------------------ #
    # Shared behavior
    # ------------------------------------------------------------------ #
    def latest_signal(self, signals_df: pd.DataFrame) -> str:
        """The most recent scan signal, or ``SCANNER_HOLD`` if none/empty."""
        if signals_df.empty or "signal" not in signals_df:
            return SCANNER_HOLD
        signal = signals_df["signal"].iloc[-1]
        return signal if signal in _ACTIONABLE else SCANNER_HOLD

    def required_data_points(self) -> int:
        """Bars needed for a valid scan (lookback + current bar)."""
        return self.config.get("lookback_periods", 20) + 1

    def is_hit(self, returns: float, price_change: float) -> bool:
        """Whether a forward outcome counts as a successful signal (override per scanner)."""
        return returns > 0

    def evaluate_forward(
        self, signals_df: pd.DataFrame, forward: pd.DataFrame, hold_bars: int
    ) -> Dict[str, float]:
        """Score a symbol's signals against subsequent price action.

        Used by the optimizer to tune scanner parameters. Returns hit rate,
        average return, Sharpe and profit factor over the realized signals.
        """
        empty = {
            "hit_rate": 0.0,
            "avg_return": 0.0,
            "total_signals": 0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
        }
        if signals_df.empty or forward.empty:
            return empty

        closes = forward["close"]
        trade_returns: List[float] = []
        hits = 0

        for timestamp, row in signals_df.iterrows():
            if row["signal"] not in _ACTIONABLE:
                continue
            future = closes[closes.index > timestamp].head(hold_bars)
            if future.empty or timestamp not in closes.index:
                continue

            entry = closes.loc[timestamp]
            exit_price = future.iloc[-1]
            ret = (exit_price / entry - 1) * 100
            if row["signal"] == SCANNER_SELL:
                ret = -ret
            trade_returns.append(ret)
            if self.is_hit(ret, abs(ret)):
                hits += 1

        if not trade_returns:
            return empty

        series = pd.Series(trade_returns)
        return {
            "hit_rate": hits / len(trade_returns) * 100,
            "avg_return": float(series.mean()),
            "total_signals": len(trade_returns),
            "sharpe_ratio": metrics.sharpe_ratio(series),
            "profit_factor": metrics.profit_factor(series),
        }

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fill defaults and range-check values against ``PARAM_RANGES``."""
        validated = dict(config)
        for param, spec in self.PARAM_RANGES.items():
            if param not in validated:
                validated[param] = spec["default"]
                continue
            value = validated[param]
            if spec["type"] in ("int", "float"):
                value = int(value) if spec["type"] == "int" else float(value)
                if not (spec["min"] <= value <= spec["max"]):
                    raise ValueError(
                        f"Scanner parameter '{param}' value {value} outside [{spec['min']}, {spec['max']}]"
                    )
                validated[param] = value
        return validated
