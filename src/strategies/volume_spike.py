"""Volume-spike trend strategy.

Enters in the direction of a short/long EMA trend when a volume spike confirms
a move out of an RSI extreme. This is a clean re-implementation of the original
project's idea, built on the pure pandas/numpy indicators in
:mod:`src.indicators.indicators` (no TA-Lib).
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.indicators import indicators
from src.strategies import signals
from src.strategies.base import Strategy


class VolumeSpikeStrategy(Strategy):
    """Trend-following entries triggered by volume spikes out of RSI extremes."""

    #: Bars this strategy is designed to trade.
    TIMEFRAME = "5Min"

    # Each entry carries min/max/step so the optimizer can search the grid.
    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "volume_threshold": {"type": "float", "min": 1.05, "max": 2.0, "step": 0.05, "default": 1.05,
                             "description": "Volume / volume-MA ratio that defines a spike"},
        "price_change_threshold": {"type": "float", "min": 0.002, "max": 0.02, "step": 0.002, "default": 0.005,
                                   "description": "Minimum fractional price move to confirm a spike"},
        "short_ema_period": {"type": "int", "min": 5, "max": 15, "step": 1, "default": 9,
                             "description": "Short EMA period (trend)"},
        "long_ema_period": {"type": "int", "min": 16, "max": 30, "step": 1, "default": 21,
                            "description": "Long EMA period (trend)"},
        "rsi_period": {"type": "int", "min": 7, "max": 18, "step": 1, "default": 12,
                       "description": "RSI period"},
        "rsi_overbought": {"type": "int", "min": 55, "max": 75, "step": 5, "default": 65,
                           "description": "RSI overbought level"},
        "rsi_oversold": {"type": "int", "min": 25, "max": 45, "step": 5, "default": 35,
                         "description": "RSI oversold level"},
        "risk_per_trade": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.02,
                           "description": "Capital fraction risked per trade"},
        "stop_loss": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.01,
                      "description": "Stop-loss distance from entry (fraction)"},
        "take_profit": {"type": "float", "min": 0.02, "max": 0.10, "step": 0.02, "default": 0.02,
                        "description": "Take-profit distance from entry (fraction)"},
    }

    #: Volume-MA window used by the spike detector (also affects lookback).
    VOLUME_MA_PERIOD = 14

    @classmethod
    def create_with_defaults(cls) -> "VolumeSpikeStrategy":
        """Construct an instance using each parameter's default value."""
        config = {param: spec["default"] for param, spec in cls.PARAM_RANGES.items()}
        return cls(config)

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault(
            "position_limits",
            {"max_positions": 1, "max_position_size": 100.0, "max_total_risk": 0.01},
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        return max(
            self.config["rsi_period"] + 1,
            self.config["long_ema_period"] + 1,
            self.VOLUME_MA_PERIOD,
        )

    def initialize(self) -> None:
        if self.config["short_ema_period"] >= self.config["long_ema_period"]:
            raise ValueError(
                f"short_ema_period ({self.config['short_ema_period']}) must be < "
                f"long_ema_period ({self.config['long_ema_period']})"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        short_ema = indicators.calculate_ema(data["close"], self.config["short_ema_period"])
        long_ema = indicators.calculate_ema(data["close"], self.config["long_ema_period"])

        enriched = data.copy()
        enriched["rsi"] = indicators.calculate_rsi(data["close"], self.config["rsi_period"])
        enriched["volume_spike"] = indicators.calculate_volume_spike(
            data["volume"],
            data["close"],
            volume_ma_period=self.VOLUME_MA_PERIOD,
            volume_threshold=self.config["volume_threshold"],
            price_change_threshold=self.config["price_change_threshold"],
        )
        # +1 uptrend, -1 downtrend
        enriched["trend"] = (short_ema > long_ema).astype(int) * 2 - 1
        enriched["short_ema"] = short_ema
        enriched["long_ema"] = long_ema
        enriched["atr"] = indicators.calculate_atr(data)
        return enriched

    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        if data.empty:
            return {}

        spike = data["volume_spike"]
        uptrend = data["trend"] == 1
        downtrend = data["trend"] == -1
        trend_flip = data["trend"] != data["trend"].shift(1)

        rsi = data["rsi"]
        prev_rsi = rsi.shift(1)
        rsi_rising = rsi > prev_rsi
        rsi_falling = rsi < prev_rsi
        was_oversold = prev_rsi < self.config["rsi_oversold"]
        was_overbought = prev_rsi > self.config["rsi_overbought"]
        is_oversold = rsi < self.config["rsi_oversold"]
        is_overbought = rsi > self.config["rsi_overbought"]

        buy = spike & uptrend & was_oversold & rsi_rising
        sell = spike & downtrend & was_overbought & rsi_falling
        close_buy = (downtrend & is_overbought & spike) | (trend_flip & downtrend & rsi_falling & spike)
        close_sell = (uptrend & is_oversold & spike) | (trend_flip & uptrend & rsi_rising & spike)

        result = pd.Series(signals.HOLD, index=data.index)
        result[buy] = signals.BUY
        result[sell] = signals.SELL
        result[close_buy] = signals.CLOSE_BUY
        result[close_sell] = signals.CLOSE_SELL
        return result.to_dict()
