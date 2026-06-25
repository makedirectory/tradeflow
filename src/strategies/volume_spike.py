"""Volume-confirmed trend strategy.

Long **and** short. Its conviction score leans in the direction of the short/long
EMA trend, scaled by how much the current bar's volume exceeds its moving-average
baseline:

    score = (short_ema - long_ema) / long_ema  ·  volume / volume_ma

The sign is the trend; the magnitude is the trend's strength amplified by volume
confirmation - a spike on a real move counts for more than a drift on quiet tape,
which is exactly the conviction a cross-sectional alpha wants. The base class
derives ``BUY``/``SELL``/``CLOSE`` from the score's sign, so there's one source of
truth and no separate signal code. Pure pandas/numpy indicators (no TA-Lib).
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.indicators import indicators
from src.strategies.base import Strategy


class VolumeSpikeStrategy(Strategy):
    """Volume-confirmed EMA trend follower, long and short."""

    #: Bars this strategy is designed to trade.
    TIMEFRAME = "5Min"

    #: Trades both directions: a negative score is a genuine short, not just flat.
    LONG_ONLY = False

    # Each entry carries min/max/step so the optimizer can search the grid.
    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "short_ema_period": {
            "type": "int",
            "min": 5,
            "max": 15,
            "step": 1,
            "default": 9,
            "description": "Short EMA period (trend)",
        },
        "long_ema_period": {
            "type": "int",
            "min": 16,
            "max": 30,
            "step": 1,
            "default": 21,
            "description": "Long EMA period (trend)",
        },
        "volume_ma_period": {
            "type": "int",
            "min": 5,
            "max": 30,
            "step": 1,
            "default": 14,
            "description": "Window for the volume moving-average baseline",
        },
        "risk_per_trade": {
            "type": "float",
            "min": 0.01,
            "max": 0.05,
            "step": 0.01,
            "default": 0.02,
            "description": "Capital fraction risked per trade",
        },
        "stop_loss": {
            "type": "float",
            "min": 0.01,
            "max": 0.05,
            "step": 0.01,
            "default": 0.01,
            "description": "Stop-loss distance from entry (fraction)",
        },
        "take_profit": {
            "type": "float",
            "min": 0.02,
            "max": 0.10,
            "step": 0.02,
            "default": 0.02,
            "description": "Take-profit distance from entry (fraction)",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault(
            "position_limits",
            {"max_positions": 1, "max_position_size": 100.0, "max_total_risk": 0.01},
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        return max(self.config["long_ema_period"] + 1, self.config["volume_ma_period"] + 1)

    def initialize(self) -> None:
        if self.config["short_ema_period"] >= self.config["long_ema_period"]:
            raise ValueError(
                f"short_ema_period ({self.config['short_ema_period']}) must be < "
                f"long_ema_period ({self.config['long_ema_period']})"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        enriched["short_ema"] = indicators.calculate_ema(data["close"], self.config["short_ema_period"])
        enriched["long_ema"] = indicators.calculate_ema(data["close"], self.config["long_ema_period"])
        enriched["volume_ma"] = data["volume"].rolling(window=self.config["volume_ma_period"]).mean()
        return enriched

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        # Signed trend strength, amplified by relative volume (a spike confirms the
        # move). volume_ma == 0 would be a dead symbol; treat its confirmation as 1.
        trend_strength = (data["short_ema"] - data["long_ema"]) / data["long_ema"]
        volume_confirmation = (data["volume"] / data["volume_ma"]).replace([float("inf")], 1.0).fillna(1.0)
        return trend_strength * volume_confirmation
