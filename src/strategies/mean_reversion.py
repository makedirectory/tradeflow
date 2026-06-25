"""RSI mean reversion - the trend follower's temperamental opposite.

Long-only and contrarian: buy when RSI dips **into** oversold territory (betting
the dip snaps back), and exit when RSI climbs **into** overbought territory (or a
stop / take-profit fires first). Both entries and exits are edge-triggered - they
fire on the bar that *crosses* a threshold, not on every bar that sits beyond it -
so a single dip produces one entry, not a flurry.

A useful foil to :mod:`ma_crossover`: the two disagree by construction, which is
exactly what makes side-by-side walk-forward results worth looking at.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.indicators import indicators
from src.strategies import signals
from src.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Long-only RSI mean reversion: buy oversold dips, exit on the rebound."""

    #: Bars this strategy is designed to trade.
    TIMEFRAME = "1Day"

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "rsi_period": {
            "type": "int",
            "min": 5,
            "max": 30,
            "step": 1,
            "default": 14,
            "description": "RSI lookback period",
        },
        "oversold": {
            "type": "int",
            "min": 15,
            "max": 40,
            "step": 1,
            "default": 30,
            "description": "RSI level a dip must cross below to trigger a buy",
        },
        "overbought": {
            "type": "int",
            "min": 60,
            "max": 85,
            "step": 1,
            "default": 70,
            "description": "RSI level a rebound must cross above to exit",
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
            "min": 0.02,
            "max": 0.10,
            "step": 0.01,
            "default": 0.05,
            "description": "Stop-loss distance from entry (fraction)",
        },
        "take_profit": {
            "type": "float",
            "min": 0.02,
            "max": 0.15,
            "step": 0.01,
            "default": 0.08,
            "description": "Take-profit distance from entry (fraction)",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault(
            "position_limits",
            {"max_positions": 1, "max_position_size": 100_000.0, "max_total_risk": 0.05},
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        return self.config["rsi_period"] + 1

    def initialize(self) -> None:
        if self.config["oversold"] >= self.config["overbought"]:
            raise ValueError(
                f"oversold ({self.config['oversold']}) must be < overbought ({self.config['overbought']})"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        enriched["rsi"] = indicators.calculate_rsi(data["close"], self.config["rsi_period"])
        return enriched

    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        if data.empty:
            return {}

        rsi = data["rsi"]
        prev_rsi = rsi.shift(1)
        # Edge-triggered: the bar that crosses the threshold, not every bar past it.
        crossed_into_oversold = (rsi < self.config["oversold"]) & (prev_rsi >= self.config["oversold"])
        crossed_into_overbought = (rsi > self.config["overbought"]) & (prev_rsi <= self.config["overbought"])

        result = pd.Series(signals.HOLD, index=data.index)
        result[crossed_into_oversold] = signals.BUY
        result[crossed_into_overbought] = signals.CLOSE_BUY
        return result.to_dict()
