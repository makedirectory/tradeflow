"""RSI mean reversion - the trend follower's temperamental opposite.

Long-only and contrarian. Its conviction score is **how oversold the name is**,
``50 - rsi``: positive when RSI is below the midpoint (a dip worth fading), negative
when it's stretched up. The base class derives the trade signal with *asymmetric*
hysteresis (see :meth:`signal_thresholds`): enter long when RSI dips below
``oversold``, hold through the middle, exit only when RSI climbs above
``overbought``. So the discrete behavior - enter the dip, exit the rebound - falls
out of the score, with no separate signal code.

A useful foil to :mod:`ma_crossover`: the two disagree by construction, which is
exactly what makes side-by-side walk-forward results worth looking at.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.indicators import indicators
from src.strategies.base import ScoreThresholds, Strategy

#: RSI midpoint the score is centered on, so score = MIDPOINT - rsi.
_RSI_MIDPOINT = 50.0


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

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        # Oversold-ness: positive when RSI is below the midpoint (a dip to fade).
        return _RSI_MIDPOINT - data["rsi"]

    def signal_thresholds(self) -> ScoreThresholds:
        # Enter when RSI < oversold (score above 50-oversold); exit only when
        # RSI > overbought (score below 50-overbought). The wide middle band is the
        # "hold the position" zone - this is what makes entries and exits asymmetric.
        return ScoreThresholds(
            enter_long=_RSI_MIDPOINT - self.config["oversold"],
            exit_long=_RSI_MIDPOINT - self.config["overbought"],
        )
