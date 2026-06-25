"""Moving-average crossover - the "hello world" of trend following.

Long-only: enter when the fast EMA crosses **above** the slow EMA (a golden
cross) and exit when it crosses back **below** (a death cross), with a protective
stop and take-profit in between. Deliberately simple and only five parameters, so
it's an honest baseline - and a clean second example of how little it takes to add
a strategy (one file, the indicators you already have, register the name).
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.indicators import indicators
from src.strategies import signals
from src.strategies.base import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """Long-only EMA trend follower: buy the golden cross, exit the death cross."""

    #: Bars this strategy is designed to trade.
    TIMEFRAME = "1Day"

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "fast_ema_period": {
            "type": "int",
            "min": 5,
            "max": 20,
            "step": 1,
            "default": 10,
            "description": "Fast EMA period (the responsive line)",
        },
        "slow_ema_period": {
            "type": "int",
            "min": 21,
            "max": 60,
            "step": 1,
            "default": 30,
            "description": "Slow EMA period (the trend baseline)",
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
            "max": 0.08,
            "step": 0.01,
            "default": 0.03,
            "description": "Stop-loss distance from entry (fraction)",
        },
        "take_profit": {
            "type": "float",
            "min": 0.02,
            "max": 0.15,
            "step": 0.01,
            "default": 0.06,
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
        return self.config["slow_ema_period"] + 1

    def initialize(self) -> None:
        if self.config["fast_ema_period"] >= self.config["slow_ema_period"]:
            raise ValueError(
                f"fast_ema_period ({self.config['fast_ema_period']}) must be < "
                f"slow_ema_period ({self.config['slow_ema_period']})"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        fast = indicators.calculate_ema(data["close"], self.config["fast_ema_period"])
        slow = indicators.calculate_ema(data["close"], self.config["slow_ema_period"])
        enriched["fast_ema"] = fast
        enriched["slow_ema"] = slow
        # +1 when the fast line is above the slow line (uptrend), -1 otherwise.
        enriched["trend"] = (fast > slow).astype(int) * 2 - 1
        return enriched

    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        if data.empty:
            return {}

        trend = data["trend"]
        prev_trend = trend.shift(1)
        golden_cross = (trend == 1) & (prev_trend == -1)
        death_cross = (trend == -1) & (prev_trend == 1)

        result = pd.Series(signals.HOLD, index=data.index)
        result[golden_cross] = signals.BUY
        result[death_cross] = signals.CLOSE_BUY
        return result.to_dict()
