"""The example strategy: a moving-average crossover, the "hello world" of trend following.

**This ships so the demo has something to run and so there is a smallest complete
strategy to read. It is not an edge, and it is not where your work goes** — that belongs
in your own package. ``tradeflow init --example-pack ./my-signals`` copies a complete one
you own, with two strategies, a scanner and configs already wired up.

It reaches the engine through the ``tradeflow.strategies`` entry-point group declared in
this project's ``pyproject.toml`` — the same mechanism your pack will use, so the path
is exercised by every install rather than only in tests.

Long-only. Its conviction score is the **normalized EMA gap**
``(fast - slow) / slow``: positive (and rising) when the fast line leads, negative
when it lags. The base class derives the trade signal from the score's sign, so a
golden cross (score crossing above 0) becomes a ``BUY`` and a death cross (crossing
below 0) a ``CLOSE_BUY`` - the discrete behavior is a consequence of the score, not
a second code path. Deliberately simple and only five parameters, so it's an honest
baseline - and a clean example of how little it takes to add a strategy (one file,
a score, the indicators you already have, register the name).
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from tradeflow.indicators import indicators
from tradeflow.strategies.base import Strategy


class DemoTrendStrategy(Strategy):
    """Long-only EMA trend follower: buy the golden cross, exit the death cross."""

    #: Marks this as a shipped demonstration rather than something to trade. Carried on
    #: the class rather than inferred from where it was registered, because how a
    #: strategy was discovered says nothing about what it is for - and after this moved
    #: to an entry point, registry membership stopped being able to tell.
    DEMO = True

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
        enriched["fast_ema"] = indicators.calculate_ema(data["close"], self.config["fast_ema_period"])
        enriched["slow_ema"] = indicators.calculate_ema(data["close"], self.config["slow_ema_period"])
        return enriched

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        # Normalized EMA gap: signed trend strength. Sign crossings are the golden /
        # death crosses; magnitude is how decisively the fast line leads.
        return (data["fast_ema"] - data["slow_ema"]) / data["slow_ema"]
