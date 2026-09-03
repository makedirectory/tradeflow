"""An example private strategy: a Donchian breakout with a trend filter.

Written to be *read*, not traded. It is a deliberately ordinary idea — buy a break of
the recent high while the longer trend agrees — because the point is the shape of a
strategy, not the alpha in it.

Four methods and a parameter table is the whole contract. The engine calls these and
nothing else, which is why sizing, fills, exits, book limits, cost accounting, the
walk-forward search, the trial store and the live path all work for your strategy
without knowing anything about it.

**One timing rule to understand before writing your own.** The score you return for bar
``i`` is derived from bar ``i``'s close, and the engine executes the resulting signal at
the open of bar ``i + 1``. You do not shift anything yourself — a strategy that lagged
its own scores would be lagged twice. That is also what the live path does: a closed bar
produces a signal, and an order fills after it. Backtest and deployment therefore agree
about what was knowable when.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from tradeflow.strategies.base import Strategy


class BreakoutStrategy(Strategy):
    """Long-only Donchian breakout, gated by a longer-term trend filter."""

    #: The bars this strategy is designed for. The engine reads it rather than guessing.
    TIMEFRAME = "1Day"

    #: What the optimizer is allowed to search, and between which bounds.
    #:
    #: This table is the only place a parameter's range is declared. Walk-forward reads
    #: it to build a search, `get_param_ranges` exposes it over MCP, and a config
    #: carrying a value outside these bounds is refused rather than clamped — a run that
    #: quietly ran different parameters from the ones it recorded would be worse than
    #: one that failed.
    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "entry_period": {
            "type": "int",
            "min": 20,
            "max": 80,
            "step": 10,
            "default": 40,
            "description": "Lookback for the breakout high",
        },
        "trend_period": {
            "type": "int",
            "min": 50,
            "max": 200,
            "step": 25,
            "default": 100,
            "description": "Lookback for the trend filter",
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
            "min": 0.04,
            "max": 0.20,
            "step": 0.02,
            "default": 0.10,
            "description": "Take-profit distance from entry (fraction)",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        # Declared, not left to the defaults. Book limits are part of what gets
        # validated, so a strategy that does not state them is validated against
        # whatever the base class happens to declare.
        config.setdefault(
            "position_limits",
            {
                "max_positions": 5,
                "max_position_size": 20_000.0,
                "max_total_risk": 0.06,
                "max_gross_exposure": 0.90,
            },
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        """Bars needed before the indicators mean anything.

        The live path warms up with exactly this many and refuses to start if it could
        not get them, so understating it produces confident signals from too little
        history — which is indistinguishable, from inside the loop, from a quiet market.
        """
        return max(self.config["entry_period"], self.config["trend_period"]) + 1

    def initialize(self) -> None:
        """Validate the parameter combination, once, before any bar is processed.

        Ranges are checked by the optimizer; *relationships* between parameters are not,
        and this is where they belong. Failing loudly here beats a run that silently
        produces nothing.
        """
        if self.config["entry_period"] >= self.config["trend_period"]:
            raise ValueError(
                f"entry_period ({self.config['entry_period']}) must be shorter than "
                f"trend_period ({self.config['trend_period']}), or the filter can never "
                f"disagree with the breakout"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the indicator columns `calculate_scores` will read.

        Split from scoring so the engine can compute indicators once per symbol and
        score many times during a parameter search.
        """
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        # Shifted by one: the highest high *before* this bar. Without the shift, today's
        # own high is part of the level today has to break, and nothing can ever break
        # it — the indicator would quietly never fire.
        enriched["breakout_high"] = data["high"].rolling(self.config["entry_period"]).max().shift(1)
        enriched["trend_ma"] = data["close"].rolling(self.config["trend_period"]).mean()
        return enriched

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Conviction per bar. Positive means long, negative means flat or short.

        The base class turns this into BUY / CLOSE_BUY signals at the sign crossings, so
        the discrete behaviour is a consequence of the score rather than a second code
        path that could disagree with it. Magnitude matters too: the engine ranks
        competing entries by it when the book cannot hold them all.
        """
        if data.empty:
            return pd.Series(dtype=float)

        # How far above the breakout level the close sits, as a fraction. Zero when the
        # level has not been taken out; the trend filter zeroes it again when the longer
        # trend disagrees.
        distance = (data["close"] - data["breakout_high"]) / data["breakout_high"]
        broke_out = data["close"] > data["breakout_high"]
        trending = data["close"] > data["trend_ma"]
        return (distance.where(broke_out & trending, -1.0)).fillna(-1.0)
