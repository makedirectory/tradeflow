"""An example long/short strategy: fade the move away from a rolling mean.

The pack's second strategy exists because a long-only book cannot demonstrate half the
platform. Leg diagnostics, `max_net_exposure`, the directional-tilt derivation and the
short-borrow side of the cost model only mean anything for a book that trades both
sides — so if you are going to read one example to understand what TradeFlow measures,
read this one alongside the breakout.

The idea is deliberately plain: a name stretched far above its own recent mean is sold,
one stretched far below is bought, and the score is how stretched it is. Whether that is
an edge is not the point and it almost certainly is not.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from tradeflow.strategies.base import Strategy


class PairsReversionStrategy(Strategy):
    """Long/short mean reversion on a z-scored distance from a rolling mean."""

    TIMEFRAME = "1Day"

    #: Long *and* short. The base class reads this to decide whether a negative score
    #: means "go short" or "stay flat" — set it wrong and half the book silently
    #: disappears with nothing to show it ever existed.
    LONG_ONLY = False

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "lookback": {
            "type": "int",
            "min": 10,
            "max": 60,
            "step": 5,
            "default": 20,
            "description": "Bars in the rolling mean and standard deviation",
        },
        "entry_z": {
            "type": "float",
            "min": 1.0,
            "max": 3.0,
            "step": 0.25,
            "default": 2.0,
            "description": "Standard deviations from the mean before taking a side",
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
            "default": 0.06,
            "description": "Stop-loss distance from entry (fraction)",
        },
        "take_profit": {
            "type": "float",
            "min": 0.02,
            "max": 0.12,
            "step": 0.01,
            "default": 0.04,
            "description": "Take-profit distance from entry (fraction)",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault(
            "position_limits",
            {
                "max_positions": 8,
                "max_position_size": 12_500.0,
                "max_total_risk": 0.08,
                # Long + short. A market-neutral book still deploys capital on both
                # sides, so this is roughly twice what a long-only version would want.
                "max_gross_exposure": 1.60,
                # Long - short. Gross cannot see direction: a book inside a 1.6 gross
                # cap can be entirely long. This is the one that keeps it neutral, and
                # `backtest` derives a defensible value from the tilt actually carried.
                "max_net_exposure": 0.30,
            },
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        return self.config["lookback"] + 1

    def initialize(self) -> None:
        if self.config["take_profit"] >= self.config["stop_loss"]:
            # Not a hard rule in general, but for a reversion book that exits into the
            # mean it means the winners are cut further out than the losers, which is
            # almost never what the author intended.
            raise ValueError(
                f"take_profit ({self.config['take_profit']}) is at or beyond stop_loss "
                f"({self.config['stop_loss']}); a reversion book exits into the mean, so "
                f"the target belongs inside the stop"
            )

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        lookback = self.config["lookback"]
        mean = data["close"].rolling(lookback).mean()
        # Population standard deviation, and guarded: a name that has not moved in the
        # whole window has zero dispersion, and dividing by it turns a flat series into
        # infinite conviction.
        deviation = data["close"].rolling(lookback).std(ddof=0).replace(0.0, pd.NA)
        enriched["zscore"] = (data["close"] - mean) / deviation
        return enriched

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Negative of the z-score, so stretched-up reads short and stretched-down long.

        Returned raw rather than clipped at the entry threshold: the engine ranks
        competing entries by magnitude, and flattening everything past the threshold to
        one value would hand that ranking an arbitrary order.
        """
        if data.empty:
            return pd.Series(dtype=float)

        z = pd.to_numeric(data["zscore"], errors="coerce")
        # Inside the band there is no view, and 0.0 is the base class's flat.
        return (-z).where(z.abs() >= self.config["entry_z"], 0.0).fillna(0.0)
