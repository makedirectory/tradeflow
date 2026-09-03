"""An example private scanner: pick the names liquid enough to be worth trading.

A scanner answers a different question from a strategy. A strategy asks *when*; a
scanner asks *which names belong in the universe at all*. Splitting them is what lets
one strategy be validated against several universes, and what makes the universe a
recorded input of a run rather than an assumption inside it.

Scanners run point-in-time. The engine passes bars up to the scan clock and no further,
so a universe chosen for a backtest in 2021 is chosen from what was knowable in 2021 —
which is the difference between a survivorship-clean study and a flattering one.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from tradeflow.scanners.base import ScannerStrategy


class LiquidityScanner(ScannerStrategy):
    """Flag names whose recent dollar volume clears a floor and is not collapsing."""

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "lookback": {
            "type": "int",
            "min": 10,
            "max": 60,
            "step": 5,
            "default": 20,
            "description": "Bars of dollar volume to average",
        },
        "min_dollar_volume": {
            "type": "float",
            "min": 1_000_000.0,
            "max": 50_000_000.0,
            "step": 1_000_000.0,
            "default": 5_000_000.0,
            "description": "Average daily dollar volume a name must clear",
        },
    }

    def initialize(self) -> None:
        pass

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the columns the signal step reads."""
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        dollar_volume = data["close"] * data["volume"]
        enriched["avg_dollar_volume"] = dollar_volume.rolling(self.config["lookback"]).mean()
        # Recent against longer-run, so a name whose liquidity is draining away is
        # distinguishable from one that never had any. They fail the same threshold and
        # are not the same problem.
        enriched["liquidity_trend"] = (
            dollar_volume.rolling(max(3, self.config["lookback"] // 4)).mean() / enriched["avg_dollar_volume"]
        )
        return enriched

    def generate_signals_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """One row per bar: whether the name qualifies, and how strongly.

        ``signal_strength`` is what ranks the survivors when a universe is capped, so it
        has to be a real number rather than a constant — a scanner that flags everything
        equally hands the cap an arbitrary choice and calls it a selection.
        """
        signals = pd.DataFrame(index=data.index)
        if data.empty:
            signals["signal"] = []
            signals["signal_strength"] = []
            return signals

        liquid = data["avg_dollar_volume"] >= self.config["min_dollar_volume"]
        # Not draining: recent liquidity at least three quarters of the longer average.
        holding_up = data["liquidity_trend"] >= 0.75
        qualifies = (liquid & holding_up).fillna(False)

        signals["signal"] = qualifies.map({True: self.SCANNER_BUY, False: ""})
        # Multiples of the floor, so a name ten times the threshold outranks one that
        # scrapes past it. NaN before the lookback fills is zero strength, never a rank.
        signals["signal_strength"] = (
            (data["avg_dollar_volume"] / self.config["min_dollar_volume"]).where(qualifies, 0.0)
        ).fillna(0.0)
        return signals
