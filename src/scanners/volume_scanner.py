"""Volume scanner.

Flags symbols whose latest bar shows unusually high volume *and* a meaningful
price move - a simple liquidity/attention filter for building a trading universe.
Pure pandas/numpy; no TA-Lib.
"""

from typing import Any, ClassVar, Dict

import pandas as pd

from src.scanners.base import SCANNER_BUY, SCANNER_HOLD, SCANNER_SELL, ScannerStrategy


class VolumeScannerStrategy(ScannerStrategy):
    """Select symbols on a volume spike confirmed by price movement."""

    TIMEFRAME = "1Day"

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "min_volume": {
            "type": "float",
            "min": 50_000,
            "max": 1_000_000,
            "step": 50_000,
            "default": 500_000,
            "description": "Minimum bar volume to consider a symbol liquid",
        },
        "volume_ma_period": {
            "type": "int",
            "min": 5,
            "max": 50,
            "step": 5,
            "default": 10,
            "description": "Lookback for the volume moving-average baseline",
        },
        "volume_threshold": {
            "type": "float",
            "min": 1.0,
            "max": 5.0,
            "step": 0.25,
            "default": 1.75,
            "description": "Volume / volume-MA ratio that flags unusual activity",
        },
        "price_change_threshold": {
            "type": "float",
            "min": 0.5,
            "max": 5.0,
            "step": 0.25,
            "default": 0.5,
            "description": "Minimum |intrabar price change| (%) to confirm",
        },
    }

    def initialize(self) -> None:  # nothing to set up
        pass

    def required_data_points(self) -> int:
        return self.config["volume_ma_period"] + 1

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        enriched = data.copy()
        volume_ma = data["volume"].rolling(window=self.config["volume_ma_period"]).mean()
        enriched["volume_ratio"] = data["volume"] / volume_ma
        enriched["price_change"] = ((data["close"] - data["open"]) / data["open"] * 100).abs()
        return enriched

    def generate_signals_df(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=data.index)
        out["signal"] = SCANNER_HOLD
        out["signal_strength"] = 0.0

        qualifies = (
            (data["volume_ratio"] > self.config["volume_threshold"])
            & (data["price_change"] > self.config["price_change_threshold"])
            & (data["volume"] > self.config["min_volume"])
        )
        bullish = qualifies & (data["close"] > data["open"])
        bearish = qualifies & (data["close"] <= data["open"])

        out.loc[bullish, "signal"] = SCANNER_BUY
        out.loc[bearish, "signal"] = SCANNER_SELL
        out.loc[qualifies, "signal_strength"] = (
            data.loc[qualifies, "volume_ratio"] * data.loc[qualifies, "price_change"]
        )
        return out

    def is_hit(self, returns: float, price_change: float) -> bool:
        return price_change > self.config["price_change_threshold"]
