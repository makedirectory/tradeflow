"""Universe scanner.

Runs a chosen :class:`ScannerStrategy` across a candidate symbol list and returns
those flagged at a specific research clock - the universe the trading
strategy/engine then operates on. Only TA-Lib-free scanners are registered, in
keeping with the no-compiled-deps goal.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type

from tradeflow.data.scan import slice_to_as_of
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.scanners.base import ScannerStrategy
from tradeflow.scanners.volume_scanner import VolumeScannerStrategy
from tradeflow.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

# Extra calendar days fetched per required bar, to absorb weekends/holidays.
_LOOKBACK_DAY_BUFFER = 3


#: The scanners that ship with this package - the literal to extend when adding a
#: public example. Kept separate from :attr:`SymbolScanner.SCANNERS`, which discovery
#: overwrites with built-ins *plus* whatever installed packages contribute: deriving
#: the built-in set from that live attribute meant a module reload absorbed third-party
#: scanners into it, and the reserved names an extension may not override then silently
#: included names that came from an extension.
BUILTIN_SCANNERS: Dict[str, Type[ScannerStrategy]] = {
    "volume": VolumeScannerStrategy,
}


class SymbolScanner:
    """Filters a candidate universe down to scanner-signaled symbols."""

    #: Scanners resolvable by name. Built-ins until discovery runs, then built-ins
    #: plus installed contributions - see :mod:`tradeflow.services.registry`.
    SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(BUILTIN_SCANNERS)

    def __init__(
        self,
        data_client: MarketDataClient,
        strategy_name: str = "volume",
        config: Optional[dict] = None,
    ):
        if strategy_name not in self.SCANNERS:
            raise ValueError(f"Unknown scanner '{strategy_name}'. Available: {self.available()}")
        scanner_cls = self.SCANNERS[strategy_name]
        defaults = {p: spec["default"] for p, spec in scanner_cls.PARAM_RANGES.items()}
        defaults.update(config or {})

        self.data_client = data_client
        self.strategy = scanner_cls(defaults)
        self.strategy.initialize()
        self.timeframe: str = getattr(scanner_cls, "TIMEFRAME", "1Day")

    @classmethod
    def available(cls) -> List[str]:
        return list(cls.SCANNERS)

    def scan(
        self, symbols: List[str], timeframe: Optional[str] = None, as_of: Optional[datetime] = None
    ) -> List[Tuple[str, str]]:
        """Return ``(symbol, scan_signal)`` for every flagged symbol in ``symbols``."""
        timeframe = timeframe or self.timeframe
        start, end = self._scan_window(timeframe, as_of=as_of)
        bars = self.data_client.get_bars(symbols, timeframe, start, end)

        flagged: List[Tuple[str, str]] = []
        for symbol, frame in bars.items():
            if frame.empty:
                continue
            frame = slice_to_as_of(frame, end)
            if frame.empty:
                continue
            processed = self.strategy.process_data(frame)
            signal = self.strategy.latest_signal(self.strategy.generate_signals_df(processed))
            if signal in (self.strategy.SCANNER_BUY, self.strategy.SCANNER_SELL):
                flagged.append((symbol, signal))

        logger.info("Scanner flagged %d/%d symbols", len(flagged), len(symbols))
        return flagged

    def _scan_window(self, timeframe: str, as_of: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        """A lookback window large enough for the scanner's indicators."""
        tf = Timeframe.parse(timeframe)
        required = self.strategy.required_data_points()
        days = max(required * tf.amount * _LOOKBACK_DAY_BUFFER, 30)
        end = as_of or datetime.now(NEW_YORK)
        if end.tzinfo is None:
            end = NEW_YORK.localize(end)
        else:
            end = end.astimezone(NEW_YORK)
        return end - timedelta(days=days), end
