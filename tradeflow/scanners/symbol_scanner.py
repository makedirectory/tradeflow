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
from tradeflow.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

# Extra calendar days fetched per required bar, to absorb weekends/holidays.
_LOOKBACK_DAY_BUFFER = 3


#: Scanner names this package reserves for classes it defines *in this module*.
#: Empty, and deliberately so: the example scanner moved to ``tradeflow.demo`` and
#: arrives by entry point like any other pack's, so the engine discovers its own
#: demonstration the way it discovers yours. Adding a public scanner means shipping
#: it from ``tradeflow.demo`` with an entry point in ``pyproject.toml``, not
#: extending this literal - see ``docs/content/engineering/extending.md``.
#:
#: Kept as a distinct name from :attr:`SymbolScanner.SCANNERS`, which discovery
#: overwrites with the reserved set *plus* whatever installed packages contribute:
#: deriving the reserved set from that live attribute meant a module reload absorbed
#: third-party scanners into it, and the names an extension may not override then
#: silently included names that came from an extension.
BUILTIN_SCANNERS: Dict[str, Type[ScannerStrategy]] = {}


def resolve_scan_clock(as_of: Optional[datetime] = None) -> datetime:
    """The exchange-zone instant a scan is resolved at, from an optional request.

    One rule, so the clock a scan *runs* at and the clock it *reports* cannot differ.
    The payload used to echo the caller's own argument, so a naive ``2024-06-01`` was
    reported as a bare date while the scan actually ran at ``2024-06-01`` New York,
    and an omitted ``as_of`` was reported as nothing at all rather than as the
    wall-clock now it resolved to. A selection clock reported differently from the one
    applied is worse than no clock, because it reads as provenance.
    """
    end = as_of or datetime.now(NEW_YORK)
    if end.tzinfo is None:
        return NEW_YORK.localize(end)
    return end.astimezone(NEW_YORK)


class SymbolScanner:
    """Filters a candidate universe down to scanner-signaled symbols."""

    #: Scanners resolvable by name, kept in step with
    #: :data:`tradeflow.services.registry.SCANNERS` by ``refresh_registries()``.
    #:
    #: Empty until discovery has run, which is why every read goes through
    #: :meth:`_registry` rather than touching this directly. Nothing is defined in
    #: this module any more, so the seed that used to make a bare
    #: ``import symbol_scanner`` usable is gone: without the lazy resolve, importing
    #: this module alone gave ``available() == []`` and every name raised.
    SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(BUILTIN_SCANNERS)

    @classmethod
    def _registry(cls) -> Dict[str, Type[ScannerStrategy]]:
        """The resolvable scanners, running discovery first if nothing has yet.

        Delegates to the service registry rather than keeping a second answer. The
        import is deferred because ``registry`` imports *this* module; during that
        import ``registry.SCANNERS`` is already seeded with the reserved names, so a
        pack that scans while being discovered sees those rather than nothing.

        This is also the fallback the guard around import-time discovery exists for:
        if ``refresh_registries()`` raises before its final line, ``registry.SCANNERS``
        still holds its seed and every scan path keeps working, where a stale empty
        class attribute would have failed every name the CLI still advertised.
        """
        if not cls.SCANNERS:
            from tradeflow.services import registry

            cls.SCANNERS = dict(registry.SCANNERS)
        return cls.SCANNERS

    def __init__(
        self,
        data_client: MarketDataClient,
        # Required. There is no built-in to fall back to - every scanner, the demo
        # one included, arrives by entry point, so a default here could only name
        # something that may not be installed and raise one frame further in.
        strategy_name: str,
        config: Optional[dict] = None,
    ):
        registry = self._registry()
        if strategy_name not in registry:
            raise ValueError(f"Unknown scanner '{strategy_name}'. Available: {self.available()}")
        scanner_cls = registry[strategy_name]
        defaults = {p: spec["default"] for p, spec in scanner_cls.PARAM_RANGES.items()}
        defaults.update(config or {})

        self.data_client = data_client
        self.strategy = scanner_cls(defaults)
        self.strategy.initialize()
        self.timeframe: str = getattr(scanner_cls, "TIMEFRAME", "1Day")

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry())

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
        end = resolve_scan_clock(as_of)
        return end - timedelta(days=days), end
