"""Discovery: the registries of strategies and scanners, and their tunables.

Single source of truth for "what can the engine run?" - imported by the CLI, the
MCP server, and the research agent so they all see the same menu.
"""

from typing import Any, Dict, List, Type

from src.scanners.base import ScannerStrategy
from src.scanners.symbol_scanner import SymbolScanner
from src.strategies.base import Strategy
from src.strategies.ma_crossover import MovingAverageCrossoverStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volume_spike import VolumeSpikeStrategy

#: Trading strategies exposed everywhere (CLI, MCP, agent).
STRATEGIES: Dict[str, Type[Strategy]] = {
    "volume_spike": VolumeSpikeStrategy,
    "ma_crossover": MovingAverageCrossoverStrategy,
    "mean_reversion": MeanReversionStrategy,
}

#: Universe scanners (delegates to the scanner package's own registry).
SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(SymbolScanner.SCANNERS)


def list_strategies() -> List[Dict[str, str]]:
    """Names + one-line descriptions for every registered strategy."""
    return [
        {"name": name, "description": _first_line(cls.__doc__), "timeframe": getattr(cls, "TIMEFRAME", "")}
        for name, cls in STRATEGIES.items()
    ]


def list_scanners() -> List[Dict[str, str]]:
    """Names + one-line descriptions for every registered scanner."""
    return [{"name": name, "description": _first_line(cls.__doc__)} for name, cls in SCANNERS.items()]


def get_param_ranges(kind: str, name: str) -> Dict[str, Any]:
    """The ``PARAM_RANGES`` map for a strategy/scanner - what is tunable and its bounds.

    Args:
        kind: ``"strategy"`` or ``"scanner"``.
        name: A registered name.
    """
    registry = STRATEGIES if kind == "strategy" else SCANNERS if kind == "scanner" else None
    if registry is None:
        raise ValueError(f"kind must be 'strategy' or 'scanner', got {kind!r}")
    if name not in registry:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {list(registry)}")
    return {"kind": kind, "name": name, "param_ranges": registry[name].PARAM_RANGES}


def resolve_strategy_class(name: str) -> Type[Strategy]:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
    return STRATEGIES[name]


def _first_line(doc: str) -> str:
    return (doc or "").strip().split("\n", 1)[0]
