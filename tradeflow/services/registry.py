"""Discovery: the registries of strategies and scanners, and their tunables.

Single source of truth for "what can the engine run?" - imported by the CLI, the
MCP server, and the research agent so they all see the same menu.
"""

import logging
import re
from collections.abc import Mapping
from importlib import metadata
from typing import Any, Dict, List, Type

from tradeflow.scanners.base import ScannerStrategy
from tradeflow.scanners.symbol_scanner import SymbolScanner
from tradeflow.strategies.base import Strategy
from tradeflow.strategies.ma_crossover import MovingAverageCrossoverStrategy
from tradeflow.strategies.mean_reversion import MeanReversionStrategy
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy

logger = logging.getLogger(__name__)

STRATEGY_ENTRY_POINT_GROUP = "tradeflow.strategies"
SCANNER_ENTRY_POINT_GROUP = "tradeflow.scanners"

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

#: Built-in example strategies exposed everywhere (CLI, MCP, agent). Real IP can
#: live in a private package and register through the entry-point groups above.
BUILTIN_STRATEGIES: Dict[str, Type[Strategy]] = {
    "volume_spike": VolumeSpikeStrategy,
    "ma_crossover": MovingAverageCrossoverStrategy,
    "mean_reversion": MeanReversionStrategy,
}

#: Built-in example scanners.
BUILTIN_SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(SymbolScanner.SCANNERS)

#: Trading strategies exposed everywhere (CLI, MCP, agent).
STRATEGIES: Dict[str, Type[Strategy]] = {}

#: Universe scanners exposed everywhere (CLI, MCP, agent).
SCANNERS: Dict[str, Type[ScannerStrategy]] = {}


def refresh_registries() -> None:
    """Reload built-ins plus installed strategy/scanner entry points.

    Import-time discovery is enough for normal CLI/MCP use, but tests and notebooks
    can call this after changing the environment. Built-in names are reserved:
    an installed private package cannot silently replace the public examples.
    """
    STRATEGIES.clear()
    STRATEGIES.update(_merged_registry(BUILTIN_STRATEGIES, STRATEGY_ENTRY_POINT_GROUP, Strategy))
    SCANNERS.clear()
    SCANNERS.update(_merged_registry(BUILTIN_SCANNERS, SCANNER_ENTRY_POINT_GROUP, ScannerStrategy))
    # SymbolScanner is still used directly in a few paths; keep its legacy class
    # attribute in sync with the single service registry.
    SymbolScanner.SCANNERS = dict(SCANNERS)


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


def _merged_registry(builtins: Dict[str, Type], group: str, base_cls: Type) -> Dict[str, Type]:
    registry = dict(builtins)
    for name, cls in _entry_point_classes(group, base_cls).items():
        if name in registry:
            logger.warning("Ignoring %s entry point %r: built-in names cannot be overridden", group, name)
            continue
        registry[name] = cls
    return registry


def _entry_point_classes(group: str, base_cls: Type) -> Dict[str, Type]:
    """Load extension classes from an entry-point group.

    A contribution may be either a class, a mapping of name -> class, or a no-arg
    function returning that mapping. Invalid contributions are ignored with a warning
    so a broken private package cannot make the public engine unusable.
    """
    found: Dict[str, Type] = {}
    for entry_point in _select_entry_points(group):
        try:
            loaded = entry_point.load()
            contributions = _coerce_contributions(entry_point.name, loaded)
            for name, cls in contributions.items():
                _validate_extension(name, cls, base_cls, group)
                if name in found:
                    logger.warning("Ignoring duplicate %s entry point %r", group, name)
                    continue
                found[name] = cls
        except Exception as exc:  # noqa: BLE001 - discovery should degrade, not brick the CLI
            logger.warning("Ignoring %s entry point %r: %s", group, entry_point.name, exc)
    return found


def _select_entry_points(group: str):
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group=group)
    return entry_points.get(group, ())


def _coerce_contributions(default_name: str, loaded: Any) -> Dict[str, Type]:
    if isinstance(loaded, type):
        return {default_name: loaded}
    if isinstance(loaded, Mapping):
        return dict(loaded)
    if callable(loaded):
        contributed = loaded()
        if not isinstance(contributed, Mapping):
            raise ValueError("factory entry point must return a mapping of name -> class")
        return dict(contributed)
    raise ValueError("entry point must load a class, mapping, or mapping factory")


def _validate_extension(name: str, cls: Type, base_cls: Type, group: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid extension name {name!r}")
    if not isinstance(cls, type) or not issubclass(cls, base_cls) or cls is base_cls:
        raise ValueError(f"{name!r} is not a concrete {base_cls.__name__} subclass")
    if getattr(cls, "__abstractmethods__", None):
        raise ValueError(f"{name!r} leaves abstract methods unimplemented")
    if not isinstance(getattr(cls, "PARAM_RANGES", None), dict):
        raise ValueError(f"{name!r} must declare PARAM_RANGES")
    logger.info("Loaded %s extension %r from %s", group, name, cls.__module__)


def _first_line(doc: str) -> str:
    return (doc or "").strip().split("\n", 1)[0]


refresh_registries()
