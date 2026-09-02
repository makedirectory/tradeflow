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
from tradeflow.scanners.symbol_scanner import BUILTIN_SCANNERS as _BUILTIN_SCANNERS
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

#: Built-in example scanners. Taken from the scanner package's own literal rather
#: than from ``SymbolScanner.SCANNERS``, which discovery overwrites with the merged
#: set - re-deriving from that on a module reload quietly promoted installed
#: contributions into the reserved built-in names.
BUILTIN_SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(_BUILTIN_SCANNERS)

#: Trading strategies exposed everywhere (CLI, MCP, agent).
#:
#: Seeded with the built-ins rather than left empty, so these are never *less* than
#: the public engine at any instant. Discovery imports arbitrary third-party modules,
#: and a pack that reads the registry while importing used to observe the empty dict
#: this was populated from - raising, and being dropped as a broken contribution.
STRATEGIES: Dict[str, Type[Strategy]] = dict(BUILTIN_STRATEGIES)

#: Universe scanners exposed everywhere (CLI, MCP, agent).
SCANNERS: Dict[str, Type[ScannerStrategy]] = dict(BUILTIN_SCANNERS)


def refresh_registries() -> None:
    """Reload built-ins plus installed strategy/scanner entry points.

    Import-time discovery is enough for normal CLI/MCP use, but tests and notebooks
    can call this after changing the environment. Built-in names are reserved:
    an installed private package cannot silently replace the public examples.

    Both registries are built in full before either is swapped in. Clearing first
    published an empty registry for the whole of discovery - which is exactly when
    third-party code is being imported and may read it - and left the registry wiped
    rather than merely stale if anything raised on the way.
    """
    strategies = _merged_registry(BUILTIN_STRATEGIES, STRATEGY_ENTRY_POINT_GROUP, Strategy)
    scanners = _merged_registry(BUILTIN_SCANNERS, SCANNER_ENTRY_POINT_GROUP, ScannerStrategy)
    STRATEGIES.clear()
    STRATEGIES.update(strategies)
    SCANNERS.clear()
    SCANNERS.update(scanners)
    # SymbolScanner is still used directly in a few paths; keep its legacy class
    # attribute in sync with the single service registry.
    SymbolScanner.SCANNERS = dict(SCANNERS)


def list_strategies() -> List[Dict[str, Any]]:
    """Names, one-line descriptions, and whether each is a shipped example.

    ``example`` is carried rather than implied. The built-ins exist to demonstrate the
    interface, not to be traded; without a label the reasonable read of a registry
    holding three strategies is that the platform *is* those three, which is the
    opposite of the point.
    """
    return [
        {
            "name": name,
            "description": _first_line(cls.__doc__),
            "timeframe": getattr(cls, "TIMEFRAME", ""),
            "example": name in BUILTIN_STRATEGIES,
        }
        for name, cls in STRATEGIES.items()
    ]


def list_scanners() -> List[Dict[str, Any]]:
    """Names, one-line descriptions, and whether each is a shipped example."""
    return [
        {
            "name": name,
            "description": _first_line(cls.__doc__),
            "example": name in BUILTIN_SCANNERS,
        }
        for name, cls in SCANNERS.items()
    ]


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
        # Loading is all-or-nothing for one entry point: if the module will not
        # import, or hands back something that is not a set of contributions, there
        # is nothing of it to keep.
        try:
            contributions = _coerce_contributions(entry_point.name, entry_point.load())
        except Exception as exc:  # noqa: BLE001 - discovery should degrade, not brick the CLI
            logger.warning("Ignoring %s entry point %r: %s", group, entry_point.name, exc)
            continue

        # Validation is not. One unusable class in a pack of five used to abort the
        # loop, keeping whatever had already been added and silently dropping the
        # siblings it had not reached yet - so which of a pack's contributions
        # survived came down to dict ordering, and the warning named neither the
        # offending class nor the ones lost behind it.
        for name, cls in contributions.items():
            try:
                _validate_extension(name, cls, base_cls, group)
            except Exception as exc:  # noqa: BLE001 - one bad class, not one bad pack
                logger.warning(
                    "Ignoring %s contribution %r from entry point %r: %s",
                    group,
                    name,
                    entry_point.name,
                    exc,
                )
                continue
            if name in found:
                logger.warning(
                    "Ignoring duplicate %s contribution %r from entry point %r",
                    group,
                    name,
                    entry_point.name,
                )
                continue
            found[name] = cls
    return found


def _select_entry_points(group: str) -> List[Any]:
    """Installed entry points in ``group``, or none if they cannot be enumerated.

    Enumeration was the one step of discovery outside a ``try``, and it is the step
    that reads installed distribution metadata - which can be malformed by a package
    this project has never heard of. Because discovery runs at import, that took down
    every command importing this module, which is every command: precisely the
    "a broken private distribution must not make the open engine unusable" case the
    rest of this file is written around.
    """
    try:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            return list(entry_points.select(group=group))
        return list(entry_points.get(group, ()))
    except Exception as exc:  # noqa: BLE001 - built-ins only beats no engine at all
        logger.warning("Cannot enumerate %s entry points (%s); built-in names only", group, exc)
        return []


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


# Discovery at import, guarded at import. A degraded engine that still runs the
# built-ins is worth more than an ImportError on every command, and the traceback is
# logged rather than swallowed so the cause stays recoverable.
try:
    refresh_registries()
except Exception:  # noqa: BLE001
    logger.warning("Extension discovery failed; built-in strategies and scanners only", exc_info=True)
