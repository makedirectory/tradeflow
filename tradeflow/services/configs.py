"""Config persistence service.

Thin wrappers over :mod:`tradeflow.optimization.config_store` that return JSON-able
results. Saving a config writes a file a human chooses to use; it **never**
affects any running process.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from tradeflow.optimization import config_store


def save_config(
    name: str,
    *,
    strategy: str,
    params: Dict[str, Any],
    scanner: Optional[str] = None,
    symbols: Optional[Any] = None,
    capital: Optional[float] = None,
    position_limits: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a candidate config to ``configs/<name>.json``; return the path.

    Does not affect any running process - a human promotes a config to live.

    ``position_limits``, ``symbols`` and ``capital`` are here because without them this
    surface could not write a *runnable* config at all: the file omits the book, and a
    run from it inherits the strategy class's default of one position rather than the
    book that was validated. An agent holding campaign material knows that book -
    ``recorded_book`` reads it out - and had no parameter to put it in.

    Omitted values stay omitted rather than being defaulted, so a config that never knew
    its book is distinguishable from one saved at the class default.
    """
    filename = name if name.endswith(".json") else f"{name}.json"
    prov = config_store.Provenance(**provenance) if isinstance(provenance, dict) else provenance
    path = config_store.save_config(
        filename,
        strategy=strategy,
        params=params,
        scanner=scanner,
        symbols=symbols,
        capital=capital,
        position_limits=position_limits,
        provenance=prov,
    )
    return {"path": str(path), "name": name}


def load_config(name: str) -> Dict[str, Any]:
    """Load a previously saved config by name (with or without ``.json``)."""
    path = _resolve(name)
    return config_store.load_config(path)


def list_configs(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """List saved candidate configs with a compact summary of each."""
    root = Path(directory) if directory else config_store.DEFAULT_CONFIG_DIR
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            doc = config_store.load_config(path)
        except (ValueError, OSError):
            continue
        prov = doc.get("provenance", {})
        out.append(
            {
                "name": path.stem,
                "strategy": doc.get("strategy"),
                "scanner": doc.get("scanner"),
                "objective": prov.get("objective"),
                "oos_metrics": prov.get("oos_metrics", {}),
                "timestamp": prov.get("timestamp"),
            }
        )
    return out


def _resolve(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    filename = name if name.endswith(".json") else f"{name}.json"
    return config_store.DEFAULT_CONFIG_DIR / filename
