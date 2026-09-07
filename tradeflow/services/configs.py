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
    position_limits = _reconcile_book(position_limits, provenance)
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


def _reconcile_book(
    position_limits: Optional[Dict[str, Any]], provenance: Optional[Any]
) -> Optional[Dict[str, Any]]:
    """The book to write, checked against the one the provenance says was validated.

    A parameter is not a guarantee. Campaign material carries the validated book in its
    recipe, and a caller could still pass ``None`` beside it - writing a config that
    records an eight-position validation and runs at the class default of one, which is
    exactly the contradiction this parameter was added to end. So when the provenance
    knows the book:

    * omitted means *take it from the provenance*, because the caller supplied the
      evidence and leaving the runnable half blank is never what they meant;
    * disagreeing is refused outright, because one of the two numbers is wrong and
      nothing here can tell which - and a file that argues with itself is worse than
      no file.

    Provenance that carries no campaign material, or campaign material with no recorded
    book, changes nothing: absent stays absent rather than being filled from a default
    nobody validated.
    """
    from tradeflow.services.analysis import recorded_book

    campaign = (provenance or {}).get("campaign") if isinstance(provenance, dict) else None
    validated = recorded_book(((campaign or {}).get("recipe") or {}).get("folded_into_identity"))
    if not validated:
        return position_limits
    if position_limits is None:
        return validated
    if dict(position_limits) != dict(validated):
        raise ValueError(
            f"position_limits {position_limits} disagrees with the book this config's own "
            f"provenance records as validated ({validated}). One of them is wrong and this "
            "cannot tell which; pass the validated book, or omit it and it will be used."
        )
    return position_limits


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
