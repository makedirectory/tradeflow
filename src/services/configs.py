"""Config persistence service.

Thin wrappers over :mod:`src.optimization.config_store` that return JSON-able
results. Saving a config writes a file a human chooses to use; it **never**
affects any running process.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.optimization import config_store


def save_config(
    name: str,
    *,
    strategy: str,
    params: Dict[str, Any],
    scanner: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a candidate config to ``configs/<name>.json``; return the path.

    Does not affect any running process - a human promotes a config to live.
    """
    filename = name if name.endswith(".json") else f"{name}.json"
    prov = config_store.Provenance(**provenance) if isinstance(provenance, dict) else provenance
    path = config_store.save_config(
        filename, strategy=strategy, params=params, scanner=scanner, provenance=prov
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
        out.append({
            "name": path.stem,
            "strategy": doc.get("strategy"),
            "scanner": doc.get("scanner"),
            "objective": prov.get("objective"),
            "oos_metrics": prov.get("oos_metrics", {}),
            "timestamp": prov.get("timestamp"),
        })
    return out


def _resolve(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    filename = name if name.endswith(".json") else f"{name}.json"
    return config_store.DEFAULT_CONFIG_DIR / filename
