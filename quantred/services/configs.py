"""Config persistence service.

Thin wrappers over :mod:`quantred.optimization.config_store` that return JSON-able
results. Saving a config writes a file a human chooses to use; it **never**
affects any running process.
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantred.optimization import config_store

#: Device names Windows refuses as file names whatever the extension.
_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def config_filename(name: Any) -> str:
    """``name`` as the file it names inside the configs directory, or raise.

    A name is a plain file name and nothing else: this is reachable from an agent, and
    ``save_config(name="../x")`` wrote outside the configs directory (and a read of an
    absolute path returned any JSON file on the machine). A directory component of any
    kind, an absolute path, ``.`` / ``..`` and an empty stem are all refused, so every
    config lives directly under :data:`~quantred.optimization.config_store.DEFAULT_CONFIG_DIR`.
    """
    text = name if isinstance(name, str) else ""
    stem = text[: -len(".json")] if text.endswith(".json") else text
    plain = (
        bool(stem.strip())
        and stem not in (".", "..")
        and "/" not in text
        and "\\" not in text
        and ":" not in text  # a drive letter, or an alternate data stream on Windows
        and all(ord(ch) >= 32 for ch in text)
        and stem.split(".")[0].upper() not in _RESERVED_WINDOWS_NAMES
        and Path(text).name == text
        and not Path(text).is_absolute()
    )
    if not plain:
        raise ValueError(
            f"Config name {name!r} is not a plain file name. Give a short name such as "
            "'momentum_v2' - no directories, no '..', no absolute path; it is written under "
            "the configs directory."
        )
    return f"{stem}.json"


def _validate_candidate(
    strategy: str,
    params: Dict[str, Any],
    scanner: Optional[str],
    position_limits: Optional[Dict[str, Any]],
    provenance: Optional[Dict[str, Any]],
) -> None:
    """Refuse a config that could not be run, before anything is written.

    A saved config is what a person later promotes, so one that names a strategy that
    does not exist, a parameter the strategy does not declare, a value its range check
    rejects, or a book limit nothing enforces is a file that fails (or worse, runs at
    the defaults) only when someone relies on it.
    """
    from quantred.services.registry import resolve_strategy_class, scanner_refusal
    from quantred.strategies.base import build_with_limits, refuse_unknown_limits, refuse_unknown_params

    cls = resolve_strategy_class(strategy)
    if scanner is not None and (refusal := scanner_refusal(scanner)):
        raise ValueError(refusal)
    refuse_unknown_params(cls, params)
    refuse_unknown_limits(position_limits)
    # A config states only what it overrides; the run fills in the declared defaults, so
    # the candidate is built the same way to check that its overrides construct.
    candidate = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items()}
    candidate.update({k: v for k, v in (params or {}).items() if not k.startswith("_")})
    try:
        build_with_limits(cls, candidate, position_limits)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"These params do not construct {strategy!r}: {exc}") from exc
    if isinstance(provenance, dict):
        accepted = [f.name for f in dataclasses.fields(config_store.Provenance)]
        unknown = [key for key in provenance if key not in accepted]
        if unknown:
            raise ValueError(
                f"provenance has no field {', '.join(map(repr, unknown))}. Accepted fields: "
                f"{', '.join(accepted)}. Put a trial id in `notes` or in `campaign` "
                "(the get_campaign_material block)."
            )


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
    overwrite: bool = False,
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
    filename = config_filename(name)
    _refuse_replacing(filename, overwrite)
    _validate_candidate(strategy, params, scanner, position_limits, provenance)
    position_limits = _reconcile_book(position_limits, provenance)
    capital = _reconcile_capital(capital, provenance)
    cost = _recorded_cost(provenance)
    prov = config_store.Provenance(**provenance) if isinstance(provenance, dict) else provenance
    path = config_store.save_config(
        filename,
        strategy=strategy,
        params=params,
        scanner=scanner,
        symbols=symbols,
        capital=capital,
        position_limits=position_limits,
        cost=cost,
        provenance=prov,
    )
    return {"path": str(path), "name": name}


def _refuse_replacing(filename: str, overwrite: bool) -> None:
    """Refuse to replace a saved config unless asked to.

    A config is what a person promotes, and an agent proposing a new one under a name
    already in use silently replaced it. Compared without case, because a
    case-insensitive filesystem (the macOS default) maps ``Momentum.json`` onto
    ``momentum.json``.
    """
    root = config_store.DEFAULT_CONFIG_DIR
    if overwrite or not root.exists():
        return
    taken = {path.name.lower(): path.name for path in root.glob("*.json")}
    if filename.lower() in taken:
        raise ValueError(
            f"A config named {taken[filename.lower()]!r} already exists. Choose another name, or "
            "pass overwrite=true to replace it."
        )


def _reconcile_capital(capital: Optional[float], provenance: Optional[Any]) -> Optional[float]:
    """The capital to write, checked against the one the provenance recorded.

    The same rule the book gets, for the same reason: guarding one runnable field and
    not the next leaves a config that can still argue with itself, one column over. An
    omitted capital is taken from the campaign material that recorded it; a disagreeing
    one is refused rather than resolved.

    A campaign whose capital was never recorded changes nothing — absent stays absent,
    and the caller's value (including none) stands.
    """
    recorded = _campaign_fact(provenance, "capital")
    if recorded is None:
        return capital
    if capital is None:
        return recorded
    if float(capital) != float(recorded):
        raise ValueError(
            f"capital {capital} disagrees with the capital this config's own provenance "
            f"records as validated ({recorded}). One of them is wrong and this cannot "
            "tell which; pass the validated capital, or omit it and it will be used."
        )
    return capital


def _campaign_fact(provenance: Optional[Any], fact: str) -> Optional[Any]:
    """One recorded run-context fact out of campaign material, or ``None``.

    ``None`` covers both "no campaign material" and "the material recorded no such
    fact", which are the same thing to a caller deciding whether there is anything to
    reconcile against.
    """
    campaign = (provenance or {}).get("campaign") if isinstance(provenance, dict) else None
    entry = ((campaign or {}).get("recipe") or {}).get("context", {}).get(fact) or {}
    return entry.get("value") if entry.get("recorded") else None


def _recorded_cost(provenance: Optional[Any]) -> Optional[Dict[str, Any]]:
    """The cost model the provenance's campaign material says was validated, or ``None``.

    Taken rather than reconciled: this surface has no cost parameter to disagree with,
    and a config that records an evidence block but no cost trades under the defaults
    - the same file-argues-with-itself defect the book and capital guard against, and
    the one ``trials promote`` shared. Absent stays absent.
    """
    from quantred.services.analysis import recorded_cost

    campaign = (provenance or {}).get("campaign") if isinstance(provenance, dict) else None
    return recorded_cost(((campaign or {}).get("recipe") or {}).get("folded_into_identity"))


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
    from quantred.services.analysis import recorded_book

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
    filename = config_filename(name)
    path = config_store.DEFAULT_CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"No saved config named {name!r}.")
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
