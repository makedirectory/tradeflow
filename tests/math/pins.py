"""Golden pins: stored expected values for end-to-end deterministic scenarios.

A pin is a regression tripwire, not a proof. The reference tests next to it prove each
formula against an independent computation; the pins catch the round-robin failure where
a fix to one number quietly moves another. Every scenario is sanity-checked against the
reference identities *before* its values are written, so a pin never freezes a wrong
number.

Files live in ``tests/math/golden/<scenario>.json`` as a provenance header plus a flat
``values`` map (nested dicts are flattened with dotted keys). A normal run compares every
key with a tight tolerance and fails listing each change as ``old -> new``. Regeneration
is deliberate: ``QUANTRED_REGEN_GOLDEN=1 uv run --extra dev pytest tests/math -k golden``.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pytest

from quantred.engine.backtest import ACCOUNTING_VERSION

GOLDEN_DIR = Path(__file__).parent / "golden"
REGEN_ENV = "QUANTRED_REGEN_GOLDEN"

#: Default comparison: closed-form and short-recursion numbers agree to machine precision
#: across platforms, so anything looser would hide a real change.
DEFAULT_REL = 1e-9
DEFAULT_ABS = 1e-12

CHANGED_MESSAGE = (
    "a pinned number changed: confirm the new value is correct, bump ACCOUNTING_VERSION if "
    "engine output changed, and regenerate deliberately "
    f"({REGEN_ENV}=1 uv run --extra dev pytest tests/math -k golden)"
)


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings/sequences to ``{"a.b.0": scalar}``; numpy scalars become Python."""
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            out.update(flatten(value[key], f"{prefix}{key}."))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}{i}."))
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        out.update(flatten(value.tolist(), prefix))
    else:
        out[prefix[:-1]] = value
    return out


def _encode(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and "nonfinite" in value:
        return float(value["nonfinite"])
    return value


def numeric_values(payload: Dict[str, Any], drop: Iterable[str] = ()) -> Dict[str, Any]:
    """Flatten a scenario's output, dropping keys whose dotted name contains any of ``drop``
    (identifiers and timestamps that are not part of the maths)."""
    flat = flatten(payload)
    return {k: v for k, v in flat.items() if not any(d in k for d in drop)}


def _same(old: Any, new: Any, rel: float, abs_: float) -> bool:
    old, new = _decode(old), _decode(new)
    if isinstance(old, bool) or isinstance(new, bool) or isinstance(old, str) or isinstance(new, str):
        return old == new
    if old is None or new is None:
        return old is new
    if isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new):
        return True
    return bool(new == pytest.approx(old, rel=rel, abs=abs_))


def check(
    name: str,
    scenario: str,
    seed: Any,
    values: Dict[str, Any],
    *,
    rel: float = DEFAULT_REL,
    abs_: float = DEFAULT_ABS,
    loose: Optional[Dict[str, tuple]] = None,
) -> None:
    """Compare ``values`` with ``golden/<name>.json``, or rewrite it when regenerating.

    ``loose`` maps a key prefix to ``(rel, abs)`` for numbers that are legitimately noisier
    (an iterative solver's stopping tolerance); every use must be justified where it is
    declared.
    """
    path = GOLDEN_DIR / f"{name}.json"
    encoded = {k: _encode(v) for k, v in values.items()}
    if os.environ.get(REGEN_ENV) == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        header = {
            "scenario": scenario,
            "seed": seed,
            "accounting_version": ACCOUNTING_VERSION,
            "generated_by": f"{REGEN_ENV}=1 pytest tests/math -k golden (tests/math/test_golden_pins.py)",
        }
        path.write_text(
            json.dumps({"provenance": header, "values": encoded}, indent=1, sort_keys=True) + "\n"
        )
        return

    assert path.exists(), f"no golden file {path.name}; generate it with {REGEN_ENV}=1"
    stored = json.loads(path.read_text())
    assert stored["provenance"]["accounting_version"] == ACCOUNTING_VERSION, (
        f"{path.name} was generated under ACCOUNTING_VERSION {stored['provenance']['accounting_version']} "
        f"but the code is at {ACCOUNTING_VERSION}: " + CHANGED_MESSAGE
    )
    old_values = stored["values"]
    changes = []
    for key in sorted(set(old_values) | set(encoded)):
        if key not in encoded:
            changes.append(f"  {key}: {old_values[key]!r} -> (missing)")
            continue
        if key not in old_values:
            changes.append(f"  {key}: (new) -> {encoded[key]!r}")
            continue
        tol_rel, tol_abs = rel, abs_
        for prefix, (r, a) in (loose or {}).items():
            if key.startswith(prefix):
                tol_rel, tol_abs = r, a
        if not _same(old_values[key], encoded[key], tol_rel, tol_abs):
            changes.append(f"  {key}: {old_values[key]!r} -> {encoded[key]!r}")
    if changes:
        pytest.fail(
            f"{name}: {len(changes)} pinned value(s) changed\n" + "\n".join(changes) + "\n" + CHANGED_MESSAGE
        )
