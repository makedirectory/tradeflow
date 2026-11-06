"""The optional extras: what each provides, whether it is installed, how to add one.

One map and one recovery text. Every "this needs the X extra" message in the package
comes from here, because the right command depends on how the software was installed
and the wrong one is a dead end: ``uv sync`` needs a checkout an installed reader does
not have, and ``uv tool install --force 'pkg[X]'`` *replaces* the tool environment, so
on its own it removes every other extra and every ``--with`` package the reader had
added.

Depends only on :mod:`tradeflow.settings` and the standard library, so any layer can
call it - the optimizer and allocator report a missing dependency from below the
services layer.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Extra:
    """One published extra."""

    name: str  #: the extra's name in ``pyproject.toml``
    modules: Tuple[str, ...]  #: every import it provides - all must be present
    unlocks: str  #: one line on what the reader gets
    required_for: Tuple[str, ...] = ()  #: tasks that cannot run without it; () = optional everywhere


EXTRAS: Dict[str, Extra] = {
    e.name: e
    for e in (
        Extra(
            "store",
            ("pyarrow", "polars", "duckdb"),
            "the bar cache (`--cache`/`--offline`) and the Parquet bar store",
        ),
        Extra(
            "viz", ("matplotlib",), "charts in `--chart` and in HTML reports (reports fall back to tables)"
        ),
        Extra("mcp", ("mcp",), "the MCP server for AI clients", ("mcp",)),
        Extra("optimize", ("sklearn",), "`optimize --method bayesian`"),
        Extra("portfolio", ("ortools",), "`allocate` (the constraint solver)"),
        Extra("ai", ("anthropic",), "`research --provider anthropic`"),
        Extra("openai", ("openai",), "`research --provider openai`"),
    )
}


def installed(extra: str) -> bool:
    """Whether every module the extra provides can be imported here.

    All of them, not the first: ``store`` is three packages, and having one of the
    three leaves the Parquet store or the lazy compute path failing at first use.
    """
    return all(importlib.util.find_spec(module) is not None for module in EXTRAS[extra].modules)


def installed_extras() -> List[str]:
    """Names of the extras fully satisfied in this environment, sorted."""
    return sorted(name for name in EXTRAS if installed(name))


def _in_uv_tool_env() -> bool:
    """Whether this interpreter lives in a ``uv tool`` environment (a heuristic).

    The installer cannot be read off the running process, so this looks at where the
    environment sits: under ``UV_TOOL_DIR`` when set, otherwise in a ``uv/tools``
    directory (the default on every platform uv supports).
    """
    import os

    prefix = Path(sys.prefix)
    override = os.environ.get("UV_TOOL_DIR")
    if override:
        try:
            prefix.resolve().relative_to(Path(override).resolve())
            return True
        except ValueError:
            pass
    parts = [p.lower() for p in prefix.parts]
    return any(a == "uv" and b == "tools" for a, b in zip(parts, parts[1:]))


def _pack_distributions() -> List[str]:
    """Distributions, other than this one, that register strategies or scanners.

    These are what a reader installed with ``--with`` and what a rebuilt tool
    environment will not have unless they are named again. Read from the running
    environment's entry points, so nothing about anyone's packs is written down here.
    """
    from importlib import metadata

    from tradeflow.settings import DISTRIBUTION_NAME, SCANNER_ENTRY_POINT_GROUP, STRATEGY_ENTRY_POINT_GROUP

    names = set()
    try:
        entry_points = metadata.entry_points()
        for group in (STRATEGY_ENTRY_POINT_GROUP, SCANNER_ENTRY_POINT_GROUP):
            if hasattr(entry_points, "select"):
                selected = list(entry_points.select(group=group))
            else:  # pragma: no cover - the dict-shaped API of older Pythons
                selected = list(entry_points.get(group, ()))
            for entry_point in selected:
                dist = getattr(entry_point, "dist", None)
                name = getattr(dist, "name", None) if dist is not None else None
                if name and name.lower().replace("_", "-") != DISTRIBUTION_NAME:
                    names.add(name)
    except Exception:  # noqa: BLE001 - advice about installing must not fail over its own lookup
        return []
    return sorted(names)


def install_hint(*extras: str) -> str:
    """The recovery text for adding ``extras``, phrased for how this copy was reached.

    A checkout gets one ``uv sync`` line. An installed copy gets both ways to add an
    extra, because which installer made the environment cannot be known for certain:
    the ``uv tool`` form carries the extras already installed (it replaces the whole
    environment, so naming only the new one would silently remove the rest) and is
    pinned to the running version, since adding a capability is not an upgrade.
    """
    import tradeflow
    from tradeflow.settings import DISTRIBUTION_NAME, DOCS_URL, running_from_checkout

    wanted = sorted(set(extras))
    union = sorted({*installed_extras(), *wanted})

    if running_from_checkout():
        # `uv sync` also removes whatever is not named, so the union is the command.
        return "    uv sync " + " ".join(f"--extra {name}" for name in union)

    version = tradeflow.__version__
    uv_form = f"uv tool install --force '{DISTRIBUTION_NAME}[{','.join(union)}]=={version}'"
    packs = _pack_distributions()
    packs_note = ", ".join(packs) if packs else "none detected"
    pip_form = f"python -m pip install '{DISTRIBUTION_NAME}[{','.join(wanted)}]=={version}'"

    detected = _in_uv_tool_env()
    uv_label = "With uv tool (detected):" if detected else "With uv tool:"
    pip_label = "In an activated venv:"
    width = max(len(uv_label), len(pip_label)) + 1
    lines = [
        f"    {uv_label:<{width}}{uv_form}",
        f"    {'':<{width}}(re-add any packs you installed with --with: {packs_note})",
        f"    {pip_label:<{width}}{pip_form}",
    ]
    lines.append(f"    Extras guide: {DOCS_URL}/usage/extras")
    return "\n".join(lines)


def missing_extra_message(what: str, extra: str) -> str:
    """``what`` needs ``extra``, and the one way to get it."""
    return f"{what} needs the '{extra}' extra.\n{install_hint(extra)}"
