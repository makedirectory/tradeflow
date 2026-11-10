"""The optional extras: what each provides, whether it is installed, how to add one.

One map and one recovery text. Every "this needs the X extra" message in the package
comes from here, because the right command depends on how the software was installed
and the wrong one is a dead end: ``uv sync`` needs a checkout an installed reader does
not have, and ``uv tool install --force 'pkg[X]'`` *replaces* the tool environment, so
on its own it removes every other extra and every ``--with`` package the reader had
added.

Depends only on :mod:`quantred.settings` and the standard library, so any layer can
call it - the optimizer and allocator report a missing dependency from below the
services layer.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

    from quantred.settings import (
        SCANNER_ENTRY_POINT_GROUP,
        STRATEGY_ENTRY_POINT_GROUP,
        entry_point_groups,
        is_own_distribution,
    )

    names = set()
    try:
        entry_points = metadata.entry_points()
        for group in (
            *entry_point_groups(STRATEGY_ENTRY_POINT_GROUP),
            *entry_point_groups(SCANNER_ENTRY_POINT_GROUP),
        ):
            if hasattr(entry_points, "select"):
                selected = list(entry_points.select(group=group))
            else:  # pragma: no cover - the dict-shaped API of older Pythons
                selected = list(entry_points.get(group, ()))
            for entry_point in selected:
                dist = getattr(entry_point, "dist", None)
                name = getattr(dist, "name", None) if dist is not None else None
                if name and not is_own_distribution(name):
                    names.add(name)
    except Exception:  # noqa: BLE001 - advice about installing must not fail over its own lookup
        return []
    return sorted(names)


def _tool_receipt() -> Optional[Dict[str, Any]]:
    """The ``uv tool`` receipt of this environment, or ``None``.

    ``uv tool install`` writes ``uv-receipt.toml`` at the root of the tool environment:
    exactly the requirements, extras and ``--with`` entries it was built from. That is
    what a rebuild has to restate, so reading it beats guessing from the packages that
    happen to be importable. Absent for any other installer, and unreadable on a Python
    without :mod:`tomllib` (3.10) - in both cases the caller falls back to advice.
    """
    receipt = Path(sys.prefix) / "uv-receipt.toml"
    try:
        import tomllib

        data = tomllib.loads(receipt.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return None
    return data if isinstance(data.get("tool"), dict) else None


def _quote(text: str) -> str:
    return shlex.quote(text)


def _with_flags(requirements: List[Dict[str, Any]]) -> List[str]:
    """``--with`` arguments that restate every requirement but this distribution."""
    from quantred.settings import is_own_distribution

    flags: List[str] = []
    for req in requirements:
        name = str(req.get("name", ""))
        if not name or is_own_distribution(name):
            continue
        extras_part = f"[{','.join(req['extras'])}]" if req.get("extras") else ""
        if req.get("editable"):
            flags.append(f"--with-editable {_quote(str(req['editable']))}")
        elif req.get("directory") or req.get("path"):
            flags.append(f"--with {_quote(str(req.get('directory') or req['path']))}")
        elif req.get("git"):
            flags.append(f"--with {_quote(f'{name}{extras_part} @ git+' + str(req['git']))}")
        elif req.get("url"):
            flags.append(f"--with {_quote(f'{name}{extras_part} @ ' + str(req['url']))}")
        else:
            flags.append(f"--with {_quote(name + extras_part + str(req.get('specifier', '')))}")
    return flags


#: What the printed ``uv tool install`` command calls the exclusions file it asks for.
EXCLUDES_FILE = "excludes.txt"


def _uv_tool_command(wanted: List[str], extra_flags: Tuple[str, ...] = ()):
    """``(command, receipt, note)`` that rebuilds this tool environment with ``wanted``.

    With a receipt the command restates every extra and ``--with`` package the tool was
    built from, including any ``--excludes``. Without one it cannot, and ``note`` says
    what to add by hand.
    """
    import quantred
    from quantred.settings import (
        DISTRIBUTION_NAME,
        LEGACY_DISTRIBUTION_NAMES,
        invoked_legacy_name,
        is_own_distribution,
    )

    receipt = _tool_receipt()
    requirements = list(((receipt or {}).get("tool") or {}).get("requirements") or [])
    recorded: set = set()
    installed_as = None
    for req in requirements:
        name = str(req.get("name", ""))
        if is_own_distribution(name):
            recorded.update(req.get("extras") or ())
            installed_as = installed_as or name.lower().replace("_", "-")
    if installed_as is None and invoked_legacy_name():
        installed_as = LEGACY_DISTRIBUTION_NAMES[0]
    union = sorted({*installed_extras(), *recorded, *wanted})
    spec = f"[{','.join(union)}]"
    if installed_as in LEGACY_DISTRIBUTION_NAMES:
        # The tool that exists is the one installed under the old name; installing
        # `quantred` would build a second tool and leave this one without the extra.
        # That release forwards every extra to quantred and requires a current one, so
        # its own version is not the running engine's and is not pinned to it.
        command = f"uv tool install --force '{installed_as}{spec}>=2.3.1'"
    else:
        command = f"uv tool install --force '{DISTRIBUTION_NAME}{spec}=={quantred.__version__}'"
    note = None
    if receipt is not None:
        excludes = [str(name) for name in ((receipt.get("tool") or {}).get("excludes") or [])]
        flags = [*_with_flags(requirements), *extra_flags]
        prefix = ""
        if excludes:
            # Restated for the same reason as the extras: the tool environment is rebuilt
            # and would otherwise lose a choice the reader made. It is their choice, not
            # ours, so it is carried and then described. uv takes exclusions from a file,
            # not inline, so the command creates the file first - pasted from anywhere.
            flags.append(f"--excludes {EXCLUDES_FILE}")
            names = " ".join(_quote(name) for name in excludes)
            prefix = f"printf '%s\\n' {names} > {EXCLUDES_FILE} && "
            if any(name.lower().replace("_", "-") in LEGACY_DISTRIBUTION_NAMES for name in excludes):
                note = (
                    "(this keeps your exclusion of the old tradeflow-engine; a pack that still imports "
                    "`tradeflow` needs it installed, so drop that exclusion for such a pack)"
                )
        command = prefix + " ".join([command, *flags])
    else:
        packs = _pack_distributions()
        # Silent when nothing was detected outside a uv tool environment: "re-add your
        # --with packs" is advice about a mechanism that venv does not have.
        if packs:
            note = f"(re-add any packs you installed with --with: {', '.join(packs)})"
        elif _in_uv_tool_env():
            note = "(re-add any packs you installed with --with: none detected)"
    return command, receipt, note


#: Said under a pack install command. The old distribution's final release is a thin
#: alias, so keeping it is harmless, and a pack nobody can edit still imports `tradeflow`.
_OLD_ENGINE_NOTE = (
    "Keep tradeflow-engine>=2.3.1 installed: a pack that still imports `tradeflow` works through "
    "the alias it provides, and without it that pack is dropped. If the pack pins an older "
    "tradeflow-engine, override the pin rather than excluding it (a pin below 2.3.1 would install "
    "the whole previous engine, whose classes this one rejects). Excluding it, or {skip}, is only "
    "for a pack that never imports `tradeflow`; {skip} also skips the pack's other dependencies."
)


def _old_distribution_evidence() -> bool:
    """Whether this reader plausibly has the previous distribution.

    Either its metadata is installed here, or the process was started under the old
    command name (an install upgraded in place exposes only that). Without either, the
    note about keeping it installed is advice about a package the reader never had.
    """
    from importlib import metadata

    from quantred.settings import LEGACY_DISTRIBUTION_NAMES, invoked_legacy_name

    if invoked_legacy_name():
        return True
    for name in LEGACY_DISTRIBUTION_NAMES:
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        return True
    return False


def pack_install_hint(path: str) -> str:
    """How to install a pack from a directory, without removing anything already there.

    A checkout installs it into its own environment (additive). A ``uv tool`` copy has
    to rebuild the tool environment, so the command carries everything it already had
    (including any exclusion the reader chose, created first so the line runs anywhere).
    Elsewhere it is pip. Each is followed by what to do about a pack that still depends
    on the old distribution name - only when the reader has that distribution, since a
    new reader scaffolding the example pack has nothing to keep.
    """
    from quantred.settings import running_from_checkout

    quoted = shlex.quote(path)
    if running_from_checkout():
        command = f"uv pip install -e {quoted}"
    else:
        command, receipt, note = _uv_tool_command([], (f"--with-editable {quoted}",))
        if receipt is not None:
            return f"{command}\n{note}" if note else command
        command = f"python -m pip install -e {quoted}"
    if not _old_distribution_evidence():
        return command
    return f"{command}\n({_OLD_ENGINE_NOTE.format(skip='--no-deps')})"


def install_hint(*extras: str) -> str:
    """The recovery text for adding ``extras``, phrased for how this copy was reached.

    A checkout gets one ``uv sync --inexact`` line: plain ``uv sync`` is *exact* and
    uninstalls whatever the lock does not name for the extras given - the ``dev`` tools
    and any pack installed with ``uv pip install -e`` - while ``--inexact`` only adds.

    An installed copy gets both ways to add an extra, because which installer made the
    environment cannot be known for certain. The ``uv tool`` form replaces the whole
    environment, so it restates the extras and ``--with`` packs the tool was built from
    (read from its receipt) and is pinned to the running version, since adding a
    capability is not an upgrade.
    """
    import quantred
    from quantred.settings import DISTRIBUTION_NAME, DOCS_URL, running_from_checkout

    wanted = sorted(set(extras))

    if running_from_checkout():
        return "    uv sync --inexact " + " ".join(f"--extra {name}" for name in wanted)

    version = quantred.__version__
    uv_form, receipt, note = _uv_tool_command(wanted)
    pip_form = f"python -m pip install '{DISTRIBUTION_NAME}[{','.join(wanted)}]=={version}'"

    detected = receipt is not None or _in_uv_tool_env()
    uv_label = "With uv tool (detected):" if detected else "With uv tool:"
    pip_label = "In an activated venv:"
    width = max(len(uv_label), len(pip_label)) + 1
    lines = [f"    {uv_label:<{width}}{uv_form}"]
    if note:
        lines.append(f"    {'':<{width}}{note}")
    lines.append(f"    {pip_label:<{width}}{pip_form}")
    lines.append(f"    Extras guide: {DOCS_URL}/usage/extras")
    return "\n".join(lines)


def missing_extra_message(what: str, extra: str) -> str:
    """``what`` needs ``extra``, and the one way to get it."""
    return f"{what} needs the '{extra}' extra.\n{install_hint(extra)}"


class MissingExtra(RuntimeError):
    """A capability was reached without the extra that provides it.

    Raised where the capability starts rather than where its first import happens to
    sit, and carrying the install text, so every caller ends on the same instruction -
    the CLI prints it, the MCP server returns it as the tool error.
    """


def require_extra(what: str, extra: str) -> None:
    """Raise :class:`MissingExtra` unless ``extra`` is installed."""
    if not installed(extra):
        raise MissingExtra(missing_extra_message(what, extra))
