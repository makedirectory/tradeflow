---
sidebar_position: 3
title: Migrating to QuantRed
---

# Migrating to QuantRed

The project was called TradeFlow and was distributed as `tradeflow-engine`. It is now
**QuantRed**, distributed as `quantred`, and the main command is `quantred`. This page
is for anyone who already has the old distribution installed. If you are installing for
the first time, you do not need it: see [installation](installation).

Two releases carry the move. QuantRed 2.4.0 is the first release published as `quantred`.
`tradeflow-engine` 2.3.1 is the last release under the old name and is published after
it: a transition package with no engine code of its own. It depends on `quantred`, still
provides the `tradeflow` command (which runs QuantRed's command line), and ships a
`tradeflow` import shim, so `import tradeflow` and `from tradeflow.strategies.base import
Strategy` keep working and give you the same objects as `quantred.*`, with a
`DeprecationWarning`. `python -m tradeflow.cli ...`, which older MCP client configs use,
keeps working too.

`quantred` itself installs no `tradeflow` command. That command exists only while
`tradeflow-engine` 2.3.1 is installed.

The upgrade paths below were run with locally built wheels of `quantred` 2.4.0 and
`tradeflow-engine` 2.3.1 standing in for the package index, starting from the real
`tradeflow-engine` 2.3.0 from PyPI, on Python 3.12. The published packages themselves,
Windows, the checkout swap and the `--excludes` / `--no-deps` recipes were not checked;
those are marked *not verified*.

## Which path

- **A. Upgrade in place and move at your own pace.** Upgrade `tradeflow-engine` the way
  you always have. It pulls in `quantred` and keeps old scripts, cron entries, MCP client
  configs and private packs importing `tradeflow` working, with a deprecation warning.
  Move each of those across when it suits you, then drop the transition package.
- **B. Move fully now.** Install `quantred`, change the names yourself (below), and
  remove `tradeflow-engine`.

The two distributions no longer share any file, so neither path needs the old
distribution removed first for correctness.

## What changed

| | Before | Now |
| --- | --- | --- |
| Display name | TradeFlow | QuantRed |
| Distribution (what you install) | `tradeflow-engine` | `quantred` |
| Main command | `tradeflow` | `quantred` (`tradeflow` only via `tradeflow-engine` 2.3.1) |
| Import package | `tradeflow` | `quantred` (`tradeflow` only via the 2.3.1 shim) |
| Plugin entry-point groups | `tradeflow.strategies`, `tradeflow.scanners` | `quantred.strategies`, `quantred.scanners` (the old groups are still read) |
| Provenance code identity (installed copy) | `tradeflow <version>` | `quantred <version>` |
| MCP server name (what a client sees) | `tradeflow` | `quantred` |

## What did not change

- **Your state.** `~/.tradeflow`, `TRADEFLOW_HOME`, the journal, trial store, bar cache,
  saved configs and your `.env` are untouched, so recorded history carries over.
- **Old entry-point groups still register.** A pack declared against
  `tradeflow.strategies` or `tradeflow.scanners` keeps registering. Its imports are a
  separate matter: see [Update a private pack](#update-a-private-pack).
- The documentation address.

## Path A: upgrade in place

Upgrade `tradeflow-engine` with whatever installed it.

**pip or `uv pip`, in a virtual environment:**

```bash
python -m pip install -U tradeflow-engine      # or: uv pip install -U tradeflow-engine
```

From 2.3.0 this ended with `quantred` 2.4.0 and `tradeflow-engine` 2.3.1 installed,
`pip check` clean, both `quantred` and `tradeflow` on the path printing
`quantred 2.4.0`, and `import tradeflow.strategies.base` working (the same objects as
`quantred`'s) with a `DeprecationWarning`. Extras carry over: the transition package
forwards every `quantred` extra, so `tradeflow-engine[mcp]` resolves to `quantred[mcp]`,
and pip does not remove packages an extra installed earlier.

**`uv tool`.** `uv tool` exposes only the executables of the package you installed.
`uv tool upgrade tradeflow-engine` moved a tool installed as `tradeflow-engine[mcp]` to
2.3.1 with `quantred` 2.4.0 and kept the `mcp` extra, and `tradeflow --version` printed
`quantred 2.4.0`. But only the `tradeflow` command is exposed: `quantred` is *not* on the
path from that tool. To get `quantred`, switch tools as in Path B.

A tool installed with a pin (`tradeflow-engine==2.3.0`) does not move on
`uv tool upgrade`, which reports it as pinned; `uv tool install --reinstall
tradeflow-engine` (no pin) moves it.

Whichever installer, when you are ready to finish the move, follow Path B.

## Path B: move fully now

In order: switch the requirement to `quantred`; update private packs' imports and
entry-point groups; change scripts, cron entries and MCP client configs from `tradeflow`
to `quantred`; then remove `tradeflow-engine`. Removing it takes out the shim and the
`tradeflow` command and leaves `quantred` installed.

### If you installed with `uv tool`

List what you have, including the extras you chose and any `--with` packs, and run the
old doctor:

```bash
uv tool list --show-extras --show-with
tradeflow init --check
```

The entry reads like `tradeflow-engine v2.3.0 [extras: mcp, viz]`, and `init --check`
shows an `[ok] extra:` row for each extra that is installed. Those are the extras to
re-add. Then swap:

```bash
uv tool uninstall tradeflow-engine
uv tool install 'quantred[mcp,viz]'
```

Add `--with <pack>` for any extension pack listed under `--show-with`. A pack that was
installed from a directory is listed as `name` followed by a `file://` path (for
example `my-signalsfile:///work/my-signals`); re-add it with `--with-editable
/work/my-signals`, using the path. If a pack declares a dependency on `tradeflow-engine`,
see [If you write or use an extension pack](#if-you-write-or-use-an-extension-pack).
Afterwards `quantred --version` should print `quantred <version>` and the same state
root as before (not verified end to end). `tradeflow`
is gone from the path, so update anything that called it.

### If you installed into a virtual environment

With the environment activated, run `tradeflow init --check` first and read the
`[ok] extra:` rows (`python -m pip show tradeflow-engine` does not list them), then:

```bash
python -m pip install 'quantred[mcp,viz]'
python -m pip uninstall tradeflow-engine
```

Use the extras you had. `uv pip install` and `uv pip uninstall` take the same arguments.
Reinstall editable packs afterwards, with `--no-deps` if they still depend on the old
distribution (see [below](#if-you-write-or-use-an-extension-pack)).

### If you work from a checkout

The command is unchanged for a checkout (`python main.py ...`). The environment needs
one swap, after pulling:

```bash
uv pip uninstall tradeflow-engine
uv run --extra dev --reinstall-package quantred python main.py --version
```

Do not use `uv sync` without `--inexact`: it removes editable packs installed by hand.
An earlier version of this page described rehearsing this swap, but that was against the
old shared-directory layout, which no longer exists, so it is *not verified*.

## Update a private pack

A pack that imports the engine has to change its imports eventually, because the import
package is now `quantred`. Until it does, `from tradeflow...` keeps working only while
`tradeflow-engine` 2.3.1 is installed, with a deprecation warning:

```bash
grep -rl 'tradeflow' src/ | xargs sed -i.bak -E 's/\b(from|import) tradeflow\b/\1 quantred/'
```

(`sed -i.bak` works on both GNU and macOS sed; delete the `.bak` files afterwards, and
read the diff, since this does not touch strings that mention the old name.) Then
change the dependency in the pack's `pyproject.toml` and, optionally, the entry-point
groups:

```toml
[project]
dependencies = ["quantred"]            # was "tradeflow-engine"

[project.entry-points."quantred.strategies"]   # was "tradeflow.strategies"
my_breakout = "yourfirm_signals.strategies:BreakoutStrategy"

[project.entry-points."quantred.scanners"]     # was "tradeflow.scanners"
my_liquidity = "yourfirm_signals.scanners:LiquidityScanner"
```

The group rename is optional for now: the engine reads both the old and the new group
names, so a pack that only fixes its imports keeps registering. Declare the new names in
anything you maintain.

## MCP clients

A client entry written under the old name launches `tradeflow` or
`python -m tradeflow.cli`. Both keep working only while `tradeflow-engine` 2.3.1 is
installed, and stop once it is removed (Path B). An entry whose command is a path
*inside* the old tool's environment (for example
`.../uv/tools/tradeflow-engine/bin/tradeflow`) stops working as soon as that tool is
uninstalled.

To move a client across, run `quantred mcp --self-test` and paste the absolute command
and block it prints, which uses the key and command `quantred`. The server now reports
its name as `quantred`, so a client may show it, or prefix its tools, differently from
before; agents or prompts that referred to tools by a `tradeflow` prefix may need the new
one, since a client's prefix usually follows the key in its config. How a given client
shows the renamed server was not tested.

## If you write or use an extension pack

Earlier documentation told pack authors they could depend on `tradeflow-engine`, so packs
that declare it exist. Installing one pulls `tradeflow-engine` in beside `quantred`. That
no longer collides, because the two share no files; with 2.3.1 it is the transition
package, which only depends on `quantred`, so the pack works and its `tradeflow` imports
warn. Two cases are worth knowing:

- **`tradeflow-engine` 2.3.0 is still installed** (the last release with engine code).
  Installing `quantred` beside it does not collide either, but the two run different
  code: `tradeflow` runs the old engine and `quantred` the new. QuantRed quietly ignores
  the old distribution's demo entry points. Upgrade `tradeflow-engine` to 2.3.1 or
  uninstall it.
- **You want the old distribution kept out entirely.** If you own the pack, change its
  dependency to `quantred` or drop it; the engine is always present where a pack runs.
  If you cannot edit it:

  - *uv tool.* Name the distribution in a file and exclude it; the exclusion is
    recorded and survives rebuilds of the tool environment:

    ```bash
    printf '%s\n' tradeflow-engine > excludes.txt
    uv tool install 'quantred[mcp,viz]' --with-editable ./my-pack --excludes excludes.txt
    ```

  - *Virtual environment or checkout.* Skip the pack's dependencies:
    `python -m pip install --no-deps -e ./my-pack` (checkout:
    `uv pip install --no-deps -e ./my-pack`). This also skips the pack's *other*
    dependencies, so it is only for a pack you cannot edit.

These exclusion recipes were rehearsed while the two distributions still shared a
directory, a worse case than today's; they are *not verified* against the transition
package.

Anyone who still needs the old `tradeflow` command keeps `tradeflow-engine` 2.3.1
installed: a plain `quantred` install, `uv tool install quantred` included, has no
`tradeflow` command.

The pack's own imports are covered in [Update a private pack](#update-a-private-pack).

## Availability

`quantred` has to be published to the package index before `uv tool install quantred`
works there; until then, install from a built wheel or a checkout. Publishing both
packages is the project owner's step.
