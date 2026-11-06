---
sidebar_position: 3
title: Migrating to QuantRed
---

# Migrating to QuantRed

The project was called TradeFlow and was distributed as `tradeflow-engine`. It is now
**QuantRed**, distributed as `quantred`, and the main command is `quantred`. This page
is for anyone who already has the old distribution installed. If you are installing for
the first time, you do not need it: see [installation](installation).

Commands below were run with local wheels standing in for the package index, in
throwaway environments. Anything that was not run that way is labelled.

## What changed

| | Before | Now |
| --- | --- | --- |
| Display name | TradeFlow | QuantRed |
| Distribution (what you install) | `tradeflow-engine` | `quantred` |
| Main command | `tradeflow` | `quantred` |
| MCP server name (what a client sees) | `tradeflow` | `quantred` |

## What did not change

- **The `tradeflow` command still works.** It is installed next to `quantred` as an
  alias for the same program, so scripts and cron entries keep running.
- **`import tradeflow`** and the plugin entry-point groups. A private strategy pack
  keeps working without edits.
- **Your state.** `~/.tradeflow`, `TRADEFLOW_HOME`, the journal, trial store, bar
  cache, saved configs and your `.env` are untouched. In the rehearsal a `.env` written
  by the old install was found by the new one at the same state root.
- The documentation address and the GitHub repository.

## The one rule: uninstall the old distribution first

Both distributions own the same `tradeflow/` import tree and the same `tradeflow`
script. Install `quantred` while `tradeflow-engine` is still present and the two
overwrite each other; remove the old one afterwards and its uninstall deletes files the
new one needs. Uninstall first, then install.

## If you installed with `uv tool`

List what you have, including the extras you chose and any `--with` packs:

```bash
uv tool list --show-extras --show-with
```

The old entry reads like `tradeflow-engine v2.3.0 [extras: mcp, viz]`. Then swap, using
the same extras:

```bash
uv tool uninstall tradeflow-engine
uv tool install 'quantred[mcp,viz]'
```

Add `--with <pack>` (or `--with-editable <path>`) for any extension pack that was listed
under `--show-with`. Afterwards `quantred --version` and `tradeflow --version` both
print `quantred <version>` and show the same state root as before.

Running the new install first does not damage anything, but does not work either: uv
builds the new environment, then stops with `error: Executable already exists:
tradeflow (use --force to overwrite)`, and the old tool keeps running. Uninstall the old
one and repeat the install.

## If you installed into a virtual environment

With the environment activated:

```bash
python -m pip uninstall tradeflow-engine
python -m pip install 'quantred[mcp,viz]'
```

Use the extras you had (`python -m pip show tradeflow-engine` does not list them; if
you do not remember, `quantred init --check` prints the install command for what you
use). Reinstall editable packs afterwards if you had any.

**If you installed `quantred` first** and then uninstalled the old distribution, the
environment is left without the engine files (`import tradeflow` fails with `No module
named tradeflow.cli`) and without the `tradeflow` script. Repair it with
`--force-reinstall`; that is the form rehearsed (with `uv pip`):

```bash
python -m pip install --force-reinstall --no-deps quantred
```

(The `python -m pip` spelling of the repair was not run; the `uv pip` equivalent was.)

## If you work from a checkout

The command is unchanged for a checkout (`python main.py ...`). The environment is what
needs one swap, after pulling:

```bash
uv pip uninstall tradeflow-engine
uv run --extra dev --reinstall-package quantred python main.py --version
```

The first `uv run` after the rename installs `quantred` beside the old distribution; the
uninstall removes the old one and, because the two share a script name, the
`tradeflow` script with it, which the `--reinstall-package` restores. (Rehearsed on this repository's own
environment; not on a fresh clone.) Do not use
`uv sync` without `--inexact`: it removes editable packs installed by hand.

## MCP clients

An existing client entry whose key and command are `tradeflow` keeps launching, through
the alias. The server now reports its name as `quantred`, so a client may show it, or
prefix its tools, differently from before.

To line it up, change the key and the command to `quantred`, or print a fresh entry
with `quantred mcp --self-test`. Agents or prompts that referred to tools by a
`tradeflow` prefix may need the new one, since a client's prefix usually follows the
key in its config. How a given client shows the renamed server was not tested.

## If you write an extension pack

A pack that declares `dependencies = ["tradeflow-engine>=..."]` makes an installer pull
the old distribution back in beside `quantred`, which is the corruption described
above. Either drop the dependency (the engine is always present where your pack runs),
or replace it with `quantred`, or install the pack with `--no-deps`:

```bash
uv pip install --no-deps -e ./my-pack
```

Entry-point groups and imports are unchanged, so nothing else in a pack needs editing.

## Availability

`quantred` has to be published to the package index before `uv tool install quantred`
works there; until then, install from a built wheel or a checkout. That publication is
the project owner's step and was not part of this change.
