---
sidebar_position: 3
title: Migrating to QuantRed
---

# Migrating to QuantRed

The project was called TradeFlow and was distributed as `tradeflow-engine`. It is now
**QuantRed**, distributed as `quantred`, and the main command is `quantred`. This page
is for anyone who already has the old distribution installed. If you are installing for
the first time, you do not need it: see [installation](installation).

`tradeflow-engine` 2.3.1 is the last release under the old name. It behaves exactly as
2.3.0 did, except that at a terminal it prints a note pointing here; QuantRed 2.4.0 is
the first release as `quantred`.

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
- **`import quantred`** and the plugin entry-point groups. A private strategy pack
  keeps working without edits.
- **Your state.** `~/.tradeflow`, `TRADEFLOW_HOME`, the journal, trial store, bar
  cache, saved configs and your `.env` are untouched. In the rehearsal a `.env` written
  by the old install was found by the new one at the same state root.
- The documentation address and the GitHub repository.

## The one rule: uninstall the old distribution first

Both distributions own the same `quantred/` import tree and the same `tradeflow`
script. Install `quantred` while `tradeflow-engine` is still present and the two
overwrite each other; remove the old one afterwards and its uninstall deletes files the
new one needs. Uninstall first, then install.

## If you installed with `uv tool`

Before uninstalling anything, list what you have, including the extras you chose and any
`--with` packs, and run the old doctor:

```bash
uv tool list --show-extras --show-with
tradeflow init --check
```

The old entry reads like `tradeflow-engine v2.3.1 [extras: mcp, viz]`, and `init --check`
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
see [If you write or use an extension pack](#if-you-write-or-use-an-extension-pack)
before installing. Afterwards `quantred --version` and `tradeflow --version` both
print `quantred <version>` and show the same state root as before.

Running the new install first does not damage anything, but does not work either: uv
builds the new environment, then stops with `error: Executable already exists:
tradeflow use `--force` to overwrite`, and the old tool keeps running. **Do not follow
the `--force` suggestion.** It succeeds and points `tradeflow` at the new tool, but the
`uv tool uninstall tradeflow-engine` that follows then deletes that `tradeflow` link, so
the alias stops working while `uv tool list` still shows it under `quantred`. If that
already happened, uninstall the old tool (if you have not) and re-run
`uv tool install --force 'quantred[mcp,viz]'` with your extras and packs; that restores
the link. Otherwise just uninstall the old one and repeat the install.

## If you installed into a virtual environment

With the environment activated:

```bash
python -m pip uninstall tradeflow-engine
python -m pip install 'quantred[mcp,viz]'
```

Use the extras you had. `python -m pip show tradeflow-engine` does not list them; run
`tradeflow init --check` **before** uninstalling and read the `[ok] extra:` rows. Reinstall
editable packs afterwards with `python -m pip install --no-deps -e ./my-pack` (the
`--no-deps` is explained [below](#if-you-write-or-use-an-extension-pack)).

**If you installed `quantred` first** and then uninstalled the old distribution, the
environment is left without the engine files (`import quantred` fails with `No module
named quantred.cli`) and without the `tradeflow` script. Repair it with
`--force-reinstall`:

```bash
python -m pip install --force-reinstall --no-deps quantred
```

## If you work from a checkout

The command is unchanged for a checkout (`python main.py ...`). The environment is what
needs one swap, after pulling:

```bash
uv pip uninstall tradeflow-engine
uv run --extra dev --reinstall-package quantred python main.py --version
```

The uninstall removes the old distribution and, because the two share a script name, the
`tradeflow` script with it; `--reinstall-package` restores the script and installs
`quantred`. Run in that order the two never coexist. (Rehearsed on a fresh checkout of the
old revision with an editable pack, then updated to this one: one distribution remained,
both scripts were present and the pack registered.) Do not use `uv sync` without
`--inexact`: it removes editable packs installed by hand.

## MCP clients

An existing client entry whose key and command are `tradeflow` keeps launching, through
the alias. An entry whose command is a path *inside* the old tool's environment (for
example `.../uv/tools/tradeflow-engine/bin/tradeflow`) stops working once you uninstall
it: run `quantred mcp --self-test` and paste the absolute command it prints. The server now reports its name as `quantred`, so a client may show it, or
prefix its tools, differently from before.

To line it up, change the key and the command to `quantred`, or print a fresh entry
with `quantred mcp --self-test`. Agents or prompts that referred to tools by a
`tradeflow` prefix may need the new one, since a client's prefix usually follows the
key in its config. How a given client shows the renamed server was not tested.

## If you write or use an extension pack

Earlier documentation told pack authors they could depend on `tradeflow-engine`, so packs
that declare it exist. Installing one makes the installer pull the old distribution in
beside `quantred`, and the two then share one `quantred/` tree. Nothing reports it. You
notice because:

- every command starts with `Ignoring duplicate quantred.strategies contribution
  'demo_trend' ...` lines, and
- `quantred --version` may print `tradeflow <version>`, the old name (it depends on which
  copy of the files won).

(In the rehearsals the files were sometimes the old code and sometimes the new, so
behaviour is unpredictable.) If you own the pack, change its dependency to `quantred` or
drop it; the engine is always present where a pack runs. If you cannot edit it, install
it so the old distribution is kept out:

- **uv tool.** Name the old distribution in a file and exclude it. The command records the
  exclusion, so it survives later rebuilds of the tool environment:

  ```bash
  printf '%s\n' tradeflow-engine > excludes.txt
  uv tool install 'quantred[mcp,viz]' --with-editable ./my-pack --excludes excludes.txt
  ```

  (Rehearsed with a pack depending on `tradeflow-engine`: a single `quantred`
  distribution was installed, the pack registered, and uv recorded the exclusion in the
  tool's receipt. Without `--excludes` the old distribution was installed too.)

- **Virtual environment.** Skip the dependencies when installing the pack:

  ```bash
  python -m pip install --no-deps -e ./my-pack
  ```

  Without `--no-deps`, pip installs `tradeflow-engine` over `quantred` without a warning.
  `--no-deps` also skips the pack's *other* dependencies, so it is only for a pack you
  cannot edit; if you can, change the pack's dependency to `quantred` and install it
  normally.

- **Checkout.** `uv pip install -e` of such a pack pulls `tradeflow-engine` into the
  checkout's environment, and `uv run quantred` then runs the old code (it prints
  `tradeflow <version>`) while `python main.py` runs the checkout's. Install it without
  the old distribution:

  ```bash
  uv pip install --no-deps -e ./my-pack
  ```

  (the same `--no-deps` caveat applies). If it already happened, use the swap in
  [If you work from a checkout](#if-you-work-from-a-checkout):
  `uv pip uninstall tradeflow-engine`, then
  `uv run --extra dev --reinstall-package quantred python main.py --version`.

**Repair.** If it already happened in a virtual environment, remove the old distribution
and reinstall the new one over the top:

```bash
python -m pip uninstall tradeflow-engine
python -m pip install --force-reinstall --no-deps quantred
```

For a `uv tool` install, re-run the `uv tool install --force ... --excludes excludes.txt`
command above; uv rebuilds the tool environment without the old distribution.
(Rehearsed for both: `uv tool` with a pack depending on `tradeflow-engine`, and a
virtual environment with `python -m pip`, where the repair left one `quantred`
distribution with both scripts present.)

Entry-point groups and imports are unchanged, so nothing else in a pack needs editing.

## Availability

`quantred` has to be published to the package index before `uv tool install quantred`
works there; until then, install from a built wheel or a checkout. That publication is
the project owner's step and was not part of this change.
