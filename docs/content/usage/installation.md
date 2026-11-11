---
sidebar_position: 1
title: Installation
---

# Installation

QuantRed is published on PyPI as **`quantred`**. You do not need to clone the repository
to use it. This page covers installing, checking, upgrading and removing the published
package; working on QuantRed itself from a checkout is
[further down](#from-a-checkout-contributors), kept separate on purpose.

## What you need

- **Python 3.10 or newer.** The package declares `>=3.10` and is tested on 3.10 through
  3.14. `uv` fetches a suitable Python itself when your system has none (pin one with
  `--python 3.12`).
- **Nothing else for the demo.** No keys, no account, no network, no compiler: indicators
  are pure pandas/numpy, so the install is wheels only.
- **Free Alpaca paper-trading keys** only once you want real market data
  ([first run](first-run) sets them up). Never live keys.

## Install it

Pick one. All three give you a `quantred` command.

### With uv (recommended)

```bash
uv tool install 'quantred[mcp,store,viz]'
quantred demo
```

No `uv`? [Install it first](https://docs.astral.sh/uv/getting-started/installation/).
`uv tool` puts QuantRed in its own isolated environment and its `quantred` command on your
`PATH`. If your shell then says `command not found: quantred`, run `uv tool update-shell`
and open a new terminal (`uv tool dir --bin` shows where the command went).

### With pipx

```bash
pipx install 'quantred[mcp,store,viz]'
quantred demo
```

Same isolation, same command. The install hints QuantRed prints name `uv` and `pip`
only, so under pipx translate them: `pipx install --force '<requirement>'` where they say
`uv tool install --force`, and re-add any packs you injected.

### With pip, in a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'quantred[mcp,store,viz]'
quantred demo
```

The command exists only while the environment is active. A venv made with `uv venv` has
no `pip` in it; use `uv pip install 'quantred[mcp,store,viz]'` there.

### Extras, and the quoting

The bracketed part is a list of opt-in capabilities. The three above are the MCP server
(an AI client launches it), the local bar cache that `cache warm` and every `--offline`
run read, and charts in HTML reports. `quantred` on its own works for the demo and
nothing is lost by starting lean: the [optional extras](extras) page has the full table and
how to add one later.

**Quote the requirement.** Unquoted, `quantred[mcp]` is a glob in zsh and fails with
`no matches found`.

### The old name

The package used to be called `tradeflow-engine`, with a `tradeflow` command. Installing
`quantred` is the whole move for a new reader; if you already have the old one installed,
see [migrating to QuantRed](migrating-to-quantred). `quantred` itself installs no
`tradeflow` command.

## Check it works

```bash
quantred --version
quantred demo
quantred init --check
```

- `--version` prints the version, which copy is running, and **where state lives**.
  Read the `state root` line once: it is where your trial history will accumulate.
- `demo` runs the full pipeline on synthetic data with no keys and ends in a refusal to
  promote the strategy. That refusal is correct. If it appears, the engine works.
- `init --check` is the doctor. It reports, row by row, what is set, what is missing and
  which extras are installed, writes nothing and makes no network call. A fresh install
  with no keys reports `ready for: demo`, which is right. The rows are explained on the
  [first run](first-run#4-check-what-is-ready-init---check) page.

If a command fails instead, see [when something goes wrong](first-run#when-something-goes-wrong).

## Where state lives

The research journal, trial store, bar cache, saved configs, position ledger and your
`.env` all live under one directory, the **state root**:

| Resolution order | Value |
|---|---|
| 1 | `$QUANTRED_HOME`, when set |
| 2 | `$TRADEFLOW_HOME`, the variable's old name, when set and the first is not |
| 3 | `~/.quantred` (or `~/.tradeflow` if that exists and `~/.quantred` does not) |

It is the same wherever the software came from: an installed copy, a source checkout and
the container all read the same directory. A checkout does **not** keep its own state, so
a private strategy's trial evidence never sits inside a git working tree.

Why one directory matters: the multiple-testing correction rests on a **single** journal
accumulating every trial. A campaign split across two roots would deflate its Sharpe
against half the evidence, and nothing would error. Pick a root, keep it, and set
`QUANTRED_HOME` in every place that launches QuantRed (a shell profile, a scheduler, an
MCP client's config) if you use a non-default one. `quantred --version` and
`init --check` both print the resolved root.

Uninstalling the package does not touch this directory (see [below](#uninstall)).

:::note Older releases
Releases through 2.5.0 use `TRADEFLOW_HOME` and `~/.tradeflow` and do not read
`QUANTRED_HOME`; 2.6.0 introduced the current names. If `quantred --version` shows a `.tradeflow` state root, that is why;
upgrade, and the existing directory keeps being read. The pages in this guide describe
the current names.
:::

## Upgrade

```bash
uv tool upgrade quantred                              # uv tool
pipx upgrade quantred                                 # pipx
python -m pip install --upgrade quantred              # pip, venv active
```

`uv tool upgrade` keeps the extras and `--with` packs the tool was installed with. A
tool installed with a pinned version (`quantred==2.5.0`) is reported as pinned and does
not move; reinstall it without the pin. State is not touched by an upgrade, and the
journal carries across versions. An accounting-version change in a release note means old
results are filtered out of the default `trials list`, not deleted; the
[changelog](../changelog) says when that happens.

Adding an extra later is **not** an upgrade and is not additive under `uv tool`; see
[adding an extra to an installed copy](extras#adding-an-extra-to-an-installed-copy).

## Uninstall

```bash
uv tool uninstall quantred                            # uv tool
pipx uninstall quantred                               # pipx
python -m pip uninstall quantred                      # pip, venv active
```

This removes the command and its environment. It leaves the state root alone: your
journal, trial history, cache, saved configs and `.env` stay in `~/.quantred` (or wherever
`QUANTRED_HOME` points). Delete that directory yourself to remove everything. Do that
only with nothing running, since it holds the position ledger and any standing halt.

## From a checkout (contributors)

Everything above is for using QuantRed. This section is for working **on** it. The
Makefile targets and `uv run python main.py` are the checkout forms of the `quantred`
commands used everywhere else in these docs.

You need **either** [uv](https://docs.astral.sh/uv/getting-started/installation/)
(runs the app locally) **or** [Docker](https://docs.docker.com/get-docker/) (runs it in a
container, no local Python needed). Node.js 18+ is only needed to build these docs.

### Option A: local, with uv

```bash
git clone https://github.com/makedirectory/quantred
cd quantred
make install        # == uv sync --inexact
```

`uv` reads `pyproject.toml`, creates a virtual environment and installs the pinned
dependencies from `uv.lock`. The base install is intentionally small: `alpaca-py`,
`pandas`, `numpy`, `pytz`. There is no compiler step. Run commands as
`uv run python main.py <command>`, or `uv run quantred <command>`.

Optional extras in a checkout:

```bash
make install-optimize     # scikit-learn  -> Bayesian parameter optimization
make install-portfolio    # Google OR-Tools -> portfolio allocation
uv sync --inexact --extra dev     # everything the test suite uses
```

Use `--inexact`: plain `uv sync` is exact and uninstalls whatever the lockfile does not
name, which removes the dev tools and any pack installed with `uv pip install -e`. A
checkout does not read a `.env` in its own root; state still goes to the state root. To
put it back inside the working tree, set `QUANTRED_HOME` to it explicitly
(`quantred init --dev-local-state` prints the line).

### Option B: Docker

```bash
make docker-build                     # build the image
make docker-run                       # prints the CLI help; starts nothing
make docker-run CMD="quantred demo"   # run any command in the image
```

With no command the image only prints help (turning the container on must not turn
trading on, not even paper trading). It reads credentials from `/state/.env` (it sets
`QUANTRED_HOME=/state`), so mount your host's state-root `.env` there, read-only:

```bash
docker run --rm -v "$HOME/.quantred/.env:/state/.env:ro" quantred \
    quantred backtest --symbols NVDA,META --start 2024-01-02 --end 2024-04-01
```

A plain `docker run` keeps no state between runs; [Docker](docker) covers the compose
stack, whose named volumes keep the journal and cache.

### Verify a checkout

```bash
make test           # offline test suite - no API keys or network needed
make demo           # the whole pipeline on synthetic data (also keyless)
```

A green `make test` confirms the engine, scanner, optimizer and portfolio allocator are
wired correctly; `make demo` shows them working end to end. Next:
**[Configuration](configuration)**.
