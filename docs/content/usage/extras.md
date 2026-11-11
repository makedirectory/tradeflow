---
sidebar_position: 2
title: Optional extras
---

# Optional extras

The base install is lean. Each capability below is an opt-in extra; `init --check`
reports which are installed (an `[ok] extra:` row each) and lists the ones that are not. A
command that needs a missing extra prints the exact install command, and
`init --check --for <task>` prints it for that task's needs.

Three names are easy to confuse: the **distribution** you install is
`quantred`, the **command** you run is `quantred` (the old `tradeflow` command exists only with the `tradeflow-engine` 2.3.1 transition package), and an **extra** is a
bracketed option on the distribution, as in `quantred[mcp]`.

| Extra | Provides | Unlocks | Needed for |
|-------|----------|---------|------------|
| `store` | `pyarrow`, `polars`, `duckdb` | the bar cache (`--cache`/`--offline`, `cache warm`) and the Parquet bar store | required for any cached or offline run, including `mcp --offline`; without it those commands say so and print the install command (`init --check --offline` and `mcp --offline --self-test` check it). Plain online runs need nothing |
| `viz` | `matplotlib` | charts in `--chart` and HTML reports (reports fall back to tables) | nothing required |
| `mcp` | `mcp` | the MCP server for AI clients | `quantred mcp` and the MCP task in `init --check --for mcp` |
| `optimize` | `scikit-learn` | `optimize --method bayesian` | the Bayesian method only |
| `portfolio` | `ortools` | `allocate` (the constraint solver) | `allocate` |
| `ai` | `anthropic` | `research --provider anthropic` | `research` with that provider |
| `openai` | `openai` | `research --provider openai` | `research` with that provider |

An extra counts as installed only when **every** package it provides imports; `store`
is three packages, and having one of them is not enough.

## Choosing extras at install time

```bash
uv tool install 'quantred[mcp,store,viz]'
```

In an activated virtual environment, use `python -m pip install
'quantred[mcp,store,viz]'`. Quote the requirement: an unquoted `[...]` is a
glob in zsh.

## Adding an extra to an installed copy

`uv tool install --force` **replaces** the tool environment. Naming only the new
extra therefore removes every other extra, and every pack added with `--with`, that
the environment had before. List all the extras you want, and re-add your packs:

```bash
uv tool install --force 'quantred[mcp,store,viz]==<installed version>' --with <your-pack>
```

Pinning to the running version keeps this from being an upgrade. In an environment
you manage yourself with `pip`, extras are additive:

```bash
python -m pip install 'quantred[mcp]==<installed version>'
```

uv has no subcommand that adds a package to an installed tool, and `uv pip install`
straight into the tool's environment
is lost the next time the tool is rebuilt. Reinstalling with the full extras list is the
durable form. Under pipx the equivalent is `pipx install --force '<requirement>'`, and
packs added with `pipx inject` must be injected again.

The error messages and `init --check` print these commands with the extras you
already have filled in and the running version. For a `uv tool` install they also read
the tool's own receipt (`uv-receipt.toml`, in the tool environment) and write out every
`--with` package it was built from, so the command can be pasted as it stands. The command is scoped
to the task you ask about (`init --check --for mcp`, or `--for research --provider
anthropic`): it adds the extras that task needs and the chosen provider's SDK, and the
other absent extras are named on a line of their own rather than installed for you. When
several are needed, `init --check` prints **one** command for all of them at the end, never
one per extra: each form replaces the environment, so running two would leave only the
second's extras. A report written with `--html` does not carry this text, since
the file gets attached and forwarded; it points at this page.

## Contributors (source checkout)

`uv sync --inexact --extra a` adds an extra to the checkout's environment and leaves
everything else where it is - the `dev` tools and any pack installed with
`uv pip install -e` included. Plain `uv sync` is *exact*: it uninstalls whatever the
lockfile does not name for the extras you gave, which takes those out. `init --check`
prints the `--inexact` form, and it only needs the extras you are adding.
