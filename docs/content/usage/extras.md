---
sidebar_position: 2
title: Optional extras
---

# Optional extras

The base install is lean. Each capability below is an opt-in extra; `init --check`
reports which are installed and prints the exact command for the ones that are not.

Three names are easy to confuse: the **distribution** you install is
`tradeflow-engine`, the **command** you run is `tradeflow`, and an **extra** is a
bracketed option on the distribution, as in `tradeflow-engine[mcp]`.

| Extra | Provides | Unlocks | Needed for |
|-------|----------|---------|------------|
| `store` | `pyarrow`, `polars`, `duckdb` | the bar cache (`--cache`/`--offline`) and the Parquet bar store | nothing required; repeat runs are faster |
| `viz` | `matplotlib` | charts in `--chart` and HTML reports (reports fall back to tables) | nothing required |
| `mcp` | `mcp` | the MCP server for AI clients | `tradeflow mcp` and the MCP task in `init --check --for mcp` |
| `optimize` | `scikit-learn` | `optimize --method bayesian` | the Bayesian method only |
| `portfolio` | `ortools` | `allocate` (the constraint solver) | `allocate` |
| `ai` | `anthropic` | `research --provider anthropic` | `research` with that provider |
| `openai` | `openai` | `research --provider openai` | `research` with that provider |

An extra counts as installed only when **every** package it provides imports; `store`
is three packages, and having one of them is not enough.

## Choosing extras at install time

```bash
uv tool install 'tradeflow-engine[mcp,store,viz]'
```

In an activated virtual environment, use `python -m pip install
'tradeflow-engine[mcp,store,viz]'`. Quote the requirement: an unquoted `[...]` is a
glob in zsh.

## Adding an extra to an installed copy

`uv tool install --force` **replaces** the tool environment. Naming only the new
extra therefore removes every other extra, and every pack added with `--with`, that
the environment had before. List all the extras you want, and re-add your packs:

```bash
uv tool install --force 'tradeflow-engine[mcp,store,viz]==<installed version>' --with <your-pack>
```

Pinning to the running version keeps this from being an upgrade. In an environment
you manage yourself with `pip`, extras are additive:

```bash
python -m pip install 'tradeflow-engine[mcp]==<installed version>'
```

The error messages and `init --check` print these commands with the extras you
already have filled in, and the running version.

## Contributors (source checkout)

`uv sync --extra a --extra b` is the equivalent, and it likewise removes extras that
are not named. Never run a bare `uv sync` in a checkout that holds an editable pack.
