---
sidebar_position: 2
title: Optional extras
---

# Optional extras

The base install is lean. Each capability below is an opt-in extra; `init --check`
reports which are installed and prints the exact command for the ones that are not.

| Extra | Provides | Unlocks |
|-------|----------|---------|
| `store` | `pyarrow`, `polars`, `duckdb` | the bar cache (`--cache`/`--offline`) and the Parquet bar store |
| `viz` | `matplotlib` | charts in `--chart` and HTML reports (reports fall back to tables) |
| `mcp` | `mcp` | the MCP server for AI clients |
| `optimize` | `scikit-learn` | `optimize --method bayesian` |
| `portfolio` | `ortools` | `allocate` (the constraint solver) |
| `ai` | `anthropic` | `research --provider anthropic` |
| `openai` | `openai` | `research --provider openai` |

An extra counts as installed only when **every** package it provides imports; `store`
is three packages, and having one of them is not enough.

## Adding an extra to an installed copy

`uv tool install --force` **replaces** the tool environment. Naming only the new
extra therefore removes every other extra, and every pack added with `--with`, that
the environment had before. List all the extras you want, and re-add your packs:

```bash
uv tool install --force 'tradeflow-engine[mcp,store,viz]==<installed version>'
```

Pinning to the running version keeps this from being an upgrade. In an environment
you manage yourself with `pip`, extras are additive:

```bash
python -m pip install 'tradeflow-engine[mcp]==<installed version>'
```

Quote the requirement: an unquoted `[...]` is a glob in zsh.

In a checkout, `uv sync --extra a --extra b` is the equivalent, and it likewise
removes extras that are not named.

The error messages and `init --check` print these commands with the extras you
already have filled in.
