---
sidebar_position: 16
title: MCP server & service core
---

# MCP server & service core

The MCP server exposes TradeFlow's deterministic capabilities as agent tools
without embedding an LLM in the engine. Intelligence lives outside and *calls in*;
the live order path is never reachable.

```
MCP client (Claude Code / Desktop / research loop)
        │  stdio (JSON-RPC)
        ▼
src/mcp/server.py        ← thin adapter, NO business logic
        │  calls
        ▼
src/services/*.py        ← plain functions over engine/optimizer/walk-forward/analytics
        │
        ▼
existing src/ layers (unchanged)
```

## The shared service core (`src/services/`)

One orchestration code path, reused by the CLI, the MCP server, and the research
agent — no business logic lives in any adapter. Every function takes a data-only
`MarketDataClient` and returns a JSON-serializable dict:

- `registry.py` — `STRATEGIES` / `SCANNERS` registries and discovery
  (`list_strategies`, `list_scanners`, `get_param_ranges`).
- `analysis.py` — `run_scan`, `run_backtest`, `run_optimization`,
  `run_walk_forward`, `summarize_bars`. Large outputs (trade tables, full
  optimization grids) are written to an artifact file under `logs/artifacts/` and
  referenced by path — never inlined. Optimization output is capped to the top-N
  rows with a truncation count.
- `glossary.py` — `metrics_glossary()`: definition + pitfalls per metric, plus the
  closed-trade equity-curve caveat and the multiple-testing warning, so an agent
  doesn't over-trust in-sample Sharpe.
- `configs.py` — `save_config` / `load_config` / `list_configs` over the config
  store.
- `audit.py` — append-only `logs/mcp_audit.jsonl` (tool, inputs, run id, git SHA,
  server timestamp) so every decision is replayable.
- `data.py` — `build_data_client()` constructs **only** a historical-data client,
  never a broker.

## The server (`src/mcp/server.py`)

A FastMCP adapter (the `mcp` SDK is imported lazily, behind the `mcp` extra). Each
tool is a typed function whose docstring is written for the agent reader, calls a
service function, logs the call, and returns JSON. The exposed surface:

- Discovery: `list_strategies`, `list_scanners`, `get_param_ranges`
- Analyze: `run_scan`, `run_backtest`, `run_optimization`, `run_walk_forward`,
  `get_metrics_glossary`, `summarize_bars`
- Research: `compute_alphas`, `combine_alphas`, `compute_risk`,
  `construct_portfolio`, `compute_information`, `compute_horizon`
- Propose (writes a file, never live state): `save_config`, `load_config`,
  `list_configs`

:::note Known gap
The research tools don't expose the CLI's `neutralize_factors` parameter yet
([factor-neutral alphas](./alphas.md#neutralization)) — their results echo a
`neutralized_against` field, but via this surface it is always empty. Wiring the
parameter through the tool signatures is a small follow-up for when the agent
surface needs it.
:::

## The hard wall

The safety model is **structural absence**, not a check that can be prompt-injected
around: there is no `place_order`, `start_live`, `cancel`, `set_paper_trade`, or
account/position-mutation tool. `EXPOSED_TOOLS` is asserted disjoint from
`FORBIDDEN_TOOLS` in the test suite, and `build_server` refuses to start unless its
client is a plain `MarketDataClient` with no broker attached. Promoting a config to
live is a manual human step outside MCP.

## Honest-evaluation guardrails for agents

`run_optimization` results are explicitly labelled in-sample and tell the caller to
validate with `run_walk_forward`. `run_walk_forward` returns the promotion-gate
verdict as its advancement criterion. The glossary spells out the deflated-Sharpe
/ multiple-testing trap. These keep an agent from optimizing and then trusting the
in-sample Sharpe.
