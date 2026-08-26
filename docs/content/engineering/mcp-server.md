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
tradeflow/mcp/server.py        ← thin adapter, NO business logic
        │  calls
        ▼
tradeflow/services/*.py        ← plain functions over engine/optimizer/walk-forward/analytics
        │
        ▼
existing tradeflow/ layers (unchanged)
```

## The shared service core (`tradeflow/services/`)

One orchestration code path, reused by the CLI, the MCP server, and the research
agent — no business logic lives in any adapter. Every function takes a data-only
`MarketDataClient` and returns a JSON-serializable dict:

- `registry.py` — `STRATEGIES` / `SCANNERS` registries and discovery
  (`list_strategies`, `list_scanners`, `get_param_ranges`).
- `analysis.py` — `run_scan`, `run_backtest`, `run_optimization`,
  `run_walk_forward`, draft code validation/evaluation helpers, and
  `summarize_bars`. Large outputs (trade tables, full optimization grids) are
  written to an artifact file under `logs/artifacts/` and referenced by path —
  never inlined. Optimization output is capped to the top-N rows with a
  truncation count.
- `glossary.py` — `metrics_glossary()`: definition + pitfalls per metric, plus the
  closed-trade equity-curve caveat and the multiple-testing warning, so an agent
  doesn't over-trust in-sample Sharpe.
- `configs.py` — `save_config` / `load_config` / `list_configs` over the config
  store.
- `audit.py` — append-only `logs/mcp_audit.jsonl` (tool, inputs, run id, git SHA,
  server timestamp) so every decision is replayable.
- `data.py` — `build_data_client()` constructs **only** a historical-data client,
  never a broker.

## The server (`tradeflow/mcp/server.py`)

A FastMCP adapter (the `mcp` SDK is imported lazily, behind the `mcp` extra). Each
tool is a typed function that calls a service function, logs the call, and returns
JSON. The exposed surface:

- Discovery: `list_strategies`, `list_scanners`, `get_param_ranges`
- Analyze: `run_scan`, `run_backtest`, `run_optimization`, `run_walk_forward`,
  `get_metrics_glossary`, `summarize_bars`
- Research: `compute_alphas`, `combine_alphas`, `compute_risk`,
  `construct_portfolio`, `compute_information`, `compute_horizon`,
  `run_verdict` (the whole pipeline as one call — see
  [One-command verdict](../usage/verdict.md))
- Draft code: `validate_draft_strategy_code`, `validate_draft_scanner_code`,
  `run_draft_walk_forward`
- Artifact: `render_report` (a result dict → one self-contained HTML document, the
  same renderer `--html` uses — see [HTML reports](../usage/html-reports.md))
- Campaign memory: `list_trials`, `get_trial`, `best_trials` (read-only views of
  the [trial store](../usage/trials.md))
- Propose (writes a file, never live state): `save_config`, `load_config`,
  `list_configs`

Every CLI research capability has an MCP equivalent, except anything touching live
trading — that is the parity principle, and the exception is the whole safety model.

Draft code tools are for the workbench phase. They let an agent check or
walk-forward-test generated/private strategy code in memory without putting that
source in this repository. Once a candidate deserves a name, ship it in a private
package using the `tradeflow.strategies` or `tradeflow.scanners` entry-point group;
the MCP server will discover it at startup and the normal named-strategy tools will
work.

### Pointing a client at it

An installed copy needs no paths:

```json
{"mcpServers": {"tradeflow": {"command": "tradeflow", "args": ["mcp"]}}}
```

From a checkout, the equivalent is the script it wraps:

```json
{"mcpServers": {"tradeflow": {"command": "uv",
  "args": ["run", "--project", "/path/to/tradeflow", "python", "main.py", "mcp"]}}}
```

Both reach the same `tradeflow.cli:main`. Note that they resolve **different state
roots** (`~/.tradeflow` vs. the checkout), so an agent and a human should be pointed
at the same one — or `TRADEFLOW_HOME` set explicitly — if they are meant to share a
campaign's trial history.

## Descriptions are an interface, not documentation

A human who reads a stale doc can notice it is stale. An agent cannot: it reads a
description as a statement of fact and acts on it at machine speed, and every action
it takes burns a journaled trial. So descriptions here are treated as a contract and
pinned by tests:

- **Metric vocabulary is pulled from the glossary**, not restated.
  `glossary.definitions_for()` supplies the canonical definition (and pitfall) of
  every metric a tool reports, appended to its description at registration time. Two
  descriptions of one metric would drift; one definition with two readers cannot.
- **Journaling is stated wherever it happens.** Every tool in `JOURNALING_TOOLS`
  says, in identical words, that the call records a trial, counts toward the
  campaign's multiple-testing total, and serves a memoized prior run unless forced.
- **Evidence-gated features are never presented as neutral options.** Conditional
  risk, the aim trading policy, and the Black–Litterman posterior each ship off
  because their own adoption gates do not clear on this repository's data; the
  descriptions of tools near them say so rather than listing a flag.
- **The leaderboard's honesty rules live in the payload.** `best_trials` returns its
  `rank_by`, each row's family `n_trials`, and the caveat text as *data* — an agent
  never sees a terminal's caveat line, so the caveat has to travel with the numbers.

The mechanism is a small registration helper that composes each tool's description
from its docstring plus the shared, glossary-derived pieces. Tests assert every
registered tool has a substantive description, that journaling tools mention
journaling and memoization, and that gated tools name their gate. String assertions
are crude, but they catch silent regressions to stale text, which is the failure that
actually happens.

## Known gap

The tool surface still lags the CLI on some parameters. What is genuinely missing,
as of the description audit:

- `neutralize_factors` ([factor-neutral alphas](./alphas.md#neutralization)) —
  results echo a `neutralized_against` field, but via this surface it is always
  empty.
- `construct_portfolio` solves the long-only, cash-relative book. It **is**
  cost-aware (the objective carries turnover and square-root impact by default), but
  `book`/`gross_leverage`/`short_max_weight`
  ([long/short](./portfolio-construction.md#longshort---book-market-neutral)),
  `benchmark_holdings` ([benchmark-relative](./portfolio-construction.md#benchmark-relative-construction---benchmark-holdings)),
  `conditional` ([conditional risk](./risk-model.md#conditional-risk)),
  `posterior` ([Black–Litterman](./portfolio-construction.md#blacklitterman---posterior-bl)),
  and `policy`/`trade_rate` ([multi-period trading](./multi-period-trading.md)) are
  not arguments here. The tool's own description says so, so an agent does not assume
  otherwise.
- `compute_attribution`, `run_conditional_risk_ab`, `run_policy_ab`, and
  `evaluate_conditional_risk` are not exposed as tools.

Wiring these through is a small, mechanical follow-up — the underlying service
functions already support everything; only the tool signatures are behind.

## The hard wall

The safety model is **structural absence**, not a check that can be prompt-injected
around: there is no `place_order`, `start_live`, `cancel`, `set_paper_trade`, or
account/position-mutation tool. `EXPOSED_TOOLS` is asserted disjoint from
`FORBIDDEN_TOOLS` in the test suite, and `build_server` refuses to start unless its
client is a plain `MarketDataClient` with no broker attached. Promoting a config to
live is a manual human step outside MCP.

## Honest-evaluation guardrails for agents

`run_optimization` results are explicitly labeled in-sample and tell the caller to
validate with `run_walk_forward`. `run_walk_forward` returns the promotion-gate
verdict as its advancement criterion. The glossary spells out the deflated-Sharpe
/ multiple-testing trap. These keep an agent from optimizing and then trusting the
in-sample Sharpe.
