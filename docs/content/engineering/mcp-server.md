---
sidebar_position: 16
title: MCP server & service core
---

# MCP server & service core

The MCP server exposes QuantRed's deterministic capabilities as agent tools
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
  equity-basis caveat (bar-close marks vs closed-trade P&L) and the multiple-testing warning, so an agent
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
- Screen: `run_screen` sweeps a parameter space cheaply to ask whether a family holds
  anything at all, and **journals nothing**; `confirm_screen_point` re-runs exactly one
  screened point as an ordinary backtest and **journals one trial**, the only way a
  screen's finding becomes evidence
- Probes: `run_causality_probes` perturbs what only becomes knowable after a fill and
  requires each decision to be unchanged (execution clock, same-bar ranking, benchmark
  alignment, as-of scanner); it is not the leakage probe and does not replace it
- Research: `compute_alphas`, `combine_alphas`, `compute_risk`,
  `construct_portfolio`, `compute_information`, `compute_attribution` (where realized
  active return came from: timing, factors, signals, picking; read-only, journals
  nothing), `compute_horizon`,
  `run_verdict` (the whole pipeline as one call — see
  [One-command verdict](../usage/verdict.md); it takes the allocation knobs
  `max_weight`, `max_names`, `lookback_days`, `neutralize_factors`,
  `participation_cap` and `n_points`, so an `infeasible` verdict can be answered with an
  explicit change to the universe or the caps)
- Draft code: `validate_draft_strategy_code`, `validate_draft_scanner_code`,
  `run_draft_walk_forward`
- Artifact: `render_report` (a result dict → one self-contained HTML document, the
  same renderer `--html` uses — see [HTML reports](../usage/html-reports.md))
- Campaign memory: `list_trials`, `get_trial`, `best_trials`, `analyze_trial` (what one
  run's trades did, by exit reason), `compare_trials` (are two results in fact one bet:
  return-series correlation, refused rather than caveated on thin overlap or mixed
  accounting), `get_campaign_material` (a trial's recipe, evidence and metadata, to pass
  through as `provenance.campaign`) — all read-only views of the
  [trial store](../usage/trials.md)
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

### Stopping

`serve()` owns the process's stop signals (`signal.signal`, not a loop handler: tool
functions are synchronous and run on the loop's thread, so a loop callback would wait
for the call in progress). The first SIGINT/SIGTERM ignores further ones, prints what
stopping did (calls handled this session, where the journal is, that a request in
progress was abandoned) and leaves through `utils.streaming.exit_now`, which flushes
logging and the std streams and skips interpreter finalization.

That last part is forced by the dependency, not chosen for tidiness: the SDK's stdio
transport reads `sys.stdin.buffer` through `anyio.wrap_file`, i.e. a worker thread
blocked in `read(2)` that cannot be cancelled or woken (closing the stream needs the
buffer lock that read holds). Without the handler, one interrupt left `anyio.run`
waiting forever, and interrupts during the resulting finalization aborted the
interpreter with a fatal error about that lock - reproduced on 3.12 and 3.14 with a pipe
and a pseudo-terminal as stdin. Leaving early is safe because the journal append closes
its file and the trial index commits per write, so nothing finished is buffered. EOF
(a client closing stdin) is untouched and returns normally; a client that vanishes
mid-write surfaces as a broken pipe and is reported as a disconnect.
`tests/test_mcp_shutdown.py` runs the real server through each of these.

### Pointing a client at it

An installed copy needs no paths:

```json
{"mcpServers": {"quantred": {"command": "quantred", "args": ["mcp"]}}}
```

From a checkout, the equivalent is the script it wraps:

```json
{"mcpServers": {"quantred": {"command": "uv",
  "args": ["run", "--project", "/path/to/tradeflow", "python", "main.py", "mcp"]}}}
```

**One prerequisite, and a lazy second.** The server needs the `mcp` extra; without it
the command prints its reason and exits, which a client shows as a server that would not start. Market-data credentials are *not* a startup requirement: the data client is built
on the first tool that needs bars (`build_server(data_client=None)` holds a builder, not a
client), so a keyless server still lists its tools and answers the catalogue and
trial-history tools. A bar tool then raises a tool error saying the keys are missing and
to run `init` from a terminal; a failed build is not remembered, so the next call retries.
The server holds only a data client either way, so the read-only wall is unchanged.
`quantred mcp --offline [--cache-dir DIR]` builds the cache-only client
(`build_data_client(offline=True)`) instead: no keys, no network, and a request outside the
cache is a `CacheMiss` naming the warm command. The flags are part of the one launch
command (`services.setup.mcp_launch_command`), so the self-test and the printed client
config carry them together.

**Checking discovery without a chat.** `quantred mcp --self-test` launches a child copy
of the server over stdio (the interpreter running the CLI, so an installed copy and a
checkout behave alike), initializes with the real MCP SDK client, lists the tools and
calls `list_strategies` and `list_scanners`. It prints the tool count and how many
strategies and scanners are demo versus installed, exits 0 when that worked and 1 naming
the step that failed, with whatever the child printed on stderr. It goes through the SDK
client on purpose: the in-process `call_tool` skips the output-schema validation a real
client performs, which is how a mistyped discovery row once passed every test. It does
**not** connect anything to a chat; your MCP client launches its own copy from its
config, and the command says so on every run. The discovery rows are typed
(`StrategyEntry`, `ScannerEntry`): `demo` is a boolean and a strategy that declares no
timeframe reports `timeframe: null`, not an empty string.

Both reach the same `tradeflow.cli:main`. Note that they resolve **different state
roots** (`~/.tradeflow` vs. the checkout), so an agent and a human should be pointed
at the same one — or `TRADEFLOW_HOME` set explicitly — if they are meant to share a
campaign's trial history.

**Bad input is a tool error.** An unknown trial id (`get_trial`, `analyze_trial`,
`get_campaign_material`), fewer than two known ids for `compare_trials`, and an unknown
config name for `load_config` all raise, so the client sees `isError` and the message
names the tool to use next (`list_trials`, `list_configs`). They used to come back as an
ordinary result holding an `error` field, which a client reads as success. A payload field
is kept only where the result is genuinely partial: `compare_trials` with two or more
known ids still answers, and lists the rest in `unknown_trial_ids`. The services keep
their return shapes (the CLI renders them); the raise happens in this module.

**Every result is strict JSON.** JSON has no NaN or infinity, so every tool result passes
through `mcp.server._json_safe` on its way out, which spells a non-finite number `"nan"`,
`"inf"` or `"-inf"`. A field that is normally a number can therefore be one of those
strings; read it as undefined (or unbounded), never as zero.

**`save_config` and `load_config` take a plain file name.** An agent supplies the name, so
`services.configs.config_filename` refuses anything with a directory component, `..`, an
absolute path, an empty stem, a `:`, a control character or a Windows device name, and
every config is read and written directly under the configs directory (a read used to
accept any existing path). A name already in use - compared without case - is refused
unless `overwrite=true`, since a saved config may be one a person has promoted.
`save_config` also refuses, before writing, a strategy or scanner that is not registered,
params that do not construct the strategy or that carry a near-miss of a declared
parameter name (a typo; other undeclared keys are strategy-specific and pass), and a book
limit nothing enforces (the same `refuse_unknown_params` / `refuse_unknown_limits` the run
paths use). `provenance` accepts
only the `Provenance` dataclass's fields; an unknown key is refused with the accepted list
rather than a constructor `TypeError`. There is no trial-id field: the trial id goes in
`notes`, or in the `campaign` block `get_campaign_material` returns.

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

## What this surface deliberately withholds

The tool surface reaches everything the CLI can express **except** the evidence-gated
construction families, which are withheld on purpose:

- **Conditional risk**, the **Black–Litterman posterior**, and the **multi-period aim
  policy** ship *off* because their own adoption gates do not clear on this
  repository's data. The command line makes a human read that before using them; an
  agent reads a description as fact and acts on it at machine speed, so offering the
  same knob as a neutral argument would make this the easier way to switch on a feature
  nothing has validated. Being reachable is not being validated.
- `trade_rate` is withheld for a second reason worth keeping distinct: the service
  passes it only when `policy` is `"aim"`, so exposing it without the gated policy would
  be a knob that reaches nothing.

`mcp.server.DEFERRED_PARAMS` is that list, with a reason per parameter, and a test
requires every service parameter to be either exposed or listed there. Revisit only when
a gated feature is promoted.

Everything else is reachable. `construct_portfolio` takes the long/short book
(`book`, `gross_leverage`, `short_max_weight`), the benchmark-relative solve
(`benchmark_holdings`, `benchmark_premium`), `neutralize_factors`, `min_weight`,
`current_weights` and the cost assumptions; `compute_alphas` takes `neutralize_factors`,
so `neutralized_against` reports what was actually removed rather than always being
empty; and `compute_attribution` is exposed as a read-only diagnostic that journals
nothing.

### An argument this surface does not accept is an error

The framework validates a call against the tool's schema and then **drops** every key the
schema does not declare — nothing raises, and the tool returns a good result computed
without the argument. So an agent told to "turn on conditional risk" could pass
`conditional="ewma"`, get a portfolio back, and report that it had done so. The wall
held; the agent's account of what it did did not, which is the failure this whole surface
is designed against.

Unknown arguments are now refused at dispatch, and the refusal says which kind of mistake
it was:

```
construct_portfolio does not accept: conditional.
  conditional: withheld from this surface — evidence-gated: the conditional-risk
  adoption gate does not clear.
Refused rather than ignored: an argument dropped in silence leaves you believing it
was applied.
```

```
construct_portfolio does not accept: targt_te.
  targt_te: not a parameter of this tool. Did you mean target_te?
```

A typo is refused for the same reason a withheld knob is: it silently leaves the default
in place. A human would see that in the output and wonder; an agent has nothing to wonder
at.

It is wrapped at dispatch rather than declared per tool because `**kwargs` cannot express
it — the framework turns that into a *required* schema property called `extra`, changing
every tool's contract to fix a problem in none of them. Any doubt about what a tool
accepts passes the call through untouched: refusing on a guess would be worse than the
silence it replaces.

The parity guard enumerates the **service signature** rather than a remembered list, so
a parameter added to a service later fails the test instead of quietly becoming
unreachable — which is how the previous gap opened. A second test calls each tool and
asserts the service received the value, because presence in the schema is not reach: the
original defect was a tool advertising a `neutralized_against` field it had no parameter
to populate.

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
verdict as its advancement criterion, and an `evidence` block beside it: the same
promotion prerequisites, evidence completeness and per-fold excess return the CLI
prints, from one service function (`walk_forward_evidence`) that both surfaces call.
`promotable` is the statistical gates only; an agent reads `evidence.clear` and
`evidence.evidence_completeness` before acting on it, and an unevaluated check is
unknown rather than passed. A memoized answer keeps no per-fold detail, so its
`evidence` is null and `evidence_unavailable` says why. The glossary spells out the deflated-Sharpe
/ multiple-testing trap. These keep an agent from optimizing and then trusting the
in-sample Sharpe.
