---
sidebar_position: 10
title: AI agents (MCP & research)
---

# AI agents (MCP & research)

QuantRed has an **opt-in** automation layer that lets an AI help tune scanners
and strategies and do market analysis — without ever putting a model in the live
order path. Two entry points share the same deterministic core:

- **`mcp`** — serve QuantRed's capabilities to any MCP client (Claude Code,
  Claude Desktop) as tools. A human (or another agent) drives.
- **`research`** — a bounded, self-pacing research loop that proposes
  hypotheses, validates them out-of-sample, and hands you a shortlist.

Both work on the **research clock** (offline, exploratory). The live order path
(`quantred live`) stays deterministic and LLM-free. Neither can place an
order — that capability is simply not wired in.

## MCP server

Needs the `mcp` extra ([optional extras](extras)). Your usual data credentials
([first run](first-run)) are needed only by the tools that fetch bars; the server
starts without them. Your MCP client **launches the server itself** from its
config; starting `quantred mcp` in a terminal only proves it starts, and connects
no chat.

```bash
claude mcp add quantred -- quantred mcp      # Claude Code
```

Any other client takes the same command in its JSON config:

```json
{ "mcpServers": { "quantred": { "command": "quantred", "args": ["mcp"] } } }
```

A desktop client often has a smaller `PATH` than your terminal, so a bare `quantred`
may not be found. `quantred init`
prints this same block with the **absolute** command for your copy, ready to paste, and
`quantred mcp --self-test` runs exactly that command (and prints the block again when
it passes), so what passes is what you configure.

**What the client does not inherit.** A desktop client launches the server without your
shell's exports. If you set `TRADEFLOW_HOME`, the printed block carries it in an `env`
entry (and the `claude mcp add` line in `--env`), because without it the client's server
would journal to the default state root and split your campaign's history. Credentials are
never printed there: put them in the `.env` the server reads (`quantred init` writes it).
The self-test child does inherit your shell, so when it passes on keys that came only from
exported variables it says so rather than letting that read as proof the client will find
them.

Run `quantred mcp` once in a terminal first. Without the extra it exits immediately
with the reason, which from inside a client looks only like a server that never started.

**Without keys.** The server still starts. The catalogue and trial-history tools
(`list_strategies`, `list_scanners`, `get_param_ranges`, `list_trials`, `get_trial`,
the glossary and so on) answer, and a tool that needs bars is refused with a message
saying the keys are missing and how to add them from a terminal (`quantred init`). To
serve from bars you already cached, with no keys at all, launch it as
`quantred mcp --offline` (optionally `--cache-dir DIR`): a request outside the cache is
refused with the `cache warm` command that fills it. `quantred mcp --self-test --offline`
checks, and prints a client block for, exactly that command.

To check that the server starts and answers, without a chat client, run
`quantred mcp --self-test`. It lists the tools and calls the discovery tools
through the real MCP client. `quantred init --check --for mcp` reports whether the
extra is installed.

The agent gets tools for discovery (`list_strategies`, `list_scanners`,
`get_param_ranges`), analysis (`run_scan`, `run_backtest`, `run_optimization`,
`run_walk_forward`, `summarize_bars`, `get_metrics_glossary`), and config
persistence (`save_config`, `load_config`, `list_configs`). Every call is logged
to `logs/mcp_audit.jsonl`. There is **no** order, live-trading, or
`PAPER_TRADE`-toggling tool — that absence is the safety model.

### Stopping the server

A server a client launched is the client's to stop: it closes the server's input and
the server exits with status 0. A server you started by hand in a terminal stops on
**one** Ctrl-C (or `SIGTERM`); further Ctrl-C presses during the stop are ignored.
It exits with status 130 (143 for `SIGTERM`) and says what that did:

```text
MCP server stopped (interrupted). It handled 0 tool call(s) this session; trials
already journaled are kept in <state root>/logs/research_journal.jsonl. A request in
progress, if any, was not completed.
```

Every tool call is appended to the audit log, and every tool that produces a trial
journals it before it returns, so stopping loses nothing that finished; only a request still running is abandoned. The server leaves without
the interpreter's usual shutdown on purpose: the MCP SDK reads input on a thread that
cannot be woken, and waiting for it is what used to hang a single Ctrl-C and abort
the interpreter on a repeated one.

## Research agent

`quantred research` runs the proposer loop and calls a model. It is not the same as
`quantred info`, which measures information diagnostics and calls no model
([information report](information)).

```bash
# needs the `ai` extra (or `openai`), or a local Ollama server
quantred research \
  --strategy demo_trend --symbols NVDA,AAPL,MSFT \
  --start 2024-01-01 --end 2025-12-31 \
  --goal "improve OOS Sharpe without raising max drawdown" \
  --holdout-days 60 --max-trials 10
```

It prints one line per round as it runs: `evaluated` with the round's median
out-of-sample Sharpe across folds (the measure the promotion gate applies, and the one
candidates are compared and ranked on), or `rejected` with the reason the journal also
records. A reply that is not a proposal (prose, `params` that is not an object, a text
value for a number, `10.5` for an integer parameter) is a rejected round, not the end of
the session; only a provider with nothing left to propose ends it early. A session whose
rounds were all rejected says nothing was evaluated, rather than "no candidate cleared
the gates".

Budgets, in plain units:

- `--max-trials` counts *evaluated* trials, proposals that reached a walk-forward. A
  rejected proposal does not spend one. `--max-dry-rounds` counts consecutive proposals
  that did not produce a better candidate, rejections included, so it is what ends a
  proposer that keeps being refused.
- `--max-tokens` is a total model-token budget for the whole session.
- `--holdout-days` is calendar days reserved at the end of the range; at least 1. A
  holdout that leaves no research window is refused.
- The trial count printed is this session's own. The campaign-wide multiple-testing
  count is a separate number.

Like every other command, `research` evaluates **net of costs** (`--commission-bps`,
`--impact-eta`, `--borrow-bps`) and says so at the start; `--gross` selects on gross
returns and is a diagnostic. The cost basis is part of each round's recorded identity
and of the saved config, so a gross round is never served for a net request. The holdout
is scored only when there is a shortlist to score; otherwise the output says none was.
If an earlier session in the journal already scored the same holdout on the same
universe, `research` warns: every look lets the next search fit to it.

With `--allow-code-gen`, a candidate the agent *wrote* is recorded with its source: on the
round's journal line (`code`, `code_hash`) and in the saved config's
`provenance.generated_code`. The config's strategy is the `draft:<Class>:<hash>` identity,
and loading it (`--config` on any command, `live` included) is refused: generated code is
never run by loading a config. Review the source and add it to a strategy pack
([private strategies](private-strategies)); a repeat of the same source over the same
window is a duplicate, like a repeated tune.

The loop proposes a hypothesis → validates it with walk-forward (never on the
holdout) → keeps it only if it's *promotable* and beats the incumbent
out-of-sample → repeats until its trial/token budget or a "no improvement" stop →
scores the survivors once on the sacred holdout → writes a shortlist of
provenance-stamped configs to `configs/` for you to review. Nothing is promoted to
live automatically.

## Choosing an LLM provider

The research agent is provider-agnostic — pick with `--provider`:

| Provider | Extra | Default model | Credential |
|----------|-------|---------------|------------|
| `anthropic` (default) | `ai` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| `openai` | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `ollama` (local) | none | `llama3.1` | none (local server) |

```bash
# Local model, no API key, no cloud calls:
quantred research --provider ollama --model llama3.1 ...

# OpenAI:
quantred research --provider openai --model gpt-4o ...
```

### Setting credentials

You can provide a key **either** in `.env` **or** as the standard environment
variable — they resolve through the same chain (`environment` → `.env`), so the
SDK defaults keep working and you don't have to duplicate keys.

`.env` (alongside your Alpaca keys — see `.env.example`):

```bash
ANTHROPIC_API_KEY=sk-ant-...                # for --provider anthropic
OPENAI_API_KEY=sk-...                       # for --provider openai
OLLAMA_BASE_URL=http://localhost:11434      # optional, for --provider ollama
```

…or the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export OLLAMA_BASE_URL=http://localhost:11434   # optional
```

Ollama needs no key — just run a model locally (`ollama run llama3.1`) and the
agent talks to its HTTP API at `OLLAMA_BASE_URL` (default `http://localhost:11434`).
Three settings tune it for a remote or slow machine:

| Setting | Default | Meaning |
|---------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Server root. A full `.../api/chat` URL is accepted and trimmed |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for one response; raise it for a slow GPU |
| `OLLAMA_NUM_CTX` | not sent | Context length to request; the model's own default applies when unset |

The per-response generation cap is the research loop's own token limit, not an
Ollama setting.

### When the provider says no

Run `quantred init --check --provider <name> --for research` to see what is missing
locally, and add `--probe-model` to make one real request. If a provider fails
*during* a session, `research` stops with the category (`auth`, `quota`, `rate_limited`,
`unavailable`, `timeout`, ...) and exits non-zero. A transient one (`rate_limited`,
`unavailable`, `unreachable`) is retried twice with a short wait first. An exhausted
balance, a bad key or a missing model is reported once and never sent again. Each retry
prints a line as it happens, and the final message states what actually occurred: a
`Retry-After` longer than the 30 seconds this tool will wait is reported as not retried.
A failure part-way through a session prints the session id, how many rounds were
journaled (they count toward the campaign), and that no holdout was scored; the session's
end is journaled with the stop reason.

`demo-agent` takes the same `--gross` switch and `--no-journal`, which journals the demo
to a scratch file so repeated runs do not add identical drafts to the real campaign.

See the engineering wiki's **MCP server** and **Research agent** pages for the
architecture and guardrails.

## From a source checkout (contributors)

Replace `quantred` with `uv run python main.py`, and install an extra with
`uv sync --inexact --extra mcp` (or `ai`, `openai`; plain `uv sync` removes extras and packs you do not name).
To register a checkout with an MCP client, give it the checkout's interpreter and the
absolute path to `main.py` - no `uv` on the client's `PATH` and no `cwd` needed.
`python main.py init` prints the block with your paths filled in; its shape is:

```json
{ "mcpServers": { "quantred": {
    "command": "/path/to/tradeflow/.venv/bin/python",
    "args": ["/path/to/tradeflow/main.py", "mcp"] } } }
```
