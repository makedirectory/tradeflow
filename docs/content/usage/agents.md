---
sidebar_position: 10
title: AI agents (MCP & research)
---

# AI agents (MCP & research)

TradeFlow has an **opt-in** automation layer that lets an AI help tune scanners
and strategies and do market analysis — without ever putting a model in the live
order path. Two entry points share the same deterministic core:

- **`mcp`** — serve TradeFlow's capabilities to any MCP client (Claude Code,
  Claude Desktop) as tools. A human (or another agent) drives.
- **`research`** — a bounded, self-pacing research loop that proposes
  hypotheses, validates them out-of-sample, and hands you a shortlist.

Both work on the **research clock** (offline, exploratory). The live order path
(`tradeflow live`) stays deterministic and LLM-free. Neither can place an
order — that capability is simply not wired in.

## MCP server

Needs the `mcp` extra ([optional extras](extras)) and your usual data credentials
([first run](first-run)). Your MCP client **launches the server itself** from its
config; starting `tradeflow mcp` in a terminal only proves it starts, and connects
no chat.

```bash
claude mcp add tradeflow -- tradeflow mcp      # Claude Code
```

Any other client takes the same command in its JSON config:

```json
{ "mcpServers": { "tradeflow": { "command": "tradeflow", "args": ["mcp"] } } }
```

A desktop client often has a smaller `PATH` than your terminal, so a bare `tradeflow`
may not be found. `tradeflow init`
prints this same block with the **absolute** command for your copy, ready to paste, and
`tradeflow mcp --self-test` runs exactly that command (and prints the block again when
it passes), so what passes is what you configure.

Run `tradeflow mcp` once in a terminal first. Without the extra or credentials it
exits immediately with the reason, which from inside a client looks only like a
server that never started.

To check that the server starts and answers, without a chat client, run
`tradeflow mcp --self-test`. It lists the tools and calls the discovery tools
through the real MCP client. `tradeflow init --check --for mcp` reports whether the
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

Every tool call appends to the journal before it returns, so stopping loses nothing
that finished; only a request still running is abandoned. The server leaves without
the interpreter's usual shutdown on purpose: the MCP SDK reads input on a thread that
cannot be woken, and waiting for it is what used to hang a single Ctrl-C and abort
the interpreter on a repeated one.

## Research agent

`tradeflow research` runs the proposer loop and calls a model. It is not the same as
`tradeflow info`, which measures information diagnostics and calls no model
([information report](information)).

```bash
# needs the `ai` extra (or `openai`), or a local Ollama server
tradeflow research \
  --strategy demo_trend --symbols NVDA,AAPL,MSFT \
  --start 2024-01-01 --end 2025-12-31 \
  --goal "improve OOS Sharpe without raising max drawdown" \
  --holdout-days 60 --max-trials 10
```

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
tradeflow research --provider ollama --model llama3.1 ...

# OpenAI:
tradeflow research --provider openai --model gpt-4o ...
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

Run `tradeflow init --check --provider <name> --for research` to see what is missing
locally, and add `--probe-model` to make one real request. If a provider fails
*during* a session, `research` stops with the category (`auth`, `quota`, `rate_limited`,
`unavailable`, `timeout`, ...) and exits non-zero. A transient one (`rate_limited`,
`unavailable`, `unreachable`) is retried twice with a short wait first. An exhausted
balance, a bad key or a missing model is reported once and never sent again, and the
session journal stays as it was.

See the engineering wiki's **MCP server** and **Research agent** pages for the
architecture and guardrails.

## From a source checkout (contributors)

Replace `tradeflow` with `uv run python main.py`, and install an extra with
`uv sync --inexact --extra mcp` (or `ai`, `openai`; plain `uv sync` removes extras and packs you do not name).
To register a checkout with an MCP client, give it the checkout's interpreter and the
absolute path to `main.py` - no `uv` on the client's `PATH` and no `cwd` needed.
`python main.py init` prints the block with your paths filled in; its shape is:

```json
{ "mcpServers": { "tradeflow": {
    "command": "/path/to/tradeflow/.venv/bin/python",
    "args": ["/path/to/tradeflow/main.py", "mcp"] } } }
```
