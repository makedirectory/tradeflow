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
(`python main.py live`) stays deterministic and LLM-free. Neither can place an
order — that capability is simply not wired in.

## MCP server

```bash
uv sync --extra mcp
uv run python main.py mcp     # serves over stdio
```

Register it with an MCP client:

```json
{ "mcpServers": { "tradeflow": {
    "command": "uv",
    "args": ["run", "--extra", "mcp", "python", "main.py", "mcp"],
    "cwd": "/path/to/tradeflow" } } }
```

The agent gets tools for discovery (`list_strategies`, `list_scanners`,
`get_param_ranges`), analysis (`run_scan`, `run_backtest`, `run_optimization`,
`run_walk_forward`, `summarize_bars`, `get_metrics_glossary`), and config
persistence (`save_config`, `load_config`, `list_configs`). Every call is logged
to `logs/mcp_audit.jsonl`. There is **no** order, live-trading, or
`PAPER_TRADE`-toggling tool — that absence is the safety model.

## Research agent

```bash
uv sync --extra ai
uv run python main.py research \
  --strategy volume_spike --symbols NVDA,AAPL,MSFT \
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
| `anthropic` (default) | `uv sync --extra ai` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| `openai` | `uv sync --extra openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `ollama` (local) | none | `llama3.1` | none (local server) |

```bash
# Local model, no API key, no cloud calls:
uv run python main.py research --provider ollama --model llama3.1 ...

# OpenAI:
uv run python main.py research --provider openai --model gpt-4o ...
```

### Setting credentials

You can provide a key **either** in `.env` **or** as the standard environment
variable — they resolve through the same chain (`environment` → `.env` → legacy
`config.py`), so the SDK defaults keep working and you don't have to duplicate keys.

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

See the engineering wiki's **MCP server** and **Research agent** pages for the
architecture and guardrails.
