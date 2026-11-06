---
sidebar_position: 2
title: First run (`init`)
---

# First run

This page takes you from nothing installed to a connected MCP client, in the order
that costs least: install, check, a keyless demo, then keys, then (optionally) an AI
client, and only then a model for the autonomous `research` agent. Each step stands
on its own; you can stop after any of them.

## 1. Install

```bash
uv tool install tradeflow-engine
tradeflow --version
```

In an activated virtual environment instead: `python -m pip install tradeflow-engine`.
The distribution is `tradeflow-engine`; the command is `tradeflow`. Capabilities such
as charts, the MCP server and the model SDKs are opt-in extras; see
[optional extras](extras) for which one unlocks what and how to add one later without
losing the rest.

## 2. See it work with no keys

```bash
tradeflow demo
```

`demo` runs the entire pipeline on synthetic data with no keys and no network, and
ends in an honest promotion verdict. You do not need `init` to try TradeFlow.

## 3. `tradeflow init`

```bash
tradeflow init
```

A guided setup that writes a valid `.env` in your state directory (`~/.tradeflow`
unless `TRADEFLOW_HOME` is set), checks the credentials against Alpaca, and tells you
what to run next. It exists because the alternative first run is: discover from a
stack trace that you need a `.env`, create it by hand, learn that a paper-trade flag
exists, and only then find out whether the keys work.

1. **Detects an existing `.env`** and shows what is in it, masked. If your keys are
   already set it asks before replacing them, and if they come from an exported
   environment variable it says so — you are not asked to fill in a file that would
   have no effect anyway.
2. **Prompts for the Alpaca key and secret**, hidden (`getpass`), so nothing lands
   in your terminal scrollback or shell history. Press Enter at both prompts to
   skip and stay in keyless demo mode.
3. **Validates the credentials** with one cheap historical-data request through the
   data-only client — never a trading client. Validating a key must not require the
   ability to trade with it.
4. **Confirms `PAPER_TRADE=true`.** Turning it off requires typing an exact phrase,
   not answering a yes/no prompt.
5. **Asks whether to set up a model provider for `research`**: `none` (the default),
   `anthropic`, `openai` or `ollama`. Skipping is fine; only the autonomous research
   agent uses a model, and you can redo this step any time. For a hosted provider it
   asks for the key (hidden); for Ollama it asks for the endpoint, a response timeout
   and an optional context length, validating each. It never makes a request to the
   provider on its own, and if the provider's SDK extra is missing it prints the
   command that installs it, plus the exact `--probe-model` command (see below).
6. **Offers a first cache warm** (a small universe, about a year of daily bars) so
   your first `verdict` or `backtest` is fast and `--offline` works immediately.
7. **Prints the commands to try next**, including how an MCP client attaches.

## 4. Check what is ready: `init --check`

```bash
tradeflow init --check
```

```
TradeFlow setup check

  state root: ~/.tradeflow
  (journal, trial store, bar cache, and configs all live here)

  [ok] version                tradeflow-engine 2.3.0
  [optional] env file         ~/.tradeflow/.env not found
        -> run `init` to create it, or export the keys as environment variables
  [blocked] APCA_API_KEY_ID        (not set) - no command that touches the market will run
        -> run `init` to enter it, or export it as an environment variable
  [blocked] APCA_API_SECRET_KEY    (not set) - no command that touches the market will run
        -> run `init` to enter it, or export it as an environment variable
  [ok] PAPER_TRADE            true - orders go to the paper account
  [ok] state directory        ~/.tradeflow/logs is writable
  [ok] extra: mcp             installed - the MCP server for AI clients
  [optional] extra: viz             not installed - charts in `--chart` and in HTML reports
        uv tool install --force 'tradeflow-engine[mcp,viz]==2.3.0'
  [--] model: ollama          endpoint http://localhost:11434 (reachability not checked; run with --probe-model)
  [blocked] model provider         no provider is set up for `research`

  ready for:  demo
  blocked for: data (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
  blocked for: research (APCA_API_KEY_ID, APCA_API_SECRET_KEY, model provider)
  blocked for: mcp (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
```

That is a trimmed example from a machine with no keys; your rows will differ (the
real output also lists every extra and each provider). It is a doctor, not a linter:
every check is evaluated and reported independently, because the second and third
problems are exactly what you want to know about. It **writes nothing** and makes no
network call, so it is safe to run any time.

Each row has one of five states, and the summary says what each *task* can do:

| Label | Meaning |
|-------|---------|
| `[ok]` | verified locally |
| `[blocked]` | a task needs it and it is missing; the row names the next action |
| `[optional]` | an extra or provider you have not asked for |
| `[--]` | needs the network or was not requested (Ollama reachability) |
| `[FAIL]` | a broken state, such as a state directory you may not write to |

`--for {demo,data,research,mcp}` picks the task the exit code answers for (default
`data`). A backtest is not blocked by a missing model key; a missing optional extra
never fails anything; a `[FAIL]` row fails every task. `--provider
{anthropic,openai,ollama}` checks that provider's SDK and key for `research`:

```bash
tradeflow init --check --for research --provider ollama
tradeflow init --check --for mcp
```

The "state directory" check is worth understanding: a session that cannot write its
journal silently loses its own trial history, which corrupts every multiple-testing
correction after it. Better to learn that before the first run.

### Is the model actually reachable? `--probe-model`

```bash
tradeflow init --probe-model --provider anthropic
```

Makes **one** request of one token, with no retries, and names the outcome: `auth`,
`quota` (the detail says whether it is exhausted credit or a rate limit),
`entitlement`, `unreachable`, `timeout`, `model_not_found`. It may be billed by the
provider, so it is never part of a plain `--check`, and it is not available over MCP.
`--model` probes a model other than the research default.

## 5. Connect an MCP client

```bash
claude mcp add tradeflow -- tradeflow mcp      # Claude Code
```

or, for any client that takes a JSON config:

```json
{ "mcpServers": { "tradeflow": { "command": "tradeflow", "args": ["mcp"] } } }
```

Your client **launches the server itself**, from that command. Starting `tradeflow mcp`
in a terminal does not connect any chat; it only proves the server starts. Needs the
`mcp` extra and market-data keys. To confirm discovery works without a chat client:

```bash
tradeflow mcp --self-test
tradeflow init --check --for mcp
```

The first lists the tools and calls the discovery tools through a real MCP client;
the second reports whether the extra and keys are in place. See
[agents](agents) for what the tools are.

## 6. Run a real example

With keys in place, `verdict` runs the whole cross-sectional pipeline once and ends
in one consolidated verdict:

```bash
tradeflow verdict --strategy demo_trend --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
    --start 2024-01-01 --end 2024-12-31
```

Read the result top to bottom: the last line is the verdict, and the block above it
says which stage limited the book. [One-command verdict](verdict) walks through a
full report line by line.

## 7. Only then, autonomous research

`tradeflow research` is the one command that calls a model. Set a provider up in
`tradeflow init` (step 3), probe it with `--probe-model`, and then run it ([agents](agents#research-agent)).
It is separate from `tradeflow info`, which measures information diagnostics and
calls no model.

## From a source checkout (contributors)

Everything above works from a checkout too, with `uv run python main.py` in place of
`tradeflow` (`make init`, `make check`, `make demo` are the Makefile shortcuts), and
`uv sync --extra <name>` in place of an installer command. A checkout does not keep
its own state: state lives in `~/.tradeflow` unless `TRADEFLOW_HOME` is set, and
`tradeflow init --dev-local-state` prints the setting that puts it back inside the
working tree.

## `--non-interactive`

```bash
APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... tradeflow init --non-interactive
```

Builds the `.env` purely from environment variables, with no prompts and no retry
loop — for scripts, containers, and any shell where `getpass` has no TTY. The model
settings are read the same way when set: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`, `OLLAMA_NUM_CTX`.

## What it will not do

- **Create an Alpaca account.** It links to where keys come from; it drives no
  browser and stores nothing beyond `.env`.
- **Manage settings generally.** `settings.py` stays the single reader of the
  `.env` contract; this is not a config system.
- **Recommend live trading.** It can preserve an existing `PAPER_TRADE=false` after
  an explicit typed confirmation, but the guided path never suggests it.

## Your existing `.env` is safe

A dotenv file is shared convention and may hold keys this project does not own.
Rewrites are careful about that:

- **Nothing else changes.** Comments, blank lines, ordering, and every key outside
  `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` / `PAPER_TRADE` and the model settings
  above survive byte for byte unless you answer them.
- **A backup is written first**, to `.env.bak`, before any modification.
- **The write is atomic** — a temp file and a rename, so an interruption leaves
  either the old file or the new one, never a torn one that would read as
  half-configured credentials.
- **Only what you answered is written.** Filling in one missing key never rewrites
  the other, and pressing Enter at the model-provider prompt never blanks a key you
  already have.

## Secrets hygiene

No credential you type is ever echoed, logged, journaled, or included in an error
message. Everything printed back is masked (`PKD2…B7Q3`), and short values are
masked entirely — revealing most of a short string reveals the string.

Credential validation distinguishes three outcomes, and says which one happened:

| Outcome | Means |
|---|---|
| Credentials work | Alpaca returned market data |
| Alpaca rejected these credentials | Check you copied from the **Paper Account** section — paper and live keys differ |
| Could not reach Alpaca | A network or rate-limit problem, **not** a credential problem — the keys are saved |

That distinction matters more than it looks: sending someone to regenerate keys
that were never the problem is a worse failure than saying nothing.
