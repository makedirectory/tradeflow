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
uv tool install 'quantred[mcp,store,viz]'
quantred --version
```

[Installation](installation) has pipx, pip in a virtual environment, upgrading and
uninstalling. The `mcp` extra is what step 5 needs (the client launches the server from
this install), `store` is the local bar cache that the setup's optional cache warm and
every `--offline` run read, and `viz` draws charts in HTML reports; leave any of them off
if you will not use it. `--version` also prints the **state root**, the directory your
journal and `.env` will live in. See [optional extras](extras) for what each extra
unlocks and how to add one later without losing the rest.

## 2. See it work with no keys

```bash
quantred demo
```

`demo` runs the entire pipeline on synthetic data with no keys and no network, and
ends in an honest promotion verdict. You do not need `init` to try QuantRed.

## 3. `quantred init`

```bash
quantred init
```

A guided setup that writes a valid `.env` in your state directory (`~/.quantred`
unless `QUANTRED_HOME` is set), checks the credentials against Alpaca, and tells you
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
   asks for the key (hidden); for Ollama it asks for the endpoint, a response timeout,
   an optional context length and the model name to probe (validating each; the name is
   only used to print the probe command, since `research` takes `--model`). It never makes a request to the
   provider on its own, and if the provider's SDK extra is missing it prints the
   command that installs it, plus the exact `--probe-model` command (see below).
6. **Offers a first cache warm** (a small universe, about a year of daily bars) so
   your first `verdict` or `backtest` is fast and `--offline` works immediately.
7. **Prints the commands to try next**, including how an MCP client attaches.

## 4. Check what is ready: `init --check`

```bash
quantred init --check
```

```
QuantRed setup check

  state root: /home/you/.quantred
  (journal, trial store, bar cache, and configs all live here)

  [ok]       version                        quantred 2.6.0
  [optional] env file                       <state root>/.env not found
        -> run `init` to create it, or export the keys as environment variables
  [blocked]  APCA_API_KEY_ID                (not set) - no command that touches the market will run
        -> run `init` to enter it, or export it as an environment variable
  [blocked]  APCA_API_SECRET_KEY            (not set) - no command that touches the market will run
        -> run `init` to enter it, or export it as an environment variable
  [ok]       PAPER_TRADE                    true - orders go to the paper account
  [ok]       state directory                <state root>/logs is writable
  [ok]       state root                     /home/you/.quantred (QUANTRED_HOME)
  [optional] extra: viz                     not installed - charts in `--chart` and in HTML reports
  [optional] model provider                 no provider is set up for `research`

  ready for:  demo
  blocked for: data (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
  blocked for: research (APCA_API_KEY_ID, APCA_API_SECRET_KEY, model provider)

  Also not installed (optional): viz - what each unlocks and how to add it: https://quantred.mk-dir.com/usage/extras
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
| `[--]` | needs the network or was not requested (Ollama reachability). A configured Ollama endpoint is listed as *unverified* for `research`, not blocking, until `--probe-model` has run; after it the row shows the probe's result |
| `[FAIL]` | a broken state, such as a state directory you may not write to |

`--for {demo,data,research,mcp}` picks the task the exit code answers for (default
`data`), and scopes the install command: when an extra that task needs is absent (plus
the `--provider`'s SDK extra), the report ends with **one** command that adds exactly those
and keeps what you already have; the other absent extras are listed on one line, not
installed for you. A backtest is not blocked by a missing model key; a missing optional extra
never fails anything; a `[FAIL]` row fails every task. `--provider
{anthropic,openai,ollama}` checks that provider's SDK and key for `research`. `--offline`
checks the cache-only path instead: the `store` extra then gates `data` and `mcp`, and the
market-data keys gate only `research`:

```bash
quantred init --check --for research --provider ollama
quantred init --check --for mcp
quantred init --check --for mcp --offline
```

(`init --offline` arrived in 2.6.0; on that release it is refused with a pointer to
the commands that do take `--offline`.)

The "state directory" check is worth understanding: a session that cannot write its
journal silently loses its own trial history, which corrupts every multiple-testing
correction after it. Better to learn that before the first run.

### Is the model actually reachable? `--probe-model`

```bash
quantred init --probe-model --provider anthropic --for research
```

`--for research` makes the exit code answer for the research task (the market-data keys
the agent backtests with, plus a provider); without it the check defaults to `data`,
which ignores the provider. Either way the probe itself runs first and its result is
folded into the provider's row, so the table and the line beneath it agree. A failed
probe always exits non-zero. If the provider's SDK extra or key is missing the probe is
**not sent**: the output says so ("No request was sent"), nothing is billed, and the
exit code is still non-zero.

Makes **one** request of one token, with no retries, and names the outcome: `auth`,
`quota` (an empty balance), `rate_limited`, `unavailable` (the provider is overloaded
or failing), `entitlement`, `unreachable`, `timeout`, `model_not_found`. A hosted
provider may bill it, so it is never part of a plain `--check`, and it is not available
over MCP. `--model` probes a model other than the research default.

`research` itself retries a transient `rate_limited`, `unavailable` or `unreachable`
failure twice with a short wait; it never re-sends an empty balance, a bad key or a
missing model.

## 5. Connect an MCP client

```bash
claude mcp add quantred -- quantred mcp      # Claude Code
```

or, for any client that takes a JSON config:

```json
{ "mcpServers": { "quantred": { "command": "quantred", "args": ["mcp"] } } }
```

A desktop client often has a smaller `PATH` than your terminal, so a bare `quantred` may not be found. `quantred init` prints this same block with the **absolute** command for your copy, ready to paste, and `quantred mcp --self-test` runs exactly that command (and prints the block again when it passes), so what passes is what you configure.

Your client **launches the server itself**, from that command. Starting `quantred mcp`
in a terminal does not connect any chat; it only proves the server starts. Needs the
`mcp` extra. The server starts **without** market-data keys: the catalogue and trial-history
tools answer, and the tools that need bars need keys (`quantred init`) or a cache launched with
`quantred mcp --offline` (see [agents](agents)). To confirm discovery works without a chat client:

```bash
quantred mcp --self-test
quantred init --check --for mcp
```

The first lists the tools and calls the discovery tools through a real MCP client;
the second reports whether the extra is installed (keys are reported too, but the
server does not need them to start). For a keyless server over the cache, add
`--offline` to both: each then fails, naming the install command, when the `store`
extra the cache needs is missing. See
[agents](agents) for what the tools are.

## 6. Run a real example

With keys in place, `verdict` runs the whole cross-sectional pipeline once and ends
in one consolidated verdict:

```bash
quantred verdict --strategy demo_trend --scanner none \
    --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,JPM,XOM,UNH,COST,WMT \
    --start 2024-01-01 --end 2024-12-31
```

This starter is shaped to be feasible with its defaults: twelve liquid, well-known
names and no scanner, so the book can be fully invested under the 25% per-name cap
and there are enough names to measure rebalances. It is a usability example, not a
validated universe, and the verdict it prints says nothing about whether the signal
works. Three names at the same cap cannot fund a book, and the verdict then says
`infeasible` rather than asking for more data - see
[an infeasible book](verdict#an-infeasible-book).

Read the result top to bottom: the `VERDICT` line is the headline, the check lines under it
say which gates passed or failed, and the blocks above it say which stage limited the book. [One-command verdict](verdict) walks through a
full report line by line.

## 7. Only then, autonomous research

`quantred research` is the one command that calls a model. Set a provider up in
`quantred init` (step 3), probe it with `--probe-model`, and then run it ([agents](agents#research-agent)).
It is separate from `quantred info`, which measures information diagnostics and
calls no model. They are sibling commands: `quantred info --strategy ... --symbols ...`,
not `quantred research info`.

## `--non-interactive`

```bash
APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... quantred init --non-interactive
```

Builds the `.env` purely from environment variables, with no prompts and no retry
loop — for scripts, containers, and any shell where `getpass` has no TTY. The Alpaca keys
and `PAPER_TRADE` are copied when set.

**Model settings are copied only for the provider you name with `--provider`**, and only
that provider's variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OLLAMA_BASE_URL` /
`OLLAMA_TIMEOUT` / `OLLAMA_NUM_CTX`):

```bash
ANTHROPIC_API_KEY=... quantred init --non-interactive --provider anthropic
```

Without `--provider`, model variables in the environment are left alone and named in the
output as not copied. Those keys are often exported for unrelated tools, and a billable
key should not land in a project file because the shell happened to hold it.

`--provider`, `--for`, `--offline` and `--model` are refused where the mode would discard them (for
example `--provider` with the interactive wizard) rather than ignored.

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
- **Both files are owner-only (`0600`).** The file can hold API keys that bill; it is
  created private rather than tightened afterwards, and the backup is restricted too,
  whatever mode your original had.
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

## When something goes wrong

| You see | Why, and what to do |
|---|---|
| `command not found: quantred` | The installer's bin directory is not on `PATH`. With uv: `uv tool update-shell`, then a new terminal. In a venv: activate it first. |
| `zsh: no matches found: quantred[mcp]` | Quote the requirement: `'quantred[mcp,store,viz]'`. |
| `... needs the 'store' extra` (or `mcp`, `ai`, ...) | An optional capability is not installed. The message ends with the exact install command; `quantred init --check --for <task>` prints one command for everything that task needs. Run it as printed, since a bare `uv tool install --force 'quantred[x]'` would drop the other extras. uv cannot add a package to an installed tool in place. |
| `Missing Alpaca credentials` | Market data needs keys: `quantred init`. Or use `quantred demo` (no keys), or `--offline` against bars already in the cache. |
| Alpaca rejected the credentials | Re-copy from **Paper Account**, not the live dashboard; the two differ. "Could not reach Alpaca" is a network problem, not a key problem, and the keys are saved. |
| `PermissionError` creating `logs` | The state root is not writable (a sandboxed terminal, a read-only home). Point `QUANTRED_HOME` at a directory you can write. A session that cannot journal loses its own trial history, so this is refused rather than ignored. |
| Trials or history seem to be missing | Check `quantred --version` for the state root. A different `QUANTRED_HOME` (or none, in a client's environment) is a different journal. An MCP client needs the variable in its own config; `quantred mcp --self-test` prints it. |
| `quantred mcp` runs but the chat has no tools | Running it in a terminal connects nothing. The client launches the server itself from its config (step 5). Close the terminal copy, run `quantred mcp --self-test`, paste what it prints, restart the client. |
| `verdict` says `infeasible` | The universe is too small to fund the book under the per-name cap. Use more names or a higher `--max-weight`; see [an infeasible book](verdict#an-infeasible-book). |
| `research` fails on the first request | `quantred init --probe-model --provider <name> --for research` names the cause: `auth`, `quota`, `rate_limited`, `model_not_found` and so on. A saved key alone does not install the provider's SDK extra. |
| `init --check` shows `[FAIL]` | Only a broken state does (for example an unwritable directory). `[blocked]` and `[optional]` rows are not failures; the summary says which tasks are ready. |
