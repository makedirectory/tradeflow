---
sidebar_position: 2
title: First run (`init`)
---

# First run

```bash
make init          # or: python main.py init
```

A guided setup that writes a valid `.env`, checks the credentials against Alpaca,
and tells you what to run next. It exists because the alternative first run is:
clone, discover from a stack trace that you need a `.env`, create it by hand, learn
that a paper-trade flag exists, and only then find out whether the keys work.

You do **not** need this to try TradeFlow. `make demo` runs the entire pipeline on
synthetic data with no keys and no network at all.

## What it does

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
5. **Offers a first cache warm** (a small universe, about a year of daily bars) so
   your first `verdict` or `backtest` is fast and `--offline` works immediately.
6. **Prints the three commands to try next.**

## `--check` — the doctor

```bash
make check         # or: python main.py init --check
```

```
TradeFlow setup check

  [ok]   env file               /path/to/.env exists
  [ok]   APCA_API_KEY_ID        PKD2…B7Q3
  [FAIL] APCA_API_SECRET_KEY    (placeholder) — no command that touches the market will run
  [ok]   PAPER_TRADE            true — orders go to the paper account
  [ok]   state directory        logs is writable
  [FAIL] extra: viz             not installed — charts in --chart and in HTML reports: uv sync --extra viz
```

It is a doctor, not a linter: every check is evaluated and reported independently,
because the second and third problems are exactly what you want to know about. It
**writes nothing** and makes no network call, so it is safe to run any time.

Only the essentials fail the run (exit code 1) — a missing optional extra is
reported with the exact `uv sync --extra …` command and nothing more. The wizard
configures; you install.

The "state directory" check is worth understanding: a session that cannot write its
journal silently loses its own trial history, which corrupts every multiple-testing
correction after it. Better to learn that before the first run.

## `--non-interactive`

```bash
APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python main.py init --non-interactive
```

Builds the `.env` purely from environment variables, with no prompts and no retry
loop — for scripts, containers, and any shell where `getpass` has no TTY.

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
  `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` / `PAPER_TRADE` survive byte for byte.
- **A backup is written first**, to `.env.bak`, before any modification.
- **The write is atomic** — a temp file and a rename, so an interruption leaves
  either the old file or the new one, never a torn one that would read as
  half-configured credentials.
- **Only what you answered is written.** Filling in one missing key never rewrites
  the other.

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
