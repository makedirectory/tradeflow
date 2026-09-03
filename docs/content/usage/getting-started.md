---
sidebar_position: 0
title: Getting started
---

# Getting started

:::note The shipped strategies are examples

`demo_trend` and the `demo_volume` scanner exist to demonstrate the interface and give
`tradeflow demo` something to run. Neither is an edge, and a bare install ships nothing
else. Your own work belongs in a private package — see
[your own strategies](private-strategies.md).

:::

From nothing to a real research result with Claude driving it, in six steps. Each
one works on its own, and you can stop at any of them.

You need a terminal. Steps 1–2 need nothing else — no keys, no account, no network.

---

## 1. Install it

```bash
uv tool install tradeflow-engine
```

No `uv`? [Install it first](https://docs.astral.sh/uv/getting-started/installation/),
or use `pipx install tradeflow-engine`. This gives you a `tradeflow` command.

```bash
tradeflow --version
```

That prints the version, which copy is running, and — worth noting now — **where its
state lives**. An installed copy keeps its research journal, trial history, and cache
in `~/.tradeflow`.

## 2. See it work, with no keys at all

```bash
tradeflow demo
```

This runs the entire pipeline on synthetic data: it backtests every bundled
strategy, picks the one that looks best, walk-forward validates it, and refuses to
promote it.

**The refusal is the point.** The synthetic series is a seeded random walk with no
edge in it, and a strategy that looks profitable in-sample gets called noise
out-of-sample. If that had *not* happened, the tool would be broken. Watching it
work on data you know is worthless is the fastest way to understand what it is for.

Nothing so far touched the network or needed an account.

---

## 3. Get keys and add them

Real market data needs free Alpaca paper-trading credentials:

1. Sign up at [alpaca.markets](https://app.alpaca.markets/).
2. Go to **Paper Account → API Keys** and generate a key and secret.

You want the **paper** keys. Paper and live credentials are different, and this tool
defaults to paper for a reason.

```bash
tradeflow init
```

The wizard prompts for both (hidden — nothing lands in your shell history), checks
them against Alpaca with one cheap request, confirms paper trading is on, and offers
to warm a small local data cache. It writes `~/.tradeflow/.env`.

Check it any time:

```bash
tradeflow init --check
```

That reports every setting independently — what is set, what is missing, whether
optional extras are installed — and writes nothing.

## 4. Your first real result

```bash
tradeflow verdict --symbols NVDA,AAPL,META,AMD,TSLA --start 2024-01-01 --end 2024-12-31
```

One command runs the whole cross-sectional pipeline — scan, alphas, portfolio
construction, information analysis — over one universe, one window, and one cost
model, and ends in a single verdict.

**Read the verdict line first, then the checks under it.** You will most likely get
`mixed` or `not promotable`, and that is the normal outcome. Every check shows its
value and its threshold, so you can see exactly which part failed:

| Check | What it means when it fails |
|---|---|
| `ic_tstat` | The signal's accuracy is not distinguishable from luck |
| `ir_above_noise` | The realized information ratio is inside its own error band — indistinguishable from zero |
| `sanity_ceiling` | A result *too* good on public data — suspect a bug or a data leak, not skill |
| `sample_size` | Too few rebalances to measure anything with confidence |
| `net_of_cost_alpha` | There may be an edge, but trading costs eat it |

Want it as a file you can keep or share?

```bash
tradeflow verdict --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html result.html
```

That writes one self-contained page — charts embedded, no external requests when
opened.

## 5. Connect Claude

```bash
uv tool install --force "tradeflow-engine[mcp]"
```

Then register the server with your MCP client.

**Claude Code:**

```bash
claude mcp add tradeflow -- tradeflow mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tradeflow": { "command": "tradeflow", "args": ["mcp"] }
  }
}
```

Restart the client and ask it something:

> *What has this campaign already tried, and what was the best result?*
>
> *Run a verdict on NVDA, AAPL and META for 2024 and tell me honestly whether there's
> an edge.*
>
> *Explain what the deflated Sharpe ratio is correcting for.*

**Claude cannot trade.** The server constructs only a market-data client — no broker,
no trading client — so placing an order is not a capability it has, rather than a
rule it has been told to follow. There is no order tool to prompt-inject around.
Promoting a strategy to live trading is a manual step you take yourself, outside the
agent entirely.

Worth knowing: everything Claude runs is journaled the same way your own commands
are, and counts toward the same multiple-testing total. An agent that runs fifty
backtests has genuinely made your next result harder to believe, and the tool will
say so.

## 6. Where to go next

- **[One-command verdict](verdict)** — the full flag set and how the gates work
- **[Backtesting](backtesting)** and **[walk-forward validation](walk-forward)** —
  "did this ever work", the honest way
- **[Browsing the trial store](trials)** — what you have already tried, and the
  leaderboard that will not lie to you about it
- **[The two clocks](../engineering/architecture)** — why research and trading are
  kept apart, which is the idea the whole design rests on

---

## A word about expectations

Most things you try will not work, and the tool is built to tell you that quickly
rather than slowly. That is the feature. A backtest that looks fantastic is the
normal result of trying many configurations, and nearly every guardrail here exists
to separate that from a real edge.

If a strategy survives walk-forward validation, clears the deflated Sharpe against
everything you have tried, and still has positive expected return net of trading
costs — then you have something worth a longer look.

:::warning
Educational software. Keep `PAPER_TRADE=true` unless you fully understand the
consequences. Trading carries real financial risk, and nothing here is investment
advice.
:::
