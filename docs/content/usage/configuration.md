---
sidebar_position: 2
title: Configuration
---

# Configuration

:::note Two different things called "config"

This page is the **strategy config dict** — one strategy's parameters and limits. The
saved **run config** file, which `--config` reads and `--save-config` writes, is a
separate artefact with its own page: [run configs](run-configs.md).

:::

:::tip No keys needed to explore
Want to see TradeFlow work before signing up for anything? Run `make demo` — it
runs the whole pipeline (backtest + walk-forward) on synthetic data, offline.
:::

Credentials live in a `.env` file, which is **gitignored** so your keys are never
committed. Create it from the template:

```bash
cp .env.example .env
```

```bash title=".env"
APCA_API_KEY_ID=<your key id>
APCA_API_SECRET_KEY=<your secret>

# Keep this true until you are absolutely sure you want to trade real money.
PAPER_TRADE=true
```

Get paper keys from the [Alpaca dashboard](https://app.alpaca.markets/) under
**Paper Account → API Keys**.

## Where settings come from

Every credential is resolved through one place (`tradeflow/settings.py`) in a fixed
order, so there's never any ambiguity:

1. **Environment variables** — the standard, 12-factor way (`export APCA_API_KEY_ID=…`).
2. **`.env`** in the project root — loaded automatically; real environment
   variables already set always win.

The same chain resolves the optional LLM provider keys for the research agent
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`) — set them in `.env`
or export them, whichever you prefer.

## Paper vs. live

`PAPER_TRADE` selects the Alpaca endpoint. Leave it `True` while learning — every
`live` run then trades against the paper account with no real money at risk.

## Strategy config keys

A strategy's config carries its tunable parameters plus a few behavioral keys the
base class recognizes:

| Key | Meaning |
|---|---|
| `risk_per_trade` | Fraction of capital risked per trade |
| `stop_loss` / `take_profit` | Fractional distances from entry price |
| `position_limits` | Portfolio limits the [engine](../engineering/engine.md) enforces across the book: `max_positions` (open positions), `max_position_size` (notional per position, in dollars), `max_total_risk` (risk budget as a fraction of equity), `max_gross_exposure` (long + short as a fraction of equity; unset by default), `max_net_exposure` (|long − short| as a fraction of equity; unset by default), `min_notional` (dollar floor below which an order would be refused by a venue; unset by default). See [what `max_total_risk` caps](#what-max_total_risk-caps) and [execution and cost](#execution-and-cost). |
| `reaffirm_entries` | Live only. Open a position the score implies even when its entry edge was missed — a rejected bar, a dropped stream, a restart, or a crossing inside the warm-up history. Default `true`, so a strategy started mid-trend takes the position rather than waiting for the next crossing. `--no-reaffirm-entries` on `live` turns it off. Exits are never gated by it. |

## What `max_total_risk` caps

`max_total_risk` is a **risk budget**, not an exposure cap, and the two read alike
enough to be worth stating plainly.

Each open position contributes `notional × stop_loss` — what it gives up if it stops
out — and the engine refuses an entry that would push the sum past
`max_total_risk × equity`. So it bounds *loss if everything stops out*. It says
nothing about how much notional is deployed, and the relationship between the two
runs through the stop distance:

| `stop_loss` | Notional admitted by `max_total_risk: 0.05` |
|---|---|
| 5% | 1× equity |
| 1% | 5× equity |
| 0.5% | 10× equity |

A tighter stop buys more notional for the same budget. Reading `max_total_risk: 0.05`
as "at most 5% of the book is deployed" is wrong in every row but the first.

`max_gross_exposure` is the cap on deployed notional: marked gross notional over
equity, shorts counted by magnitude. It is unset by default, which leaves free cash
as the only bound — and because [shorts are fully
cash-collateralized](../engineering/engine.md), that holds a backtest near 1× on its
own. Set it when a config should sit deliberately below that:

`max_net_exposure` is the cap gross cannot see. Gross bounds long **plus** short, so a
book sitting inside it can be entirely one-directional: $4,000 long and $4,000 short is
$8,000 gross and $0 net, while $8,000 long is the same gross and $8,000 net — and only
the second is a bet on direction. A long/short config bounded by gross alone is either
throttled or unhedged and never neither. It is judged on the *resulting* net, so an
entry that moves the book toward flat is admitted even from over the cap; refusing a
hedge for being a trade would leave the tilt it corrects in place.

**Derive it rather than pick it.** A backtest of a book that trades both sides prints
the tilt it actually carried — median, p90/p95/p99 and maximum as fractions of equity —
and, for each cap worth considering, how often that cap would have bound:

```
=== Directional tilt actually carried (400 steps) ===
  |net| / equity        median 19.2%  p90 29.5%  p95 32.1%  p99 37.1%  max 42.7%
  signed mean           +18.4% — the book leans long by construction
  gross max             80.0% of equity

  A cap and what it would have done:
    --max-net-exposure 0.32   would have bound on 5.0% of steps — a different book from the validated one
    --max-net-exposure 0.47   never binds — documents the intent, enforces nothing new

  Smallest cap that leaves the validated book intact: 0.47
```

A net cap at or above the gross cap can never bind: `|long − short| ≤ long + short`
identically, so gross already holds net below it. The derivation says so when it
happens, because a value the gross cap subsumes reads as a second limit when it is not
one.

The trade-off is the point, not the number. **Any cap below the observed maximum would
have changed the book that was validated** — it either never binds, in which case it
documents an intent rather than enforcing one, or it binds, in which case the thing
running is no longer the thing that was tested. A book that never carried a measurable
tilt is told so plainly, and a history too short for a percentile to mean anything says
that before it says anything else.

```python
"position_limits": {
    "max_positions": 5,
    "max_position_size": 1500.0,
    "max_total_risk": 0.05,      # lose at most 5% if every stop fills
    "max_gross_exposure": 0.60,  # never deploy more than 60% of equity
    "max_net_exposure": 0.20,    # never sit more than 20% net long or short
}
```

All of these are enforced across the whole book — by the backtest engine on the research
clock, and by the live trader on the trade clock (see [portfolio limits are enforced
live](live-trading.md#portfolio-limits-are-enforced-live), which notes where the two
counts differ). `Strategy.calculate_position_size` also clamps a *single* position
against `max_total_risk`, because sizing has no view of the open book; that clamp
asks only whether one position could exhaust the whole budget.

## Execution and cost

Sizes are floored to whole shares, on both clocks. That is invisible at $100,000 and
material at $4,000: the same config can lose a fraction of a percent of its intended
notional at one and a fifth of it at the other, and until recently nothing said so — the
equity curve was right while the reason for it was unexplained.

Every backtest now reports the gap when there is one:

```
--- Execution & cost (!) ---
Intended notional           $34,583.55
Filled notional             $32,721.56
  [PASS] rounding_drag         5.38% vs 10.00%
  [FAIL] unfillable_entries    8.51% vs 5.00%
  [PASS] cost_share_of_gross  12.40% vs 40.00%
  entries never opened      4 (4 rounded to zero, 0 below min notional)
```

Separate numbers, because they are separate problems. **Rounding drag** is the
intended notional that whole-share rounding removed — every name opens, slightly
smaller. **Unfillable entries** are positions that never opened at all, which is a
different book rather than a smaller one.

**`max_positions` decides what book you validated.** Every strategy shipped here
declares `max_positions: 1`, so a backtest over a scanned universe of sixty names
validates a book holding *one position at a time* — a correct result describing a
different strategy from the one most people think they are testing. The report flags a
cap of 1 whenever there is more than one candidate, because 1 is the default rather
than a decision. A deliberately concentrated book (five of sixty) is not flagged: the
check asks whether the cap was ever chosen, not whether it is small.

**Cost is measured against gross profit, not capital.** The same dollar cost is
unremarkable against a large gross return and fatal against a small one, and the two
denominators disagree most exactly when the answer matters. When a run made no gross
profit the check is skipped rather than reported: there was no edge for cost to eat, and
a ratio against a non-positive denominator would be arithmetic rather than a fact.

`min_notional` adds a venue's own floor: an order below it would be refused, so filling
it in a backtest validates a book that could not be traded. It is unset by default, so
a config keeps the behaviour it was validated under until you say otherwise.

**This verdict is deliberately separate from `promotable`.** The promotion gates ask
whether the edge was real and whether it was overfit; this asks whether the book can be
traded at this capital. They are different questions, and one number cannot answer both
without silently changing what it meant for every trial already recorded.

## Run-time options

Everything else (symbols, dates, capital, strategy, scanner) is passed per run on
the command line or via Make variables — nothing else needs editing:

```bash
make backtest SYMBOLS=AAPL,MSFT START=2024-06-01 END=2024-09-01 CAPITAL=50000
```

See the individual workflow pages for the full option list.
