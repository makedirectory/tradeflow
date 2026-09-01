---
sidebar_position: 5
title: Live (paper) trading
---

# Live (paper) trading

Live mode warms up the strategy with recent history, subscribes to the Alpaca
real-time bar stream, and routes signals to the broker as bracket orders.

```bash
make live
# or
uv run python main.py live --strategy volume_spike --scanner volume --symbols NVDA,META,TSLA
```

:::warning
With `PAPER_TRADE = True` (the default) this trades the **paper** account. Set it
to `False` only when you intend to trade real money.
:::

## What happens on start

1. The scanner picks the universe from your candidate symbols.
2. The engine fetches enough history to make every indicator valid, and seeds the
   strategy's rolling buffers (**warm-up**).
3. It subscribes to the **live bar stream** for every monitored symbol. The
   stream auto-reconnects with backoff if the socket drops.
4. Each streamed OHLCV bar updates the strategy, which emits a signal.
5. Actionable signals go to the `LiveTrader`, which sizes the position and submits
   a **bracket order** (entry + stop-loss + take-profit) through the broker.
6. In parallel, the **trade-update stream** logs fills/cancels/rejects so you can
   see what the account is actually doing.

Press `Ctrl-C` to stop.

### Order safety

- Entries are **skipped while an order is pending** for that symbol, so a repeated
  signal can't double-submit before the first fills.
- A discretionary close **cancels the resting bracket legs first**, so you're never
  left with an orphaned stop/take order.
- Orders are only sent during **market hours** (the clock is checked, with a short
  cache; disable with `respect_market_hours=False` if you need extended-hours).

## Position sizing

By default each entry is sized by the strategy's risk-per-trade / stop-loss
config (`RiskBasedSizer`). Add `--portfolio` to instead size positions by
**portfolio weights** computed with the OR-Tools allocator — capital is shared
across the universe rather than sized per trade:

```bash
make live-portfolio
# or
uv run python main.py live --scanner volume --symbols NVDA,META,TSLA \
    --portfolio --max-positions 5 --max-weight 0.25
```

With `--portfolio`, only the symbols the allocator funds are traded; if OR-Tools
isn't installed or nothing is funded, it falls back to risk-based sizing. See
[Portfolio allocation](portfolio).

Or use `--beta-sizing` to scale each position inversely by its **beta** vs a
benchmark (default `SPY`) — higher-beta names get smaller positions, evening out
risk:

```bash
make live-beta
# or
uv run python main.py live --scanner volume --symbols NVDA,META,TSLA --beta-sizing --benchmark SPY
```

## Managing the account

```bash
make cancel-orders      # cancel all open orders
make close-positions    # liquidate all positions (also cancels orders)
```

The full real-time path is described in **[The Engine](../engineering/engine)** and
**[Broker Abstraction](../engineering/broker-abstraction)**.

## Bar-quality guards

The live loop validates every bar before the strategy sees it. Guards are **on by
default** — the live path is the only place a corrupt bar costs money.

| Check | Rejects |
|---|---|
| OHLC consistency | `high < low`, open/close outside the range, non-positive prices, negative volume |
| Ordering | a timestamp at or before the last accepted bar for that symbol |
| Staleness | a bar arriving more than ~3 intervals late |
| Spike | a single-bar move beyond `--max-bar-return` (default 35%) |
| Zero volume | no volume on a symbol that has traded before |

**A guard rejects; it never repairs.** Nothing is interpolated, gap-filled, or
corrected. The moment the live path fixes its inputs it stops being the thing the
backtest validated, and every historical result quietly stops describing what will
happen. A rejected bar is skipped and logged with the offending values; the strategy
simply never sees it.

**The threshold is deliberately loose.** A 35% single-bar move is news, and the
strategy should act on it. The spike check exists to catch a decimal-point error or
a crossed quote, not a violent day — a guard tight enough to catch every bad tick
also removes the strategy's best opportunities.

At shutdown the loop reports what it discarded, and flags an elevated rejection rate
loudly. A guard quietly eating a third of the feed looks, from the strategy's side,
exactly like a quiet market.

```bash
python main.py live --symbols NVDA,AAPL             # guards on
python main.py live --symbols NVDA --max-bar-return 0.15   # stricter
python main.py live --symbols NVDA --no-bar-checks         # off (not recommended)
```

## Position reconciliation

Orders used to be submitted and forgotten, so a partial fill, a rejection, or a
position closed by hand in the broker's UI was discovered by reading the P&L and
being surprised.

The live loop now keeps an append-only ledger of **intent** (what was submitted) and
**observation** (what the broker reported), and sweeps it against the broker's actual
account state on a timer. Check it any time:

```bash
python main.py reconcile          # or: reconcile --json
```

```
RECONCILIATION FOUND 2 DIVERGENCE(S):
  [quantity_drift] NVDA: ledger expects +10, broker holds +4 — likely a partial fill
  [unexpected] TSLA: broker holds +7 that this ledger never ordered — opened manually
The broker's state is authoritative. Nothing has been corrected automatically.
```

Three rules govern it, and the first two are what keep it safe:

- **The broker is authoritative, always.** The ledger records what we believed so a
  difference can be *noticed*. When they disagree, the broker is right and the
  ledger is a question for a human.
- **It reports; it never remediates.** No corrective order is ever placed. An
  automated system that notices a missed fill and fixes it is one that can double a
  position at 3am while nobody is watching.
- **Append-only.** Entries are never edited or deleted, and the file *is* the state —
  a restarted process recovers its expectation by replaying it.

The sweep costs one `list_positions` call, never one per symbol, because it runs
inside the trade-clock loop. Exit code is non-zero when divergence is found, so a
scheduled `reconcile` can page you.

`--no-ledger` disables recording; `--reconcile-every 0` disables the in-loop sweep.

## Missed edges, and starting mid-trend

Signals are edge-triggered: an entry fires on the bar the score crosses and never
again. Live, that edge can be missed — a bar rejected by the quality guard, a dropped
stream, a restart, or a crossing that happened inside the warm-up history. The score
would still say "should be long" while every bar emitted `HOLD`, and the position was
simply never opened. The mirror case is worse: a missed exit leaves a real position
that nothing will close.

So the live loop compares the direction the score implies against the position book
(kept in sync with broker truth) and re-states the difference. Where an edge says
*change*, this says *what should be true now*.

**One consequence worth knowing before you run it.** If you start the engine while
the score already implies a position — a trend-follower started mid-trend, say — it
will open that position on the first live bar rather than waiting for the next fresh
crossing. Stops and targets are computed from the current price, not the price at the
original crossing. This is the default, on the view that a trend-follower started
mid-trend should hold the trend rather than sit flat until the next crossing.

If you would rather wait for a fresh edge:

```bash
tradeflow live --strategy ma_crossover --no-reaffirm-entries
```

or set `reaffirm_entries: false` in a strategy config. **Exits are never gated by
it.** Declining to open a position is a preference about what you trade; declining to
close one the strategy no longer wants is a stuck position, so a missed exit is always
re-stated whatever the flag says.

Backtests are unaffected either way: they derive the book from the same signals, so
the two can never disagree there.

## Why nothing happened

A signal that produces no order used to leave a log line and nothing else — and
"no order" is the same outcome for *the market is closed*, *we are halted*, *you
already hold this*, *the size rounded to zero*, and *the broker refused*. So the one
question worth asking afterwards was answerable only from logs, if they still
existed.

Execution now returns a **decision** for every signal, and the ledger records it:

```json
{"event": "decision", "symbol": "NVDA", "signal": "BUY", "allowed": false,
 "reason": "insufficient buying power: need $12400.00, have $9800.00",
 "guards_consulted": ["hold", "market_hours", "halt", "existing_position",
                      "pending_order", "account", "sizing", "buying_power"]}
```

`guards_consulted` lists the guards that actually ran, not only the one that
fired — a list naming just the veto cannot distinguish a guard that passed from one
that never ran, which is how a check silently stops being applied and nobody
notices. Declined decisions are recorded precisely because they leave no other
trace.

## Preflight: the contract before the order path

Every live run prints what it is about to do, before any order logic runs:

```
=== Live preflight ===
  broker mode           PAPER
  account               equity $100,000.00  cash $100,000.00
  capital this run      $8,000.00
  universe              61 symbols (replayed)
  max positions         8
  max position size     1200.0
  max gross exposure    0.9
  entries               re-affirmed
  bar guards            on
  reconcile every       300s
  ledger                ~/.tradeflow/logs/positions.jsonl
  journal               ~/.tradeflow/logs/research_journal.jsonl
  halt state            ~/.tradeflow/logs/halts.json
```

`--preflight` prints it and exits without starting anything. It is printed on every run
regardless, because a check you have to remember to ask for is one that gets skipped
exactly when it matters.

### `--capital` — what this run may deploy

The two most important lines above are adjacent on purpose: a paper account arrives
with whatever equity the venue handed out, and sizing against **that** trades a
different book from the one that was validated. It does not merely flatter the result —
it invalidates the execution telemetry, because fills, slippage and share rounding are
all properties of a book at a size.

`--capital` (or the `capital` a [saved config](walk-forward.md#reusing-a-saved-config)
carries) caps what the sizer may use. It is a ceiling, never a claim: an $8,000 config
on a $3,000 account deploys $3,000. Position limits expressed as fractions —
`max_total_risk`, `max_gross_exposure` — are fractions of *that capital*, not of the
account balance. Without it, sizing uses the whole account, which is the historical
behaviour.

### Real money must be said twice

`PAPER_TRADE` defaults to `true`, which is the right default and precisely why the
check exists: a default nobody set looks identical to a decision somebody made, right
up until it is wrong. With `PAPER_TRADE=false`, `live` **refuses to start** unless
`--live-money` also says so on the command line — two independent statements of the
same intent, because one of them can be inherited from a shell nobody remembers
exporting.

Paper runs are never asked to confirm anything. Making them would train the reflex the
guard depends on you not having.

## Portfolio limits are enforced live

The `position_limits` in a strategy's config — `max_positions`, `max_total_risk`,
`max_gross_exposure` — are checked against the whole book before every entry, under
the `position_limits` guard. **This is newer than the rest of live trading.** Sizing
clamps one position at a time and has no view of what is already open, so before
this guard existed each entry could consume the entire risk budget on its own and
nothing capped the count at all. A config validated in a backtest at five positions
could run unbounded live, against a margin account whose buying power is a multiple
of equity.

Two things to know before your next run:

- **The defaults now bite.** A strategy that declares no `position_limits` gets
  `max_positions: 5` — the same default the backtest has always applied. If you are
  running more names than that, set the limit to what you actually intend.
- **A portfolio-weight deployment must reconcile its two caps, and `live` refuses to
  start until it does.** `--portfolio` lets the [allocator](portfolio.md) choose the
  book, but the strategy's `position_limits.max_positions` still bounds what the book
  holds. When `--max-positions` exceeds it, the surplus names are funded and never
  traded, and which ones survive is decided by signal arrival order rather than by
  the allocation — so `live --portfolio` exits with both numbers and both remedies
  instead of starting. Every strategy shipped here declares `max_positions: 1` while
  `--max-positions` defaults to `5`, so the default invocation is one of the
  configurations that gets refused: raise the strategy's limit or pass
  `--max-positions 1`.

Every refusal is logged with the numbers that caused it and recorded as a decision,
so a limit that is quietly throttling a run is visible rather than inferred.

The live count differs from the backtest's in two ways, both deliberate. It reads
the strategy's position book — broker truth at start-up and at each reconciliation —
rather than querying the broker per entry, because several symbols can signal on one
bar and a round trip on each is exactly what the bar loop must not do. And it
measures exposure at entry price rather than marking it, because the trade clock has
no price for a symbol it is not currently handling. A book that has run up is
carrying more market exposure than this counts.

## Stopping

`Ctrl-C` stops the process, but it records nothing — restart the engine and it trades
again. To stop trading in a way that *sticks*, and to close everything in an
emergency, see [Stopping trading](./stopping.md).

```bash
tradeflow halts                                  # what is currently halted
tradeflow halt all --reason "why"                # refuse new entries
tradeflow flatten --confirm --reason "why"       # halt, cancel, close everything
```

Halts block entries and never block exits, so pulling the switch can never trap the
book.
