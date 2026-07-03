---
sidebar_position: 8
title: Ranking by alpha
---

# Ranking by alpha

`python main.py alphas` ranks a universe by **continuous alpha** — a forecast of
each name's residual return (annualized, benchmark-relative), so the names are
directly comparable. It is **read-only**: it produces no orders and saves no config.

```bash
python main.py alphas \
  --strategy volume_spike \
  --symbols NVDA,AAPL,META,AMD,TSLA \
  --as-of 2025-06-01 \
  --ic 0.03
```

```
Alphas from 'volume_spike' score as of 2025-06-01 (IC=0.03, benchmark=SPY)

SYMBOL         SCORE        Z    BETA  RESID_VOL     ALPHA
NVDA           0.043     1.41    1.34      38.2%     1.62%
AMD            0.031     1.18    1.21      31.0%     1.10%
AAPL          -0.004    -0.35    1.05      22.4%    -0.24%
META          -0.018    -1.06    1.12      27.8%    -0.92%
TSLA          -0.040    -1.41    1.55      44.1%    -1.95%
```

Read it as: `SCORE` is the strategy's continuous conviction, `Z` is the
cross-sectional standardized score, and `ALPHA = RESID_VOL · IC · Z` is the forecast
residual return for the year. The ranking is what a mean-variance
[portfolio optimizer](./portfolio.md) sizes positions from.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--strategy` | `volume_spike` | Strategy whose score is the view (`--source strategy`/`signal`). |
| `--source` | `strategy` | `strategy` = the strategy's continuous conviction; `signal` = its `BUY`/`SELL`/`HOLD` as +1/−1/0; `scanner` = the scanner's continuous strength. |
| `--scanner` | `volume` | Scanner used as the metric when `--source scanner`. |
| `--symbols` | demo universe | Comma-separated candidates. |
| `--as-of` | today | Rebalance date; only data up to this date is used. |
| `--ic` | `0.03` | Assumed information coefficient (sets overall aggressiveness). |
| `--benchmark` | `SPY` | Benchmark for beta and residual volatility. |
| `--neutralize` | off | Make alphas beta-neutral (regress out benchmark beta). |
| `--neutralize-factors` | off | Make alphas **factor-neutral**: regress out the risk model's exposures. Bare flag = `market,volatility,size` (momentum kept — a momentum tilt is a return bet the alpha may intend); or pass a comma-separated subset. |
| `--lookback-days` | `180` | History fetched (≤ `as_of`) for the residual-vol estimate. |
| `--scaling` | `case1` | Per-name scaling: `case1` = `σ·IC·z`; `case2` = `IC·c_g·z` (no per-name vol multiply); `auto` = let a `Std_TS`-vs-`σ` regression decide, echoing the Case test. |

## Case scaling

`α = σ·IC·z` multiplies each name's score by its residual vol. That is right only if
the signal's per-name vol is *constant* across names (**Case 1**). For most
price-derived signals the vol is already *inside* the signal (**Case 2**), and the
`σ` multiply double-counts it — tilting the book into high-vol names.
`--scaling auto` runs a regression to decide and prints the verdict:

```bash
python main.py alphas --strategy mean_reversion --symbols ... --as-of 2025-06-01 --scaling auto
```

```
  scaling: case2 — Case 2 (R²=0.41, t=+3.8, candidate corr 0.62)
```

`candidate corr` is how different the two scalings actually are here (near 1 ⇒ the
choice barely matters). The ground-truth tiebreak is `info --scaling-ab`, which
walk-forwards the realized IR under both. See
[Continuous alphas](../engineering/alphas.md#forecast-refinement-v2).

Use `--source scanner` to rank by a scanner's continuous conviction instead of the
strategy's score:

```bash
python main.py alphas --source scanner --scanner volume --symbols NVDA,AAPL,META --as-of 2025-06-01
```

## Factor-neutral alphas

`--neutralize-factors` regresses the scores against the
[risk model's](../engineering/risk-model.md) standardized factor exposures, so the
alphas can't hide an incidental market/volatility/size tilt:

```bash
python main.py alphas --strategy volume_spike --symbols ... --neutralize-factors
```

The report states what was **actually applied** — and warns when it couldn't be:

```
Alphas from 'volume_spike' score as of 2025-06-01 (IC=0.03, benchmark=SPY)
  neutralized against: market, volatility, size
```

```
  neutralized against: beta
  ! requested factor(s) NOT neutralized (exposures unavailable — insufficient history?): market, volatility, size
```

The second form appears when the universe's history is too short to build the
exposures (momentum needs ~148 daily bars, the others 61) — the pipeline then falls
back to plain-beta neutralization rather than silently doing nothing. Unknown factor
names are rejected at parse time. The same flag exists on `allocate --objective
utility` (construct from factor-neutral alphas) and on `info`/`horizon` — **measure
the same alpha you deploy**. Note: the MCP tools don't expose this parameter yet.

## Combining several signals

`--combine` blends multiple strategies' signals into one alpha, weighting them by
their **measured** information coefficient and **correlation** so redundant signals
don't double-count and uncertain ones are shrunk toward zero:

```bash
python main.py alphas --combine volume_spike,ma_crossover,mean_reversion \
  --symbols NVDA,AAPL,META,AMD,TSLA --as-of 2025-06-01
```

```
Combined alpha from volume_spike, ma_crossover, mean_reversion as of 2025-06-01
  measured over 12 rebalances  |  combined IC 0.0412

SIGNAL                 IC   SHRUNK   WEIGHT
volume_spike       0.0380   0.0310    0.180
ma_crossover       0.0350   0.0280   -0.020   ← redundant with volume_spike, down-weighted
mean_reversion     0.0290   0.0210    0.240   ← independent, earns its weight
```

The ICs and the signal correlation are estimated over a trailing window of realized
residual returns — measure on out-of-sample data for an honest combination. See
[Multi-signal combination](../engineering/multi-signal.md) for the math.

## What the numbers mean (and don't)

- **The absolute scale is only as good as the assumed IC.** With no measured IC,
  treat the magnitudes as indicative; the *ranking* across names is robust regardless
  (IC is a common scalar). A future information-analysis step will measure IC and feed
  it back.
- **`low_confidence`** is flagged when the universe is thinner than ~10 names: the
  cross-sectional z-score is unstable, so the tool falls back to demean-only (no
  scaling) and says so.
- **No look-ahead.** Everything is computed from data at or before `--as-of`.

The same computation is available to agents as the read-only MCP tool
`compute_alphas` — see [Agents & MCP](./agents.md).
