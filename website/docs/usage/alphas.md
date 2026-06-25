---
sidebar_position: 8
title: Ranking by alpha
---

# Ranking by alpha

`python main.py alphas` ranks a universe by **continuous alpha** — a forecast of
each name's residual return (annualised, benchmark-relative), so the names are
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
cross-sectional standardised score, and `ALPHA = RESID_VOL · IC · Z` is the forecast
residual return for the year. The ranking is what a mean-variance
[portfolio optimiser](./portfolio.md) sizes positions from.

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
| `--lookback-days` | `180` | History fetched (≤ `as_of`) for the residual-vol estimate. |

Use `--source scanner` to rank by a scanner's continuous conviction instead of the
strategy's score:

```bash
python main.py alphas --source scanner --scanner volume --symbols NVDA,AAPL,META --as-of 2025-06-01
```

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
