---
sidebar_position: 9
title: Estimating risk (Σ)
---

# Estimating risk (Σ)

`python main.py risk` estimates the universe's covariance matrix **Σ** as of a date
and summarises its risk structure. It is **read-only**: Σ sizes conviction for the
portfolio optimiser, it never places an order.

```bash
python main.py risk \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --as-of 2025-06-01 \
  --model shrinkage
```

```
Risk model 'shrinkage' as of 2025-06-01 (1Day returns)
  names 8  shrinkage δ 0.31
  condition number 12.4  PD True  mean corr 0.42  eq-weight vol 24.8%

SYMBOL          VOL   RISK CONTRIB
TSLA         44.1%          4.10%
NVDA         38.2%          3.55%
AMD          31.0%          2.90%
...
```

Read it as: **δ** is the Ledoit–Wolf shrinkage intensity (higher = the raw sample
was noisier, so more was pulled toward the structured target); **condition number**
near 1 is well-behaved and a large value means near-singular; **risk contribution**
is each name's share of the equal-weight portfolio's volatility (they sum to it).

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--symbols` | demo universe | Comma-separated universe. |
| `--as-of` | today | Estimation date; only returns up to it are used. |
| `--model` | `shrinkage` | `shrinkage` = Ledoit–Wolf; `sample` = raw covariance; `factor` = structural `XFXᵀ+Δ` (adds the factor-vs-specific risk split). |
| `--timeframe` | `1Day` | Bar timeframe for the return series. |
| `--lookback-days` | `365` | History fetched (≤ `as_of`) for the estimate. |

## Why shrinkage

The raw sample covariance needs `N(N+1)/2` parameters and is often **non-invertible**
when you don't have far more observations than names — which breaks the portfolio
optimiser (it needs `Σ⁻¹`). Ledoit–Wolf shrinks it toward a constant-correlation
target by an analytically optimal amount, guaranteeing an invertible, well-conditioned
Σ. Use `--model sample` to see the contrast (its condition number blows up as the
universe grows relative to the history).

## What the numbers mean (and don't)

- **Risk is not additive.** A high mean correlation means your names are closer to
  one bet than the count suggests — the whole reason to estimate Σ rather than sum
  variances.
- **As-of discipline.** Σ at `--as-of` uses only returns at or before it; adding
  later bars doesn't change the result.
- **Ragged universes.** Names with too little history are kept with a fallback
  (median variance, independent) rather than dropped silently.

The same computation is available to agents as the read-only MCP tool `compute_risk`.
