---
sidebar_position: 9
title: Estimating risk (Σ)
---

# Estimating risk (Σ)

`quantred risk` estimates the universe's covariance matrix **Σ** as of a date
and summarizes its risk structure. It is **read-only**: Σ sizes conviction for the
portfolio optimizer, it never places an order.

```bash
quantred risk \
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

*(Illustrative figures.)*

Read it as: **δ** is the constant-correlation shrinkage intensity (higher = the raw sample
was noisier, so more was pulled toward the structured target); **condition number**
near 1 is well-behaved and a large value means near-singular; **risk contribution**
is each name's share of the equal-weight portfolio's volatility (they sum to it).

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--symbols` | demo universe | Comma-separated universe. |
| `--as-of` | today | Estimation date; only returns up to it are used. |
| `--model` | `shrinkage` | `shrinkage` = sample covariance shrunk toward constant correlation; `sample` = raw covariance; `factor` = structural `XFXᵀ+Δ` (adds the factor-vs-specific risk split). |
| `--timeframe` | `1Day` | Bar timeframe for the return series. |
| `--lookback-days` | `365` | History fetched (≤ `as_of`) for the estimate. |
| `--benchmark` | `SPY` | Benchmark for the market factor and beta. |
| `--conditional` | off | `ewma` or `har`: condition the volatilities (see below). |
| `--conditional-lambda` | model default | Override the EWMA decay (RiskMetrics 0.94 daily, 0.97 weekly). |
| `--evaluate-conditional` | off | Run the evidence gate instead of the risk report. |
| `--config` / `--re-resolve-universe` | none | Replay a saved [run config](run-configs)'s universe and settings. |
| `--cache` / `--offline` / `--cache-dir` | off | Reuse a local bar cache; `--offline` never touches the network. |

## Why shrinkage

The raw sample covariance needs `N(N+1)/2` parameters and is often **non-invertible**
when you don't have far more observations than names — which breaks the portfolio
optimizer (it needs `Σ⁻¹`). Shrinkage pulls it toward a constant-correlation
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

## Conditional risk

`--conditional ewma` or `--conditional har` conditions Σ's **volatilities** on
recent history (holding the correlation structure slow), so a book targeting a
tracking error is measured against *current* risk instead of a flat
trailing-window average:

```bash
quantred risk --symbols NVDA,AAPL,META,AMD,TSLA --as-of 2025-06-01 --conditional ewma
```

```
Risk model 'shrinkage' as of 2025-06-01 (1Day returns)
  names 5  shrinkage δ 0.28
  condition number 9.1  PD True  mean corr 0.38  eq-weight vol 22.4%
  conditional (ewma, λ=0.94): mean σ_t/σ_unconditional = 1.31
```

`mean σ_t/σ_unconditional` is how stressed the book is right now relative to its
own trailing average — above 1 means recent vol is running hot.

**Default off.** Before turning it on, run the evidence gate that decides
whether conditioning is actually worth it for your universe:

```bash
quantred risk --symbols NVDA,AAPL,META,AMD,TSLA --evaluate-conditional
```

This compares EWMA/HAR against the unconditional trailing baseline on two
prongs — Mincer–Zarnowitz calibration and QLIKE loss — both required to point
the same way before the gate passes. `info --conditional-ab` runs the
complementary **net-of-cost** check: the same alpha book, walked forward,
conditional vs unconditional Σ, decided by realized net IR, not TE-tracking
alone (a Σ that tracks TE better but churns the book to death should lose
there). See
[Risk model — conditional risk (engineering)](../engineering/risk-model#conditional-risk)
for why the default is off on this repo's own demo data.

The same computation is available to agents as the read-only MCP tool `compute_risk`.
It takes `symbols`, `as_of`, `model`, `timeframe` and `lookback_days`; `--conditional`
is an evidence-gated knob and is not offered over MCP, and the benchmark is fixed.

## What to do next

Σ is the second input to [`allocate --objective utility`](portfolio). Compare
`--model sample` with the default to see how much shrinkage matters for your universe size.

## Common errors

- **`Missing Alpaca credentials`** - this command fetches bars, so it needs the paper key
  pair from `quantred init`. To work with no keys, warm a cache once and add `--offline`.
- **`--offline: no cached 1day bars for ... Run quantred cache warm ...`** - the cache has
  nothing for that range. The message prints the exact `cache warm` command to run (it
  needs the keys once, and the `store` extra, see [optional extras](extras)).
- **A name with a flat, median-looking volatility** - it had too little history and was
  kept with the fallback (median variance, independent) rather than dropped. Widen
  `--lookback-days` or drop the name.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, run them as
`uv run python main.py <command> ...`. The Makefile shortcut for this page is `make risk`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
