---
sidebar_position: 14
title: Multi-signal combination
---

# Multi-signal combination

One signal becomes one [alpha](./alphas). Real research has several — a trend read, a
volume read, a mean-reversion read — and they are **correlated**. `tradeflow/alphas/combine.py`
combines them into one alpha while avoiding the two classic mistakes:

1. **Naive weighting double-counts.** Weighting by raw IC over-weights redundant
   signals (three flavors of trend look like three bets but are one) and
   under-weights a weak-but-independent signal that adds the most.
2. **Estimated ICs are uncertain**, and that uncertainty should shrink a signal's
   contribution toward zero — more for short histories and low ICs.

## Optimal combination via the signal correlation matrix

Given each signal's information coefficient `IC` and the signal correlation matrix
`Ω`, the weights that maximize the combined IC are the GLS solution:

```
w = Ω⁻¹ · IC            combined IC = √(ICᵀ Ω⁻¹ IC)
```

Because `Ω⁻¹` accounts for correlation, two redundant signals **split** a weight
rather than each getting full credit. For two signals this is exactly the closed
form:

```
IC₁' = (IC₁ − ρ·IC₂)/(1 − ρ²)        IC_comb² = (IC₁² + IC₂² − 2ρ·IC₁·IC₂)/(1 − ρ²)
```

So adding a weak *independent* signal raises the combined IC; adding a strong
*redundant* one barely does. A small ridge regularizes `Ω⁻¹` so a near-duplicate
pair (`ρ → 1`) stays finite.

## Bayesian IC-uncertainty shrinkage

Each measured IC is shrunk toward zero by its estimation confidence before
combining:

```
IC' = IC · g/(g+1)        g = n · IC²
```

`IC' → 0` when the history `n` is short or the IC is small; `IC' → IC` as `n·IC² → ∞`.
This is what stops a noisy, short-history IC from being trusted at face value — a
primary reason backtested alphas disappoint live.

## Measured, not assumed

The ICs and `Ω` are **measured** over a trailing window, never assumed. At each of
several rebalance dates, every signal is scored on the cross-section (using only bars
`≤ t`) and correlated with the subsequent realized **residual** return (return minus
`β·benchmark` — so it rewards skill, not beta). The mean over rebalances is the IC;
the mean cross-sectional correlation between signals is `Ω`. Measuring on
out-of-sample data is what keeps the combination from over-fitting its own weights.

The combined score then flows through the **same** [`refine_alpha`](./alphas)
pipeline, scaled by the combined IC. So the [single-signal](./alphas) assumed IC scalar
is *replaced* by one measured, shrunk, redundancy-aware number — applied once, never
twice.

:::note Who owns the level shrink
The per-signal Bayesian shrink here and the
[IC-uncertainty level shrink](./alphas#forecast-refinement-v2) are the *same* `g/(g+1)`
math, so applying both would double-shrink and undertrade forever. The rule: **the
level shrink owns "is the IC real"; the combination owns "how correlated signals share
credit."** On this combined path the combination discharges the level; the level shrink
is *not* re-applied. The result echoes a `shrink_chain` so the single application is
auditable.
:::

## Where it runs

`services/analysis.py::compute_combined_alphas` measures the signals, combines the
current cross-section, refines it, and returns the ranked alphas plus the measured
ICs, shrunk ICs, GLS weights, and correlation matrix. The CLI
(`python main.py alphas --combine demo_trend,example_breakout,example_reversion`) and
the read-only MCP tool `combine_alphas` route through it. Combining needs at least two
strategies installed; a bare engine ships one, so the names above include the
[example pack's](../usage/private-strategies.md).

On the bundled synthetic data this is its own honesty check: two trend signals measure
as highly correlated (`ρ ≈ 0.9`) while a mean-reversion signal is the contrarian foil
(`ρ < 0`), and the shrinkage drives the (genuinely skill-less) random-walk ICs toward
zero — so the combined alpha is near-flat, as it should be.
