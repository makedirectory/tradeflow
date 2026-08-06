---
sidebar_position: 16.5
title: Performance attribution
---

# Performance attribution

[Information analysis](./information-analysis) checks the *predicted* IR against
the realized one, pooled over a window. `tradeflow/analytics/attribution.py` (+
`services.compute_attribution`) goes further: it attributes realized active
return **per period, per source** — systematic benchmark timing, each risk
factor, each signal, and stock-picking — by an exact regression identity, then
puts the same research-integrity guardrails on the attributed series that
[information analysis](./information-analysis) puts on ICs.

> A backtest tells you *what* happened. Attribution tells you *where* it came
> from — timing the benchmark, riding a factor, or genuine name selection — which
> is what tells you whether it repeats.

:::note Research clock only
A read-only diagnostic over realized returns; it places no orders and feeds
nothing back automatically.
:::

## The regression identity

At each sampled rebalance, active return splits into two stages, each an exact
regression identity so nothing is double-counted or lost:

1. **Systematic (known, predetermined) benchmark timing.** Strip
   `βᵢ(t)·r_B(t)` per name using the *same* canonical Σ-implied beta
   [risk model](./risk-model.md) already computes, so the portfolio aggregate is
   exactly `β_a(t)·r_B(t)` — not fit to this period's data, just the already-known
   tilt times the realized benchmark move. Decomposed further into
   expected / surprise / timing.
2. **Risk factors + signals, jointly.** One cross-sectional regression of the
   beta-adjusted return on `[risk_factor_exposures, signal_scores]` **together**.

The two blocks are fit *jointly*, not signals as a second pass on the risk-factor
residual. A second pass looks cleaner (each block seems to "own" its regression)
but leaves an **omitted-variable bias** that does not vanish with more names:
when active weights are built from a signal that isn't yet among the regressors
doing the fitting — exactly how a paper book is built upstream, from its own
alpha z-score — whichever risk factor happens to correlate with that signal *in
this period's finite sample* picks up a persistent, same-sign share of the
signal's true return. That's a genuine integrity bug (measured well past
significance in development, not sampling noise around zero), not a rounding
error; fitting both blocks together removes the omission and the bias
disappears. The real remaining tradeoff — signals correlated with risk factors
destabilize a joint fit — is smaller and disclosable (the condition number is
reported); the omitted-variable bias was not disclosable, since it looks exactly
like real factor timing.

Every component sums back to `r_active` exactly (up to floating point) —
`residual` in the output is adding-up slack only, and a test pins it near zero.

## Honest, per-row statistics

Every attributed row (systematic, each factor, each signal, specific) gets its
own **t-stat**, using a **Bayesian-blended risk**: short attributed series lean on
the risk model's structural prior instead of a wild few-point sample standard
deviation, blending toward the realized variance as the sample grows relative to
a prior weight `T₀` (derived from the risk model's own `min_obs`, converted to
this attribution's rebalance-period units — "the risk model itself wouldn't
trust an estimate from fewer periods than this, so neither should the
attribution").

A ranked table of ~8 attributed rows is exactly the same multiple-testing trap a
ranked IC table is — quoting the single best row without correcting for how many
you looked at overstates significance. The whole table is deflated by the same
`P(any |t| > 2 in n_trials)` inflation [information analysis](./information-analysis)
applies to ICs.

## Honest (non-compounding) cumulation

Chain-linking a per-period additive split across many periods is **not** the same
as compounding the true portfolio and benchmark returns — treating it as if it
were is the cumulation trap. Each component is chain-linked by the compounding
path up to (not including) its own period, which telescopes so the linked
components sum to `Π(1+r_active) − 1` exactly (`naive_cumulative`). The gap
between that and the honest `ΠR_P − ΠR_B` (`honest_car`) is `delta_cp`, reported —
not hidden — and flagged `cumulation_unreliable` when it's large relative to the
attributed terms (large per-period returns compound nonlinearly enough that the
additive story stops being a good local approximation).

## Optional: conditional risk, extra signals, bootstrap skill

- **`conditional`** threads an EWMA/HAR-conditioned Σ(t) into the per-period
  covariance this function already rebuilds at every sampled rebalance; when set,
  the report adds `te_by_regime` — predicted TE vs a realized-dispersion proxy,
  bucketed by the benchmark's own trailing realized-vol tercile. See
  [Risk model — conditional risk](./risk-model.md#conditional-risk). One Σ choice
  per call; the net-of-cost conditional-vs-unconditional comparison lives in
  `info --conditional-ab`.
- **`signals`** attributes extra strategies' combined scores as additional signal
  columns (the strategy's own alpha is always included) — so a
  [multi-signal combination](./multi-signal) weight can be checked against its
  realized counterpart.
- **`bootstrap_skill`** adds a nonparametric OWN p-value next to the parametric
  `SE{IR}≈1/√Y` verdict — see [Evaluation metrics](./evaluation-metrics#bootstrap-skill-inference).

## Example

```
Performance attribution: 'volume_spike' 2024-01-01..2024-12-31
  measured over 24 rebalances (horizon 5 bars)
  row                       mean/yr      IR       t  share ψ²
  active beta · expected          —       —       —  (not skill)
  active beta · surprise          —       —       —  (not skill)
  timing (δβ·δr)               0.12    0.08    0.31       0.4%
  market factor                -0.31   -0.22   -0.84       2.1%
  momentum factor               0.44    0.31    1.20       3.8%
  specific (stock-picking)      1.02    0.71    2.76      41.2%

  cumulative active return: +2.14% (top-down parts +2.09% + δ_CP +0.05%)
  Verdict: IR +0.68 ± 0.31 — distinguishable from luck (t +2.19)
```

Read "specific" as genuine stock-picking skill, factor rows as cheap tilts you
could have gotten from an ETF, and the systematic rows as the part explained by
already-known beta times the benchmark's realized move — not a bet, an
accounting fact.

## Where it runs

`services.compute_attribution` samples rebalances, rebuilds a leakage-safe
cross-section at each (alpha, risk-factor exposures, per-period Σ), and
attributes. There is no persisted weights/exposure history yet, so — like
`compute_information` — it recomputes per-period from stored bars rather than
consuming one. The CLI is `python main.py info --attribution`; see the
[usage guide](../usage/information.md#performance-attribution---attribution).
