---
sidebar_position: 13
title: Risk model
---

# Risk model

`src/risk/` estimates the covariance matrix **Σ** over a universe — the thing
portfolio theory is actually about. Risk is **not additive**: two names you like
that move together are one bet, not two; two that move oppositely are a hedge.
Without Σ the [allocator](./portfolio) is blind to that — it can't compute tracking
error or find the genuinely diversified portfolio. Σ is the *denominator* the alpha
forecast gets divided by.

:::note Research clock only
Σ is a tool for sizing conviction, never consulted to place an order. `src/risk/`
imports no broker and the order path imports no risk model.
:::

## What it produces

`build_risk_matrix(model, bars, periods_per_year)` returns a `RiskMatrix` — an
**annualised**, positive-definite Σ over the universe plus the quantities built on
it:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| variance | `wᵀ Σ w` | portfolio variance |
| tracking error | `√(w_aᵀ Σ w_a)`, `w_a = w − w_B` | active risk vs a benchmark |
| MCR | `(Σ w_a) / ψ` | each name's marginal contribution to risk |
| condition number | `cond(Σ)` | how close to singular (the invertibility guard) |

Annualised so it shares a time unit with the [alphas](./alphas) (`α = σ·IC·z`) and
the optimiser's risk-aversion term.

## Estimators

Two estimators behind one `RiskModel.estimate(returns)` interface:

- **`shrinkage` (Ledoit–Wolf, default).** The raw sample covariance `S` over `N`
  names from `T` observations needs `N(N+1)/2` parameters; when `T` is not far
  greater than `N` it is noisy and often **non-invertible** — fatal, since the
  optimiser needs `Σ⁻¹`. Ledoit–Wolf shrinks `S` toward a structured target by the
  analytically optimal intensity `δ`:

  ```
  Σ̂ = δ·F + (1 − δ)·S
  ```

  `F` is the **constant-correlation** target (every pairwise correlation = the
  average sample correlation, variances kept). `δ ∈ [0, 1]` is larger when `S` is
  noisier (small `T`, large `N`) and `→ 0` as the sample becomes reliable. This
  guarantees a well-conditioned, invertible Σ̂ with no factor data at all. The closed
  form is vendored in pure numpy — no sklearn, no compiled dependency.

- **`sample`.** The raw sample covariance. Honest, but ill-conditioned when `T ≲ N`
  — there mostly to show *why* shrinkage exists.

- **`factor` (structural).** `Σ = X F Xᵀ + Δ` — a small factor model. Exposures `X`
  load each name on **market** (beta), **momentum** (12-1 trailing return),
  **volatility**, and **size** (log dollar volume), cross-sectionally standardized.
  Factor returns are recovered each period by a cross-sectional regression of returns
  on `X`; `F` is their covariance and `Δ` the diagonal residual variance. The parameter
  count drops from `O(N²)` to `O(N·K)`, Σ is invertible by construction, and — the
  payoff — risk becomes **attributable**: portfolio variance splits cleanly into
  *factor* risk (`w_aᵀ X F Xᵀ w_a`) and *specific* risk (`w_aᵀ Δ w_a`). `risk --model
  factor` reports that split; it's what makes performance attribution possible.

## Edge cases it handles

- **Invertibility is the whole point.** Every estimator returns a positive-definite
  Σ; a regression test inverts it and checks the condition number even when `T < N`.
- **Annualisation.** Bar-level covariance × periods/year, on the same footing as the
  alphas.
- **As-of / no look-ahead.** Σ at `t` is built from the [scan](./data-panel)'s
  returns `≤ t`; a test asserts it's unchanged whether later bars exist.
- **Ragged panel.** Names enter and leave the universe. Σ is estimated on names with
  at least `min_obs` aligned returns; names below the floor are kept with a
  cross-sectional **median variance** and zero correlation (a documented fallback,
  not a silent drop), so Σ spans the full universe and stays PD.

## Where it runs

`services/analysis.py::compute_risk` scans returns as of a date, estimates Σ, and
returns a summary — shrinkage `δ`, condition number, mean correlation, equal-weight
portfolio volatility, and the top risk contributors. The CLI (`python main.py risk`)
and the read-only MCP tool `compute_risk` route through it. See the
[usage guide](../usage/risk.md).
