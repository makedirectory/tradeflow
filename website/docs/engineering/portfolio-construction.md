---
sidebar_position: 15
title: Portfolio construction
---

# Portfolio construction

`src/portfolio/optimizer.py` turns alphas and risk into the portfolio that maximises
the **information ratio you can actually implement**. Where the OR-Tools
[allocator](./portfolio) maximises a scalar score subject to constraints — and so
piles weight onto the highest-scoring names regardless of how correlated they are —
the `MeanVarianceOptimizer` trades expected residual return against active risk:

```
maximise   U(w) = αᵀw − λ_A · wᵀΣw          (long-only absolute: benchmark w_B = 0)
subject to Σw = 1,  0 ≤ w ≤ max_weight,  ‖w‖₀ ≤ max_names
```

α is the [alpha](./alphas) vector (Spec 005), Σ the [covariance](./risk-model)
(Spec 006), and `λ_A` the aversion to active variance.

:::note Research clock — a proposal, not an order
This is portfolio *construction*: it proposes target weights (a config a human
promotes), it never places an order. It is deliberately separate from the operational
position [sizer](./portfolio) used by `live --portfolio`, so no covariance model
reaches the trade clock.
:::

## The closed form anchors everything

With no constraints or costs the optimum is closed-form, and it's worth stating
because every diagnostic is read against it:

```
w* = (1 / 2λ_A) · Σ⁻¹ α          IR* = √(αᵀ Σ⁻¹ α)
```

`IR*` is the **best achievable information ratio** — fully determined by `α` and `Σ`.
This is why Σ must be invertible (the reason for [shrinkage](./risk-model)) and why
scaling alphas correctly matters.

**You specify tracking error, not λ.** Users think in TE, so the optimizer inverts
the relation `λ_A = IR* / (2·ψ_target)` — pass `--target-te 0.04` and it solves for
`λ_A` such that the optimal tracking error is 4%.

## The transfer coefficient makes constraints visible

Constraints pull the implemented portfolio away from `w*`. The **transfer
coefficient** measures the damage, and the achievable IR degrades exactly as:

```
TC = corr(α, Σ-adjusted active weights) ∈ [0, 1]        IR_achieved = TC · IR*
```

So tightening the cardinality cap or the position limit lowers TC and the predicted
IR — a far more honest knob than "maximize score." A regression test pins this
monotonicity. The report also surfaces predicted TE, predicted IR, value added, and
turnover.

## How it solves (pure numpy)

- **Unconstrained**: the closed form above (used for `IR*`, `w*`, and the calibration
  identities).
- **Constrained**: `U` is a concave quadratic, so the long-only / box / budget optimum
  is a small convex QP solved by **projected gradient** with a capped-simplex
  projection — no heavyweight QP dependency.
- **Cardinality** (`‖w‖₀ ≤ k`): two-stage — solve, then keep the top-`k` names and
  re-solve on that subset.

## Edge cases it handles

- **Infeasibility.** `max_names · max_weight < 1` can't fund the book; the optimizer
  returns `feasible=False` and names the binding constraint rather than a silent empty
  portfolio.
- **Turnover from `w₀`.** Turnover is measured against current holdings (`w₀`, default
  cash), so the same target from where you already are costs nothing.
- **No-trade band.** A sub-threshold move keeps the current weights, suppressing churn.
  (A full cost-in-the-objective term arrives with the transaction-cost model.)

## Where it runs

`services/analysis.py::construct_portfolio` scans the universe, builds benchmark-neutral
alphas and Σ as of the date, optimises, and returns the proposed weights plus the
report. The CLI (`python main.py allocate --objective utility --target-te 0.04`) and the
read-only MCP tool `construct_portfolio` route through it. See the
[usage guide](../usage/portfolio.md).
