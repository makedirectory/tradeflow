---
sidebar_position: 10
title: Portfolio manager
---

# Portfolio manager

`src/portfolio/allocator.py` decides position **weights** with a constraint
solver instead of ad-hoc rules. It is a small mixed-integer program solved with
[Google OR-Tools](https://developers.google.com/optimization) (`pywraplp`, CBC).

## The model

Inputs are `Candidate(symbol, score, price)` where a higher score is more
attractive (the factor is the caller's choice — trailing return, momentum,
inverse volatility, signal strength, a model's expected return, ...).

For each candidate the model has:

- a continuous **weight** `w_i ∈ [0, max_weight]`,
- a binary **selection** `x_i ∈ {0, 1}`.

**Maximise** the score-weighted allocation:

```
max  Σ score_i · w_i
```

subject to:

```
w_i ≤ max_weight · x_i          # zero weight unless selected
w_i ≥ min_weight · x_i          # floor a held name
Σ w_i ≤ 1                       # invest at most 100%
Σ x_i ≤ max_positions           # cardinality cap
```

The binary variables are what let a *linear* objective express "hold at most N
names" and "if held, at least this much" — the kind of combinatorial constraint a
plain weighting formula can't.

## Output

Each selected name becomes an `Allocation(symbol, weight, dollars, shares)`, with
`dollars = weight × capital` and `shares` floored to whole units (unless
fractional shares are allowed). Candidates with non-positive scores are dropped —
they can only consume a slot without improving the objective.

## Status & extension

This is a **preliminary** manager: long-only, single-period, score-driven. Natural
next steps — all expressible in the same MIP/QP framing — include a covariance
term for diversification (mean-variance), sector caps, turnover limits, or a
Kelly-style sizing objective. OR-Tools is an optional extra (`portfolio`),
imported lazily.
