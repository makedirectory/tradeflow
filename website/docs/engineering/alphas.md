---
sidebar_position: 11
title: Continuous alphas
---

# Continuous alphas

`src/alphas/` turns a per-name signal into an **alpha**: a forecast of *residual
return* (return in excess of what beta to the benchmark explains), expressed in
the same units — annualised return — for every name. That makes views directly
comparable across symbols and directly consumable by a mean-variance
[portfolio optimiser](./portfolio.md).

A strategy's `generate_signals` returns a string (`BUY`/`SELL`/`HOLD`). That is the
right interface for the single-symbol [trade clock](./philosophy.md), but it throws
away **magnitude** ("I like NVDA twice as much as AAPL") and **comparability**
across names. Alphas recover both.

:::note Research clock only
Alphas are *forecasts*. Computing them never reads a realised forward return and
never places an order. `src/engine/live.py` is untouched — live trading still
consumes discrete signals via the existing path. See
[separation of concerns](./separation-of-concerns.md).
:::

## The refinement identity

The standard rule for turning a standardised score into a residual-return forecast:

```
α_i = σ_i · IC · z_i
```

cross-sectionally, **at each rebalance**:

- `z_i` — the standardised raw score of name *i*: `z_i = (s_i − mean(s)) / std(s)`,
  so scores have mean 0 and unit dispersion *across the universe*.
- `IC` — the information coefficient, the correlation skill of the signal
  (realistically 0.02–0.10). Supplied as a **prior** today; a future
  information-analysis step will measure it from realised outcomes and feed it back.
- `σ_i` — the annualised residual volatility of name *i* (return minus
  `β_i · benchmark_return`). The risk of the bet that *isn't* just market exposure.

The cross-sectional std of the resulting alphas is `≈ IC · σ`, so they carry the
right information ratio: an optimiser sizes positions by genuine conviction, not by
an arbitrary signal scale. Feeding un-scaled scores into a mean-variance optimiser
is the classic way to get nonsense leverage on the noisiest names.

## The pipeline (`refine.py`)

Each step is a pure cross-sectional function on a symbol-indexed `Series`, so it is
unit-testable in isolation and composable. Order matters:

```
raw scores  s_i
  1. winsorize:  clip s_i to [Q(0.025), Q(0.975)]   # kill single-name outliers
  2. z-score:    z_i = (s_i − mean) / std            # cross-sectional, this rebalance
  3. neutralize: residual of z_i regressed on exposures   [optional]
  4. scale:      α_i = σ_i · IC · z_i                # the identity above
  5. cap:        |α_i| ≤ 3 · std(α)                  # final sanity bound
```

`AlphaModel.alphas(scores, context)` is the one method that runs this for every
model, so the scaling identity and the as-of discipline live in exactly one place.

### Neutralization

Alphas should be **benchmark-neutral** — the equal-weighted average alpha ≈ 0 — so
they express only *relative* views and don't smuggle in a directional market bet.
Two levers, both in `refine.neutralize` (an OLS residual, orthogonal to the supplied
exposures by construction, with an intercept so the residual is mean-zero):

- **Benchmark-beta neutral** (always available): regress `z` on each name's beta and
  keep the residual. Enabled by `--neutralize`.
- **Factor neutral** (when a factor risk model exists): regress on the
  factor-exposure matrix so the alpha is orthogonal to size/momentum/vol.

The z-score already centres the cross-section at 0, so equal-weight benchmark
neutrality holds even before the explicit step.

## Producing raw scores

Two `AlphaModel` subclasses differ only in *where the conviction comes from*; both
share the pipeline above.

| Model | Source | Raw score |
|-------|--------|-----------|
| `SignalAlphaModel` | a `Strategy`'s discrete signal | `BUY → +1`, `SELL → −1`, `HOLD`/exits → `0` |
| `ScoreAlphaModel` | any continuous metric (a `scorer` callable) | the metric itself; `scanner_scorer` signs a scanner's `signal_strength` by direction |

`SignalAlphaModel` is the bridge for direction-only strategies — it reads the
existing trade signal as a ±1 conviction without touching the strategy or the order
path. A genuinely continuous source (a scanner's strength, a momentum z) is better
served by `ScoreAlphaModel`.

## Hidden factors

- **IC is a prior.** The *absolute* scale of alphas is only as good as the assumed
  IC (default `0.03`). The *relative* sizing across names is correct regardless — IC
  is a common scalar, redundant with the optimiser's risk-aversion term. Flagged
  loudly in every result.
- **Cross-sectional, never time-series.** Standardise *across names at one
  timestamp*. Standardising a name over time would use the full sample's mean/std —
  look-ahead.
- **Thin universes.** Below `min_universe` (default 10) names the z-score and
  winsorize quantiles are unstable, so the pipeline falls back to **demean-only** (no
  scaling by cross-sectional std) and sets `low_confidence`.
- **As-of discipline.** `compute_alphas` slices every frame to bars `≤ as_of` before
  any computation, and residual-vol windows use only data `≤ as_of`. The alpha table
  is byte-identical whether or not later bars exist (a regression test asserts this).

## Where it runs

`services/analysis.py::compute_alphas` is the shared entry point: it slices to
`as_of`, computes residual vol via `calculate_beta` + `calculate_residual_volatility`,
builds the model, and returns the ranked table. The CLI (`python main.py alphas`) and
the read-only MCP tool `compute_alphas` both route through it (one code path across
surfaces). See the [usage guide](../usage/alphas.md).

## Status and scope

`SignalAlphaModel` reads the strategy's existing discrete signal rather than
rewriting each strategy to be score-first. A literal "derive `BUY`/`SELL`/`HOLD` from
the sign of a continuous score" rewrite was **deferred**: the current strategies are
edge-triggered (a `BUY` fires only on the crossing bar, not on every bar the trend
holds) and use a four-value vocabulary including `CLOSE_BUY`/`CLOSE_SELL`, neither of
which a scalar sign can reproduce without changing trade-clock behaviour. The
signal→score bridge delivers comparable, scaled alphas while keeping the order path
untouched.
