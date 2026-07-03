---
sidebar_position: 11
title: Continuous alphas
---

# Continuous alphas

`src/alphas/` turns a per-name signal into an **alpha**: a forecast of *residual
return* (return in excess of what beta to the benchmark explains), expressed in
the same units — annualized return — for every name. That makes views directly
comparable across symbols and directly consumable by a mean-variance
[portfolio optimizer](./portfolio.md).

A strategy's `generate_signals` returns a string (`BUY`/`SELL`/`HOLD`). That is the
right interface for the single-symbol [trade clock](./philosophy.md), but it throws
away **magnitude** ("I like NVDA twice as much as AAPL") and **comparability**
across names. Alphas recover both.

:::note Research clock only
Alphas are *forecasts*. Computing them never reads a realized forward return and
never places an order. `src/engine/live.py` is untouched — live trading still
consumes discrete signals via the existing path. See
[separation of concerns](./separation-of-concerns.md).
:::

## The refinement identity

The standard rule for turning a standardized score into a residual-return forecast:

```
α_i = σ_i · IC · z_i
```

cross-sectionally, **at each rebalance**:

- `z_i` — the standardized raw score of name *i*: `z_i = (s_i − mean(s)) / std(s)`,
  so scores have mean 0 and unit dispersion *across the universe*.
- `IC` — the information coefficient, the correlation skill of the signal
  (realistically 0.02–0.10). Supplied as a **prior** today; a future
  information-analysis step will measure it from realized outcomes and feed it back.
- `σ_i` — the annualized residual volatility of name *i* (return minus
  `β_i · benchmark_return`). The risk of the bet that *isn't* just market exposure.

The cross-sectional std of the resulting alphas is `≈ IC · σ`, so they carry the
right information ratio: an optimizer sizes positions by genuine conviction, not by
an arbitrary signal scale. Feeding un-scaled scores into a mean-variance optimizer
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
Two levers, both ending in `refine.neutralize` (an OLS residual, orthogonal to the
supplied exposures by construction, with an intercept so the residual is mean-zero):

- **Benchmark-beta neutral**: regress `z` on each name's beta and keep the residual.
  Enabled by `--neutralize`.
- **Factor neutral**: regress on the [risk model's](./risk-model.md) standardized
  factor exposures (`exp_<factor>` panel columns, written by
  `add_factor_exposure_features` from the *same* builder the factor risk model uses —
  one definition of "factor", both places). Enabled by `--neutralize-factors`; the
  bare flag neutralizes **market, volatility, size**. **Momentum is deliberately not
  in the default set** — a momentum tilt is a return bet the alpha may intend;
  regress it out explicitly (`--neutralize-factors market,volatility,size,momentum`)
  if that's the goal.

The z-score already centers the cross-section at 0, so equal-weight benchmark
neutrality holds even before the explicit step.

Both levers compose into **one regression on the union** of exposures, with three
rules that keep degraded inputs honest rather than silently wrong:

1. **Usability gating.** A factor column is used only if it exists and actually
   varies across covered names. An absent, all-NaN, or constant column (e.g. the
   exposure build qualified fewer than two names on a short-history universe)
   **degrades to plain-beta neutralization — never to no neutralization**.
2. **Mean-imputation for partial coverage.** A name missing one factor value gets
   the cross-sectional mean (0 — exposures are standardized), keeping it *in* the
   regression. Without this, the union's row-wise NaN-drop would strip that name's
   beta neutralization too, and the cross-section would silently mix neutralized and
   raw scores.
3. **Report what was applied, not what was asked.** The refinement records
   `panel.meta["neutralized_against"]` (and an imputation count); the services echo
   it as `neutralized_against`, and the CLI prints it — with an explicit warning when
   a requested factor's exposures were unavailable. "Factor-neutral" is never claimed
   for un-neutralized output.

:::note Known limitations (deliberate, tracked)
- The **MCP tools don't expose `neutralize_factors` yet** — results echo the field
  (always `[]` via that surface); wiring the parameter through
  [`src/mcp/server.py`](./mcp-server.md) is a small follow-up for when the agent
  surface needs it.
- The exposure builder's **history gate is two-way, not per-factor**: a subset with
  momentum requires the full 12-1 window (~148 bars), any other subset requires
  `vol_window + 1` (61) bars — even for factors (like market) that could tolerate
  less. Names between those bounds are dropped from the exposure frame (then
  mean-imputed per rule 2).
:::

## The feature panel and refinement

The refinement runs over a [`FeaturePanel`](./data-panel.md) — the cross-sectional,
point-in-time table that holds every name's features in one place. The flow is
`scan → panel → refine`:

1. A **scorer** fills the panel's `score` column (one per name). A scorer is just
   `Callable[[DataFrame], float]`:

   | Scorer | Score |
   |--------|-------|
   | `strategy_scorer` | the strategy's own continuous conviction (`calculate_scores`) — the natural, richest source |
   | `signal_scorer` | the strategy's discrete direction as `+1 / −1 / 0` (a lossy bucketing of the score) |
   | `scanner_scorer` | a scanner's `signal_strength`, signed by direction |

2. The **risk producer** fills `beta` and `residual_vol`.
3. **`refine_alpha(panel, context)`** reads `score` + `residual_vol` (+ `beta` when
   neutralizing) and writes the `z` and `alpha` columns. One implementation, one
   place — whatever produced the score flows through the same pipeline.

This is why each strategy is now **score-first** (`calculate_scores` is its one
decision function): the same number the trade clock derives its signal from is the
conviction the alpha layer scales. No parallel discrete-signal path.

## Hidden factors

- **IC is a prior.** The *absolute* scale of alphas is only as good as the assumed
  IC (default `0.03`). The *relative* sizing across names is correct regardless — IC
  is a common scalar, redundant with the optimizer's risk-aversion term. Flagged
  loudly in every result.
- **Cross-sectional, never time-series.** Standardize *across names at one
  timestamp*. Standardizing a name over time would use the full sample's mean/std —
  look-ahead.
- **Thin universes.** Below `min_universe` (default 10) names the z-score and
  winsorize quantiles are unstable, so the pipeline falls back to **demean-only** (no
  scaling by cross-sectional std) and sets `low_confidence`.
- **As-of discipline.** The bar [scan](./data-panel.md) is the single home of the
  leakage guard — it returns no bar after `as_of`, so every panel column (and the
  residual-vol window) is point-in-time. The alpha table is byte-identical whether or
  not later bars exist (a regression test asserts this).

## Where it runs

`services/analysis.py::compute_alphas` is the shared entry point: it scans the
universe as of `as_of`, assembles a [`FeaturePanel`](./data-panel.md) (risk + score
columns), refines it, and returns the ranked table. The CLI (`python main.py alphas`)
and the read-only MCP tool `compute_alphas` both route through it (one code path
across surfaces). `--source` picks the scorer: `strategy` (continuous conviction,
default), `signal` (discrete ±1), or `scanner` (scanner strength). See the
[usage guide](../usage/alphas.md).

## One source of truth

Each strategy was **migrated to be score-first**: `calculate_scores` is its one
decision function, and the trade clock's `BUY/SELL/HOLD` is derived from that score
by the base class (edge-triggered hysteresis — see [Strategies](./strategies.md)).
So `strategy_scorer` reads exactly the same conviction the live engine acts on;
there is no parallel discrete-signal path to drift. The order path stays
deterministic and model-free — improving *how* its signal is computed is not the
same as putting a model in the order path.
