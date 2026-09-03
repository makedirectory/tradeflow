---
sidebar_position: 9
title: Optimization (modeling)
---

# Optimization (modeling)

`tradeflow/optimization/` tunes a strategy's parameters by backtesting candidate
configurations and ranking them by an objective metric. Each evaluation is an
independent `BacktestEngine` run, so it is trivially parallelizable later; it runs
serially today for determinism and simplicity.

## `ParameterSpace`

Turns a `PARAM_RANGES` declaration into the things a search needs:

- `grid()` — the full step-aligned Cartesian product.
- `grid_size()` — the product's size **without materializing it**.
- `random_samples(n, rng)` — `n` random step-aligned configs.
- `to_unit_vector` / `from_unit_vector` — map a config to/from a `[0, 1]` vector,
  snapping back to the step grid (used by the surrogate model).

Only parameters that declare `min`/`max`/`step` are searched; the rest are held at
their defaults, so every candidate is a complete, valid config.

### Constraints between parameters

Some combinations are not merely bad, they are invalid: a fast moving average slower
than the slow one, an exit window longer than the entry window. A class declares those
beside its ranges, as `(left, operator, right)` triples where each side is a parameter
name or a literal:

```python
class DemoTrendStrategy(Strategy):
    PARAM_RANGES = {...}
    PARAM_CONSTRAINTS = (("fast_ema_period", "<", "slow_ema_period"),)
```

They are enforced **by construction, not by rejection**. `grid()` never contains an
invalid point, and `random_samples()` draws parameter by parameter with each one
restricted to the values still consistent with what it has already decided — so the
invalid region is unreachable rather than reached and discarded.

That distinction is the whole reason the feature exists. An invalid combination that
gets *evaluated* is a journaled trial, and a journaled trial permanently raises the
deflated-Sharpe bar for every future candidate in its family. Wasted trials are not
just wasted compute; they make the next real result harder to promote.

Two consequences worth knowing:

- `grid_size()` reports the number of points you will actually get, not the size of the
  unconstrained product — it is what a caller budgets `max_evals` against.
- A constraint that cannot exclude anything in the declared ranges is detected and
  skipped, so a declaration that changes nothing leaves a seeded search visiting
  exactly the configs it visited before.

The surrogate path (`from_unit_vector`) is the one place enforcement cannot be
structural: a proposal in a continuous box can snap outside the feasible region. Check
`is_valid()` and drop such a proposal before evaluating it — it still costs no trial.

Constraints are also enforced when a strategy is constructed, reading the same
declaration, so there is one definition of the rule rather than a sampler's copy and a
`initialize()` copy that can disagree.

## Screening, before optimizing

`services.analysis.run_screen` is a sweep that **journals nothing**. It builds a
`ParameterSpace` (narrowed per parameter if asked), prefetches the window once into a
`PrefetchedProvider`, and runs a `ParameterOptimizer` with `trial_store=None` — so no
point is recorded and none is served from recorded evidence.

Its report leads with the distribution and puts the best point last. The reason is
structural rather than stylistic: the best of N is the maximum of N draws, which is
positive under the null and grows with N, so a leaderboard without a null beside it
reproduces the exact error the deflated Sharpe corrects, one level up.
`analytics/screening.py` computes that null with the same `expected_max_sharpe` the
Deflated Sharpe uses, from the dispersion of the screened results — and refuses to
compute one at all for an objective whose null is not zero.

`confirm_screen_point` promotes exactly one point to a journaled trial by delegating to
`run_backtest`, so a confirmed point has the same dedup identity as the same backtest
run any other way. See [screening a parameter space](../usage/screening).

## `ParameterOptimizer`

Three methods, increasing in sophistication:

| Method | Idea |
|--------|------|
| `grid_search` | sweep the grid (sampling it when larger than the budget) |
| `random_search` | random step-aligned sampling |
| `optimize_bayesian` | train a Gaussian-Process **surrogate model** of the objective; propose the next config by Upper-Confidence-Bound acquisition |

### The surrogate model ("train a model to align params")

Bayesian optimization fits a Gaussian Process (scikit-learn, `Matern` kernel) to
the `(params → objective)` observations seen so far, then picks the next config
that maximizes `mean + exploration × std` over random candidates — balancing
exploitation and exploration. It re-fits after each evaluation. scikit-learn is an
optional extra, imported lazily with a clear error if missing.

## A bug worth calling out

A naïve `grid_search` materializes `space.grid()` before capping it. With ten
searchable parameters that is **billions** of combinations — enough to OOM-kill the
process. The fix: compute `grid_size()` first and, when it exceeds `max_evals`,
randomly sample the grid instead of building it. The
[test suite](testing) caught this.

:::tip Guard against overfitting
The best in-sample configuration is often not the best out-of-sample. Re-validate
the winner on a different window before trusting it.
:::
