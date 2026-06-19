---
sidebar_position: 9
title: Optimization (modeling)
---

# Optimization (modeling)

`src/optimization/` tunes a strategy's parameters by backtesting candidate
configurations and ranking them by an objective metric. Each evaluation is an
independent `BacktestEngine` run, so it is trivially parallelisable later; it runs
serially today for determinism and simplicity.

## `ParameterSpace`

Turns a `PARAM_RANGES` declaration into the things a search needs:

- `grid()` — the full step-aligned Cartesian product.
- `grid_size()` — the product's size **without materialising it**.
- `random_samples(n, rng)` — `n` random step-aligned configs.
- `to_unit_vector` / `from_unit_vector` — map a config to/from a `[0, 1]` vector,
  snapping back to the step grid (used by the surrogate model).

Only parameters that declare `min`/`max`/`step` are searched; the rest are held at
their defaults, so every candidate is a complete, valid config.

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
that maximises `mean + exploration × std` over random candidates — balancing
exploitation and exploration. It re-fits after each evaluation. scikit-learn is an
optional extra, imported lazily with a clear error if missing.

## A bug worth calling out

A naïve `grid_search` materialises `space.grid()` before capping it. With ten
searchable parameters that is **billions** of combinations — enough to OOM-kill the
process. The fix: compute `grid_size()` first and, when it exceeds `max_evals`,
randomly sample the grid instead of building it. The
[test suite](testing) caught this.

:::tip Guard against overfitting
The best in-sample configuration is often not the best out-of-sample. Re-validate
the winner on a different window before trusting it.
:::
