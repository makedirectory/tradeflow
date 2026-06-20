---
sidebar_position: 6
title: Parameter optimization
---

# Parameter optimization

Optimization searches a strategy's parameter space for the configuration that
maximises a chosen backtest objective (Sharpe ratio, total return, ...). Each
candidate configuration is scored by running a full backtest.

A warning that bears repeating: optimization is *very* good at finding the
settings that would have printed money on this exact slice of history. That's not
the same as edge — it's often just a flattering fit to noise. Treat the result as
a hypothesis and put it through [walk-forward validation](walk-forward) before
believing a word of it.

```bash
make optimize
# or
uv run python main.py optimize \
    --strategy volume_spike --scanner none --symbols NVDA,META,TSLA \
    --start 2024-01-02 --end 2024-04-01 --method grid --max-evals 50
```

## Methods

| `--method` | What it does | Needs |
|-----------|--------------|-------|
| `grid` | Sweep the step grid; randomly samples it when larger than `--max-evals` | — |
| `random` | Random step-aligned sampling (`--max-evals` samples) | — |
| `bayesian` | Trains a Gaussian-Process **surrogate model** of the objective and proposes promising configs | `scikit-learn` |

Bayesian needs the optional extra:

```bash
make install-optimize     # or: uv sync --extra optimize
uv run python main.py optimize --method bayesian --scanner none --symbols NVDA,META
```

## Output

The best parameters and score are printed, and every evaluated configuration is
written to `optimization_results.csv`:

```
Best sharpe_ratio: 1.83
Best parameters: {'rsi_period': 10, 'volume_threshold': 1.4, ...}
Full results written to optimization_results.csv
```

:::tip Avoid overfitting
A configuration that looks great in-sample often disappoints out-of-sample.
Validate the winner on a *different* date range before trusting it.
:::

How the search avoids materialising astronomically large grids, and how the
surrogate model works, is covered in
**[Optimization (engineering)](../engineering/optimization)**.
