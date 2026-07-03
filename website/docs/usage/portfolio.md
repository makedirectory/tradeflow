---
sidebar_position: 7
title: Portfolio allocation
---

# Portfolio allocation

The portfolio manager decides **how much weight** to give each symbol, using a
constraint solver ([Google OR-Tools](https://developers.google.com/optimization))
rather than ad-hoc rules. It maximises total expected score subject to hard
constraints.

```bash
make allocate
# or
uv run python main.py allocate --scanner volume --symbols NVDA,META,TSLA,AMD \
    --capital 100000 --max-positions 5 --max-weight 0.25
```

Requires the optional extra:

```bash
make install-portfolio    # or: uv sync --extra portfolio
```

## What it solves

Each scanned symbol becomes a *candidate* with a **score** (here, its trailing
return — a transparent, swappable factor) and a price. The solver chooses weights
to **maximise the score-weighted allocation** subject to:

- invest at most 100% of capital,
- hold at most `--max-positions` names,
- cap any single name at `--max-weight`.

Example output:

```
SYMBOL     WEIGHT       DOLLARS    SHARES
NVDA        25.0%     25,000.00       190
META        25.0%     25,000.00        51
TSLA        20.0%     20,000.00        80
```

## Choosing the factor

The CLI scores by trailing return, but the allocator accepts any score. Swap in
momentum, inverse volatility, a model's expected return, or signal strength — see
**[Portfolio (engineering)](../engineering/portfolio)** for the model and how to
change the scoring factor.

## Mean-variance construction (`--objective utility`)

The default `allocate` is a scalar-score sizer. `--objective utility` instead builds
the **risk-adjusted** portfolio from a strategy's [alpha](alphas) and the
[covariance Σ](risk) — maximising `αᵀw − λ·wᵀΣw` at a target tracking error. It is a
**read-only research proposal** (it places no orders):

```bash
python main.py allocate --objective utility \
  --strategy volume_spike --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --as-of 2025-06-01 --target-te 0.04 --max-names 20
```

```
Portfolio for 'volume_spike' as of 2025-06-01 (target TE 4%)
  IR* 0.81  predicted TE 3.9%  predicted IR 0.74  transfer coef 0.91  turnover 100.0%

SYMBOL      WEIGHT       DOLLARS    SHARES
NVDA        25.0%     25,000.00       190
AMD         18.0%     18,000.00       142
...
```

Read it as: **IR\*** is the best information ratio achievable from these alphas and
this Σ; the **transfer coefficient** is how much of it survives your constraints
(tighten `--max-names` or `--max-weight` and watch it fall); **predicted IR** ≈
`TC · IR*`. See [Portfolio construction (engineering)](../engineering/portfolio-construction)
for the math.

`--neutralize-factors` builds the book from **factor-neutral alphas** (bare flag =
`market,volatility,size`; momentum kept as a deliberate return tilt) — see
[Ranking by alpha](alphas#factor-neutral-alphas) for the semantics and the
honesty warning when exposures are unavailable.
