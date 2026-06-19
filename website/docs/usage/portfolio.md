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
