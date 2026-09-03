---
sidebar_position: 7
title: Portfolio allocation
---

# Portfolio allocation

The portfolio manager decides **how much weight** to give each symbol, using a
constraint solver ([Google OR-Tools](https://developers.google.com/optimization))
rather than ad-hoc rules. It maximizes total expected score subject to hard
constraints.

```bash
make allocate
# or
uv run python main.py allocate --scanner demo_volume --symbols NVDA,META,TSLA,AMD \
    --capital 100000 --max-positions 5 --max-weight 0.25
```

Requires the optional extra:

```bash
make install-portfolio    # or: uv sync --extra portfolio
```

## What it solves

Each scanned symbol becomes a *candidate* with a **score** (here, its trailing
return — a transparent, swappable factor) and a price. The solver chooses weights
to **maximize the score-weighted allocation** subject to:

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
the **risk-adjusted, cost-aware** portfolio from a strategy's [alpha](alphas) and the
[covariance Σ](risk) — maximizing `αᵀw − λ·wᵀΣw − cost(Δw)` at a target tracking
error. It is a **read-only research proposal** (it places no orders):

```bash
python main.py allocate --objective utility \
  --strategy demo_trend --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --as-of 2025-06-01 --target-te 0.04 --max-names 20 --capital 1000000
```

```
Portfolio for 'demo_trend' as of 2025-06-01 (target TE 4%, cost-aware)
  IR* 0.81  predicted TE 3.9%  predicted IR 0.74  transfer coef 0.91  turnover 18.0%
  net active return 3.62%/yr (round-trip)  = gross 3.85% − round-trip cost 0.23%
    this rebalance: turnover cost 0.11%/yr one-way (linear 0.08% + √-impact 0.03%); one-way net 3.74%
  capacity ≈ $340,000,000 (where √-impact erases the alpha)

SYMBOL      WEIGHT       DOLLARS    SHARES
NVDA        25.0%     250,000.00       190
AMD         18.0%     180,000.00       142
...
```

Read it as: **IR\*** is the best information ratio achievable from these alphas and
this Σ, before cost; the **transfer coefficient** is how much of it survives your
constraints *and* cost (tighten `--max-names`/`--max-weight`, or trade a costlier
book, and watch it fall); **predicted IR** ≈ `TC · IR*`. `net active return` is
the headline, net of a round-trip cost haircut on the held book; the one-way
line below it is the detail — exactly what this rebalance's turnover cost, split
into linear (commission + spread) and square-root impact. See
[Portfolio construction (engineering)](../engineering/portfolio-construction) for
the math.

**Cost-aware is the default** — with `--capital` set, the objective carries a
name-specific linear turnover cost and a square-root market-impact term, and a
no-trade band *emerges* from each name's own cost (cheap, liquid names trade
freely; expensive ones need a clearer signal to move). Pass `--gross-objective`
to drop cost from the objective (it's still reported ex-post) — useful for
seeing how much the cost-aware solve actually bought you. `--holding-period`
(years, default `1/12`) sets the amortization horizon for the in-objective cost.

`--neutralize-factors` builds the book from **factor-neutral alphas** (bare flag =
`market,volatility,size`; momentum kept as a deliberate return tilt) — see
[Ranking by alpha](alphas#factor-neutral-alphas) for the semantics and the
honesty warning when exposures are unavailable.

## Benchmark-relative construction

`--benchmark-holdings equal` (or a `symbol,weight` CSV/JSON file) makes the
benchmark a genuine **portfolio**, not just a beta/vol return series. Tracking
error, alpha neutralization, and the transfer coefficient all move into **active
space** (`w_a = w − w_B`):

```bash
python main.py allocate --objective utility --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN --as-of 2025-06-01 \
  --benchmark-holdings equal --benchmark-premium 0.05
```

The report adds `active beta`, `residual risk` (the `ψ² = β_a²σ_B² + ω²` split),
and a **consensus-returns** block — the reverse-optimized expected returns for
which `w_B` is itself mean-variance optimal, so you can see what your alphas are
really deviations *from*. Without `--benchmark-holdings` everything stays
cash-relative, byte-for-byte unchanged. See
[Portfolio construction — benchmark-relative](../engineering/portfolio-construction#benchmark-relative-construction---benchmark-holdings).

## Long/short (`--book market-neutral`)

The long-only box forces every unattractive name to a *forced underweight* it
can't relax below zero. `--book market-neutral` relaxes the box to
`[−short-max-weight, max-weight]` and the budget to `Σw = 0`; a gross-leverage
cap is then **mandatory** (an unconstrained long/short book on a noisy Σ is a
leverage machine):

```bash
python main.py allocate --objective utility --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN --as-of 2025-06-01 \
  --book market-neutral --gross-leverage 2.0 --short-max-weight 0.25
```

The report adds the dollar-neutral residual, realized gross leverage vs the
cap, and the short book's borrow carry. `--longshort-report` solves the *same*
alphas/Σ/cost both ways and prices the long-only constraint directly: the
measured IR shrinkage, both transfer coefficients, and the long-only book's
incidental size tilt. See
[Portfolio construction — long/short](../engineering/portfolio-construction#longshort---book-market-neutral).

## Conditional risk (`--conditional`)

`--conditional ewma` or `--conditional har` conditions Σ's volatilities on
recent history before the solve, so `--target-te` is measured against *current*
risk rather than a flat trailing-window average. **Default off** — see
[Estimating risk — conditional risk](risk#conditional-risk) for the evidence
gate that decides whether it's worth turning on for your data.

## Black–Litterman (`--posterior bl`)

`--posterior bl` blends the refined alphas with the reverse-optimized consensus
prior, so names outside your alpha's coverage get a real, correlation-propagated
posterior instead of being silently excluded:

```bash
python main.py allocate --objective utility --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN --as-of 2025-06-01 \
  --benchmark-holdings equal --posterior bl --posterior-t-eff 60
```

`--posterior-t-eff` is required (pass the `effective_t` a prior `info` call
measured); `--tau` overrides the pinned τ for sensitivity. **Default off** until
validated out-of-sample. See
[Portfolio construction — Black–Litterman](../engineering/portfolio-construction#blacklitterman---posterior-bl).

## Multi-period trading

`--policy aim` replaces the myopic "jump to this period's optimum" with a
partial-adjustment policy: alphas are discounted for how fast they'll decay
before the book finishes trading into them, and the book moves a derived
fraction `κ` of the gap each rebalance instead of all of it:

```bash
python main.py allocate --objective utility --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN --as-of 2025-06-01 \
  --capital 1000000 --policy aim
```

```
  policy 'aim': κ 0.893 (derived 0.893)  trading half-life 0.8 rebalances  φ 0.000/rebalance  discount 1.00
    decay half-life inf bars (upper CI bound inf, fit R² 0.16, used conservatively)
```

`--trade-rate` overrides the derived `κ` directly. **Default off** — decide
whether it's worth turning on with the net-of-cost A/B:

```bash
python main.py info --policy-ab --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN --start 2024-01-01 --end 2024-12-31
```

which walks the same alpha book forward under the myopic policy and the aim
policy and reports which one actually won net of cost — not a preference. See
[Multi-period trading (engineering)](../engineering/multi-period-trading) for
the derivation.
