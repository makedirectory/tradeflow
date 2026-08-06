---
sidebar_position: 15.5
title: Multi-period trading
---

# Multi-period trading: aim in front of the target

`tradeflow/portfolio/policy.py` replaces the myopic "jump to this period's optimum" with
a **partial-adjustment** policy: trade a fraction of the way toward a target that
is itself *discounted* for how fast the alpha will decay before you finish
trading into it. It wraps
[`MeanVarianceOptimizer`](./portfolio-construction) — different alphas in, an
affine step after — and never modifies the solver itself.

:::note Default off
Like [Black–Litterman](./portfolio-construction#blacklitterman---posterior-bl),
this ships only if it beats the myopic policy net-of-cost, out-of-sample. See
[Evaluation](#evaluation) below.
:::

## The problem with jumping every period

Each rebalance, the plain construction pays real cost to reach the *current*
period's optimum. Two things make that wasteful when alphas are persistent:

1. **Alphas decay.** A signal with a short half-life is half gone by the time a
   slow, cost-constrained book finishes leaning into it — paying full freight for
   it overpays.
2. **Today's trade sets tomorrow's starting point.** A sequence of myopic solves
   over-trades relative to the true dynamic optimum whenever the alpha this
   period predicts something about the alpha next period too.

With quadratic trading costs and an exponentially-decaying signal, the dynamic
program that accounts for both collapses to two rules:

- **The aim portfolio** — the plain mean-variance solve on alphas *discounted* by
  a factor that shrinks toward zero for fast-decaying signals — "aim in front of
  the target": construct the portfolio you'll still want once you get there, not
  the one you want today.
- **Partial adjustment** — trade a fraction `κ ∈ (0, 1]` of the gap each period:

  ```
  w_t = w_{t−1} + κ·(aim_t − w_{t−1})
  ```

  `κ` increases with risk (urgency to sit near the target) and decreases with
  cost (patience pays when trading is expensive).

## Deriving κ

`κ` comes from a single-asset discrete-time LQ control problem: quadratic risk
`λ_Aσ²w²` and quadratic trading cost `(c₂/2)Δw²`. (Our real cost is linear +
3/2-power, not quadratic; `c₂` is its *local curvature* at the book's typical
trade size — a calibrated approximation, validated empirically below, not
asserted exact.) Matching the Bellman value function's quadratic coefficients at
the fixed point gives, with `s = λ_Aσ²`:

```
κ = (s + √(s² + 2sc₂)) / (s + c₂ + √(s² + 2sc₂))
```

Two clean limits pin it down: `c₂ → 0` (no cost curvature to fight) gives
`κ → 1` — trade the whole gap, every period; `c₂ → ∞` gives `κ → 0` — freeze the
book. `κ` is monotone increasing in risk-aversion·variance and decreasing in
cost, with **no dependence on decay** — the trading *speed* depends only on
risk and cost; *what you aim at* depends on decay.

`c₂` itself comes from the impact term's curvature at the book's *actual* recent
trade size (the plain myopic solve's own turnover, divided by names traded) — so
it's fit at the size the book really transacts, not a hypothetical one.
`--trade-rate` overrides the derived `κ` directly.

## The decay discount — exact, not the textbook approximation

Solving the *other* half of the same fixed point gives the alpha discount
**exactly**:

```
discount(κ, φ) = κ / (1 − δ·(1 − κ))        δ = e^(−φ)
```

where `φ` is the signal's per-rebalance decay rate (`φ = ln2 / half-life`, from
[Information horizon](./information-horizon)). This is *not* the commonly-quoted
`κ/(κ+φ)` — that form is only this formula's small-φ limit, and its error grows
with φ. An earlier draft of this module used the approximation; a synthetic-world
test that reproduces the underlying theorem numerically caught it flipping the
sign of a realized-utility comparison at large φ, confirmed against a direct
value-iteration solve of the Bellman recursion. Since the exact form costs
nothing extra to compute, there's no reason to keep the approximation.

`φ = 0` (a permanent signal) gives discount `= 1` at any `κ`, exactly — a signal
that never decays is never discounted, no matter how slowly the book trades.

### Conservative on shaky decay estimates

Short histories give wide confidence intervals on a half-life
([`fit_decay`](./information-horizon) now reports `half_life_lower` /
`half_life_upper`). The policy discounts using the **upper** half-life bound (the
slower-decay end) — the direction that avoids prematurely killing a signal that
looks fast-decaying only because the estimate is noisy.

## Composing with the no-trade band

The κ-adjusted trade is passed through the *same* proximal budget/box/band
projection the plain cost-aware solve uses (reused via
`MeanVarianceOptimizer._prox_project`, never re-derived) — so κ and the no-trade
band compose through one mechanism instead of stacking as two independent
heuristics that could freeze the book between them. A walk-forward A/B reports
total turnover so over-damping shows up as a number: if both turnover *and* net
IR fall relative to the myopic policy, the aim policy traded too little to
capture what alpha there was — not an improvement.

## The costless fallback is exact, not asymptotic

Without a usable cost curvature — no `capital`, or too little turnover from the
myopic reference solve to fit one — `κ` is undefined, not zero. Treating
"undefined" as "zero" would silently invent a `κ = 1` from a curvature that was
never measured. Instead the policy falls back to **exactly** the plain
cost-aware solve, byte-for-byte identical weights — a tested reduction, not a
limit approached only as cost shrinks to zero.

## Evaluation

`services.run_policy_ab` is the net-of-cost A/B: walk `[start, end]` at spaced
rebalances, construct the **same** alpha book against the myopic policy and the
aim policy, carry each variant's weights forward, and compare realized net IR
(turnover cost priced at the actual holding period). `over_damped` flags the
double-damping failure mode directly — lower turnover *and* lower net IR
together.

On this repo's own synthetic demo data the aim policy does not win: the series
has no genuine decay structure to discount (the fitted half-life comes back
undefined), so the only lever left is `κ`'s trading speed, and the fitted `κ`
here is already close to 1 — leaving almost nothing for the policy to do
differently from the myopic solve. That is the expected, honest outcome on data
built to have no real structure, not evidence against the method — a controlled
synthetic world with genuine decay (constructed the same way the pure-math test
suite does) shows the aim policy's realized-utility edge growing with how fast
the signal decays, exactly as the theorem predicts.

## Where it runs

`tradeflow/portfolio/policy.py` (pure math + orchestration) plus
`services.construct_portfolio(policy="aim", trade_rate=...)`. The CLI is
`allocate --objective utility --policy aim [--trade-rate κ]`; the A/B is
`info --policy-ab [--trade-rate κ]`. Both default off. See the
[usage guide](../usage/portfolio.md#multi-period-trading).
