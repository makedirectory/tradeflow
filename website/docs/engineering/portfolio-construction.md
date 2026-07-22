---
sidebar_position: 15
title: Portfolio construction
---

# Portfolio construction

`src/portfolio/optimizer.py` turns alphas and risk into the portfolio that maximizes
the **information ratio you can actually implement, net of cost**. Where the
OR-Tools [allocator](./portfolio) maximizes a scalar score subject to constraints —
and so piles weight onto the highest-scoring names regardless of how correlated
they are — `MeanVarianceOptimizer` trades expected residual return against active
risk *and* the cost of getting there:

```
maximize   U(w) = αᵀw − λ_A·wᵀΣw − Σᵢ cᵢ|Δwᵢ| − Σᵢ kᵢ|Δwᵢ|^1.5
subject to Σw = 1,  0 ≤ w ≤ max_weight,  ‖w‖₀ ≤ max_names       (long-only)
```

α is the [alpha](./alphas) vector, Σ the [covariance](./risk-model), `λ_A` the
aversion to active variance, and the last two terms are the name-specific
[transaction cost](./transaction-costs) — linear turnover plus square-root market
impact — priced *inside* the objective rather than tacked on afterward.

:::note Research clock — a proposal, not an order
This is portfolio *construction*: it proposes target weights (a config a human
promotes), it never places an order. It is deliberately separate from the operational
position [sizer](./portfolio) used by `live --portfolio`, so no covariance model
reaches the trade clock.
:::

## The closed form anchors everything

With no constraints or cost the optimum is closed-form, and it's worth stating
because every diagnostic is read against it:

```
w* = (1 / 2λ_A) · Σ⁻¹ α          IR* = √(αᵀ Σ⁻¹ α)
```

`IR*` is the **best achievable information ratio** — fully determined by `α` and `Σ`.
This is why Σ must be invertible (the reason for [shrinkage](./risk-model)) and why
scaling alphas correctly matters.

**You specify tracking error, not λ.** Users think in TE, so the optimizer inverts
the relation `λ_A = IR* / (2·ψ_target)` — pass `--target-te 0.04` and it solves for
`λ_A` such that the optimal tracking error is 4%.

## The transfer coefficient makes constraints visible

Constraints (and cost) pull the implemented portfolio away from `w*`. The **transfer
coefficient** measures the damage, and the achievable IR degrades exactly as:

```
TC = corr(α, Σ-adjusted active weights) ∈ [-1, 1]        IR_achieved = TC · IR*
```

So tightening the cardinality cap or the position limit lowers TC and the predicted
IR — a far more honest knob than "maximize score." A regression test pins this
monotonicity. The report also surfaces predicted TE, predicted IR, value added, and
turnover.

## Cost inside the objective (cost-aware by default)

A *uniform* per-notional cost on a fully-invested, long-only book is a constant in
the objective — it shifts the value but never changes the optimal weights. Real
cost isn't uniform (spreads and ADV differ across names), so putting it inside the
objective genuinely changes *which* names get weight, not just how much of them
cost:

```
cᵢ = commissionᵢ + spreadᵢ/2              # linear, one-way turnover rate
kᵢ = η·σᵢ·√(capital/ADV$ᵢ)                # √-impact coefficient (needs capital)
```

Both are annualized by `÷ holding_period_years` to match the (annualized) alpha
units. `kᵢ` is zero — the solve is linear-only — unless `capital` is supplied
(the "ship linear first" default).

**Solved exactly in pure numpy, no conic-solver dependency.** The objective's
smooth part (`αᵀw − λwᵀΣw`) gets a gradient step; the nonsmooth cost term is
applied via its **exact proximal operator**, composed with the capped-simplex
budget/box projection. That proximal step separates per name given a single
budget dual and each coordinate is closed-form: a **soft-threshold around `w₀`**
for the linear term (this is where the no-trade band comes from) and a
**quadratic-in-√ root** for the √-impact term. The whole thing is solved by the
same one-dimensional budget bisection the cost-free projection already used.
With `cᵢ = kᵢ = 0` the solve reduces *exactly*, byte-for-byte, to the cost-blind
projected gradient above.

### The no-trade band is emergent, not a threshold

A name's weight doesn't move until its marginal value crosses its own cost band —
`|move| ≤ cᵢ` keeps it exactly at `w₀ⁱ`. This isn't a post-hoc "don't trade if the
change is under X%" heuristic; it falls straight out of the proximal operator, and
its width *is* the name's own linear cost rate. Cheap, liquid names get a tight
band and trade freely; expensive or illiquid ones get a wide band and only move
when the alpha signal clearly justifies the cost.

### Two cost figures, two purposes

- **`expected_active_return_net`** (the headline) subtracts a **round-trip**
  haircut on the *held book* — entering and exiting every position, amortized over
  the holding period. This is the same model [`capacity`](#capacity) prices, so
  the two numbers agree.
- **`expected_active_return_net_oneway`**, `cost_drag`, `linear_cost`, `impact_cost`
  (the detail) are the **one-way** cost of *this rebalance's* turnover — exactly
  what the objective charged.

`--gross-objective` drops the cost term from the objective (still reports the
ex-post drag) for attribution — "how much did the cost-aware solve actually buy
me?"

### Capacity

As capital scales, √-impact cost grows ∝ `√capital`, so net alpha is
monotone-decreasing in capital. `capacity_capital` is the size at which net active
return crosses zero, found by bisection over the proposed portfolio's impact cost
— the same cost model, so the number agrees with the round-trip haircut above.

## How it solves (pure numpy)

- **Unconstrained**: the closed form above (used for `IR*`, `w*`, and the calibration
  identities).
- **Constrained, cost-free**: `U` is a concave quadratic, so the long-only / box /
  budget optimum is a small convex QP solved by **projected gradient** with a
  capped-simplex projection.
- **Constrained, cost-aware**: the same projected-gradient loop, but each
  projection step is the proximal operator described above instead of a plain
  clip — see [the cost section](#cost-inside-the-objective-cost-aware-by-default).
- **Cardinality** (`‖w‖₀ ≤ k`) and the **dust floor** (`w ∈ {0} ∪ [min_weight, max_weight]`)
  are both non-convex, so each is handled the same pragmatic way regardless of
  cost-awareness: solve the convex program, drop the offending names (the smallest
  beyond the cardinality cap, or any stuck in the `(0, min_weight)` hole), and
  re-solve on the survivors until stable.

## Edge cases it handles

- **Infeasibility.** `max_names · max_weight < 1` can't fund the book; the optimizer
  returns `feasible=False` and names the binding constraint rather than a silent empty
  portfolio.
- **Turnover from `w₀`.** Turnover is measured against current holdings (`w₀`, default
  cash), so the same target from where you already are costs nothing.

## Benchmark-relative construction (`--benchmark-holdings`)

By default every quantity above is **cash-relative** — turnover, risk, and the
optimum are measured against an all-cash `w₀ = 0`. Pass `--benchmark-holdings`
(`"equal"` over the covered universe, or a `symbol,weight` CSV/JSON holdings file)
to make the benchmark a genuine **portfolio** `w_B` instead of just the return
series used for beta/residual-vol:

- Risk and expected return move into **active space**: `w_a = w − w_B`, and
  tracking error becomes the real thing (`ψ = √(w_aᵀΣw_a)`), not the total
  volatility of a cash-relative book.
- Alpha is **neutralized against the Σ-implied benchmark beta**
  (`αᵀw_B = 0` exactly, via the one canonical `β = Σw_B/(w_Bᵀ Σ w_B)`) so the
  optimizer carries no implicit benchmark-timing view — only binding constraints
  can reintroduce a nonzero active beta, and the report flags it
  (`active_beta`, `residual_risk`, with `ψ² = β_a²·σ_B² + ω²` split out).
- **Reverse optimization** (`implied_returns`) answers the question the other
  way: what consensus expected returns would make `w_B` itself the mean-variance
  optimum? Feeding those returns back in with zero cost reproduces `w = w_B`
  exactly — the round-trip is the sharpest available integration test, and the
  report's `consensus_returns` block is what your alphas are really deviations
  *from*.

Cost stays anchored at `w₀` (current holdings) — risk and cost intentionally read
two different reference points. Without `--benchmark-holdings` every quantity
reduces byte-for-byte to the cash-relative behavior above.

## Long/short (`--book market-neutral`)

The long-only box `[0, max_weight]` forces every unattractive name to a *forced
underweight* of exactly `−w_B` (you can't go lower than zero) — a real, uncosted
constraint. `--book market-neutral` relaxes it:

```
box:     [−short_max_weight, max_weight]     budget: Σw = 0
```

A relaxed short side needs two things a long-only book doesn't:

- **A mandatory gross-leverage cap** `‖w‖₁ ≤ L` (`--gross-leverage`) — an
  unconstrained long/short book on a noisy Σ is a leverage machine (error
  maximization, un-truncated by the long-only bound). The cap is enforced by an
  outer dual on top of the budget solve: most solves never need it (it only
  sometimes binds), so the extra bisection runs only when `‖w‖₁` already exceeds
  the cap.
- **Short-side borrow carry** — `Σ borrowᵢ·max(−wᵢ, 0)`, a per-name annualized
  rate (`CostInputs.borrow`, defaulting to the cost model's flat rate) priced as
  a second, zero-anchored kink that composes with the turnover kink at `w₀`.

`--longshort-report` solves the **same** alphas/Σ/cost long-only *and*
market-neutral and reports the difference: the measured IR shrinkage from the
long-only constraint (next to an illustrative reference curve, not a verified
formula — compare the measured number, don't trust the curve as truth), both
transfer coefficients, and the long-only book's incidental **size exposure** — the
tilt toward large-caps a long-only book picks up from being unable to short the
small, unattractive names, even when alphas are drawn independent of size.

`--book long-only` (the default) is the exact code path used before this
existed, byte-identical.

## Black–Litterman (`--posterior bl`)

Two problems with treating alphas as the whole story: a name with no signal reads
as a *hard* zero-view ("this name earns exactly consensus") rather than
ignorance, and view confidence is baked only into magnitude. `--posterior bl`
blends the refined alphas with the reverse-optimized consensus prior via a
precision-weighted Black–Litterman update:

```
μ_post = τΣPᵀ(PτΣPᵀ + Ω)⁻¹q
```

— a single `K×K` solve (`K` = covered names) with prior precision from `(τΣ)⁻¹`
and view precision `Ω` derived from measured IC by a **calibration identity**: a
single covered-name view, uninformative prior, `τ = 1/T_eff`, must reproduce the
same level-shrunk alpha the [refinement pipeline](./alphas#forecast-refinement-v2)
already computes — forcing `Ω_n = ω_n²/(T_eff²·IC²)` rather than the naive
`ω²(1−IC²)`. This is the proof that IC-uncertainty is never double-counted
between the refinement's level shrink and BL's `Ω`: the refine step stays
*unshrunk* on this path (`shrink_chain` gains a `bl` step instead), so the
haircut is applied exactly once.

The payoff: names outside the alpha's coverage get a real, Σ-**propagated**
posterior (correlated with a name that *was* scored) instead of being excluded
outright — "never zero the correlated forecasts." `--posterior-t-eff` is
required (τ is pinned, never tuned — pass the `effective_t` a prior `info` call
measured); `--tau` overrides it for sensitivity. The report gains a
per-name consensus/view/posterior/source table plus τ-sensitivity
(recomputed at `τ/2` and `2τ`).

**Default off** until validated out-of-sample net of cost — the propagated
alphas are a real forecast, not just a reporting convenience, and this pipeline
ships nothing without evidence (see [Evaluation metrics](./evaluation-metrics)).

## Multi-period trading (`--policy aim`)

The construction above is still **myopic**: every rebalance it pays real cost to
reach *this period's* optimum, ignoring that alphas decay (some of what it trades
into is gone by the time it fully arrives) and that today's trade sets tomorrow's
starting point. `--policy aim` replaces the single-period jump with a
partial-adjustment policy — see [Multi-period trading](./multi-period-trading) for
the full derivation. **Default off**, same evidence-gated posture as
Black–Litterman above.

## Where it runs

`services/analysis.py::construct_portfolio` scans the universe, builds
benchmark-neutral alphas and Σ as of the date, optimizes, and returns the proposed
weights plus the report. The CLI (`python main.py allocate --objective utility
--target-te 0.04`) and the read-only MCP tool `construct_portfolio` route through
it — though the MCP surface doesn't yet expose the cost-aware, benchmark,
conditional, posterior, or policy knobs (see
[MCP server](./mcp-server.md#known-gap)). See the [usage guide](../usage/portfolio.md).
