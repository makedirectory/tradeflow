---
sidebar_position: 18
title: Information horizon
---

# Information horizon

Every alpha is perishable: its forecasting power (IC) decays over time at a rate that
is an intrinsic property of the signal. `src/alphas/horizon.py` measures that decay and
turns it into two free wins — a principled **rebalance cadence** and a **lagged blend**
that raises the information ratio.

:::note Research clock only
A diagnostic over an existing alpha's decay structure; it places no orders.
:::

## Decay and half-life

IC measured at increasing lags `n` (the alpha at `t` vs the residual return realized
`n` periods later) follows:

```
IC(n) = IC₀ · δ^n            half-life HL = −ln 2 / ln δ
```

`fit_decay` fits `δ` by log-linear regression on the reliable, positive-IC lags and
reports the fit `R²` — a low `R²` means the decay isn't a clean exponential, so don't
over-trust the half-life. Because value added scales with IR², the **value-added
half-life is half the IC half-life** — stale alphas are more damaging than they look.

The fit also reports a confidence band on the half-life (`half_life_lower` /
`half_life_upper`, from the regression's own slope standard error) — short
histories give wide bands, and a point estimate alone invites over-trusting a
noisy fit. The [multi-period trading policy](./multi-period-trading) uses the
upper (slower-decay) bound when discounting alphas, so a shaky short-history
estimate doesn't prematurely kill a signal that's actually persistent.

## Cadence and the breadth↔frequency tradeoff

```
IR(Δt) = IC(Δt) · √(1/Δt)
```

Rebalancing faster raises `√BR` but acts on less-confirmed information (lower `IC`) and
costs more. The optimum is interior: `recommended_cadence` returns the `Δt` that
maximizes this IR proxy, so the cadence is *chosen*, not assumed. The half-life is also
the holding period [transaction cost](./transaction-costs) should be amortized over.

## The optimal current/lagged blend

Given decay `γ` (= δ) and the signal's autocorrelation `ρ`, the IR-maximizing weight on
the current signal is:

```
w_now = (1 − γ·ρ) / (1 + γ² − 2·γ·ρ)        w_lag = 1 − w_now
```

- `δ > ρ` → **diversify**: `w_lag > 0` — the lag carries independent information.
- `δ < ρ` → **hedge**: `w_lag < 0` — it mostly cancels current noise.
- `δ = ρ` → the latest signal alone is sufficient.

:::note Superseded by the multi-period trading policy
This two-point blend was the first, ad-hoc answer to "should I slow down
trading into a decaying signal?" The
[multi-period trading policy](./multi-period-trading) (`allocate --policy aim`)
generalizes it into a continuous decay discount composed with the optimizer's
own no-trade band — the blend here is the degenerate two-point special case.
Prefer `--policy aim` for new work; this report's recommendation stays accurate
on its own terms either way.
:::

## Matching signal to return horizon

A signal of half-life `HL` predicts returns best at a horizon ≈ `1.26·HL` (where the
signal-return correlation peaks). Surfaced as `peak_return_horizon` so a short-horizon
signal isn't scored against a long-horizon return — a common way to understate real
skill.

## Caveats it surfaces

- **Decay should be measured out-of-sample** — fitting it on the selection data
  overstates persistence.
- **The blend adds turnover** — its lagged leg is priced through the [cost](./transaction-costs)
  model and `blend_recommended` is only true when the blend *diversifies* and its
  annual turnover cost is modest; a high-turnover blend that would lose net is flagged.
- **Regime-dependent decay** — the half-life isn't constant; re-estimate on a rolling
  window rather than freezing one number.

## Where it runs

`services.compute_horizon` measures the IC-vs-lag profile (reusing the
[information-analysis](./information-analysis) forward-residual machinery), fits the
decay, and recommends the cadence + blend. The CLI (`python main.py horizon`) and the
read-only MCP tool `compute_horizon` route through it.
