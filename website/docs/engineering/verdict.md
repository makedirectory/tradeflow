---
sidebar_position: 21
title: The composite verdict
---

# The composite verdict

The research pipeline's five stages — universe scan, alpha refinement, signal
combination, portfolio construction, information analysis — each have their own
service function and their own CLI verb. The composite is one more service function
that runs them together under a single set of inputs, and one renderer that prints
the result. It adds **no analytics**: every number in a verdict report was produced
by the step that owns it, and the verdict line is assembled from gates those steps
already computed.

For how to run it, see [One-command verdict](../usage/verdict.md).

## Why composition needs its own function

Four commands run by hand are not the same as one command running four steps. Each
service function resolves its own universe through the scanner and applies its own
default lookback, so nothing forces the flags to agree between invocations. The
report that results *looks* joined-up while potentially describing four different
universes over four different windows — a failure that is harder to spot than four
obviously separate reports, not easier.

`run_verdict` (`tradeflow/services/analysis.py`) closes that by construction:

1. Resolve the universe **once**, from one scan (or the candidates as given, when
   the scanner flags nothing or is disabled). Every later step receives that list,
   never the candidate list.
2. Hand every step the same window, timeframe, benchmark, and cost model. The cost
   assumptions are parameters on `construct_portfolio` for exactly this reason — the
   optimizer prices turnover the same way the rest of the run does.
3. Assemble one dict with stable top-level keys (`scan`, `alphas`, `combination`,
   `portfolio`, `information`, `verdict`, `provenance`) and a `schema` stamp. The
   CLI renders it; the MCP tool returns it; nothing recomputes it a second way.

## One shared fetch

Steps repeat each other's bar requests constantly — the scan, the alpha panel, the
covariance lookback, and the information sampler all want overlapping frames.
`SessionBarCache` (`tradeflow/marketdata/session.py`) wraps the data client's provider for
the lifetime of one composite run and serves a repeat from memory.

It is deliberately **exact-match only**: a response is reused for an identical
`(symbols, timeframe, start, end)` request and never sliced down from a wider
fetch. Slicing would make a step's frames depend on what ran before it — a
reproducibility hazard that would buy nothing a second fetch does not. The steps
legitimately want different lookbacks, so a run issues more than one fetch; what it
never does is issue the same one twice. The report's provenance line prints the
measured ratio (requests issued vs. requests that reached the provider), so the
claim is checkable rather than asserted.

Frames are copied on the way out. Callers downstream mutate what they receive, and a
cache that handed out its own objects would let one step's edits leak into the next
step's inputs.

## The verdict line is a gate, not a summary

`_verdict_gates` reads numbers the steps produced and compares each to that step's
own threshold. Nothing is re-derived and nothing is averaged:

| Check | Source | Threshold |
|---|---|---|
| `ic_tstat` | information | 2 |
| `ir_above_noise` | information | the realized IR's own standard error |
| `sanity_ceiling` | information | 2 |
| `sample_size` | information | the minimum measurable rebalance count |
| `net_of_cost_alpha` | portfolio | 0 |

Three rules keep the one-line answer honest:

- **Disagreement stays visible.** When some checks pass and others fail the verdict
  is `mixed` and names both sides. Collapsing a split decision into one reassuring
  number is the specific thing a summary line invites.
- **A partial run gets no verdict.** Any failed step makes the verdict `incomplete`,
  whatever the completed sections say — a report where the portfolio printed but the
  information analysis did not is an invitation to act on unvalidated weights. The
  CLI exits non-zero so a script can tell the two apart.
- **No evidence is not weak evidence.** An information step that ran but measured
  nothing produces `needs more data`, never a pass earned by the portfolio alone.

Step failures are caught per step rather than allowed to propagate: a run that
completes four stages and loses the fifth reports four stages and says which one
broke, instead of losing thirty seconds of work to a traceback.

## One trial, and the book it proposed

The composite journals exactly **one** trial (kind `verdict`). The steps run as
library calls, not as their own CLI trials — five rows per command would quintuple a
campaign's multiple-testing total and corrupt the deflated-Sharpe accounting the
[trial store](./walk-forward.md#the-trial-store) exists to protect. That same
campaign count is what the report's multiple-testing line is computed against, and
it counts the current run, which is the conservative direction.

The run's dedup identity covers every input any step reads — the strategy's tunable
params, the cost assumptions, and the step-level knobs (`target_te`, `risk_model`,
`lookback_days`, and the rest) that a caller can vary without the composite's own
flags changing. An identity built only from the flags the parser declares would let
two materially different runs collide as the same trial.

Memoization needs the whole composite back, not just the summary floats a trial row
carries, so a journaled run also writes its full result object to
`logs/artifacts/verdict_<trial_id>.json`. The path is derived from the trial id
rather than recorded anywhere, so nothing has to stay in sync; a missing or
unreadable artifact simply sends the caller down the re-run path, which is always
safe.

### `trial_weights`

A verdict run's proposed book — weights, active weights when a benchmark portfolio
was given, and the portfolio's factor exposures (`Xᵀw` over the risk model's own
exposure block) — is journaled with the trial and mirrored into a `trial_weights`
companion table, on the same journal-first terms as the per-trial return series:
the journal is the source of truth, and `rebuild()` reconstructs the table from it
alone.

This starts closing a real gap. No persisted holdings artifact existed anywhere,
which is why [attribution](./attribution.md) recomputes each period's book from
stored bars. This change only *writes* the artifact, for its own runs; teaching
attribution to consume it is a separate change.

Every trial recorded before the table existed has no book stored, and every trial
kind that proposes no portfolio never will. Both read as *not recorded* — distinct
from a book that is genuinely empty.
