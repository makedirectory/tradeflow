# Parity points

Some ideas in this codebase exist in more than one place on purpose — the two clocks
must not import each other, the CLI and the MCP server are separate transports, an
installed copy and a checkout are different environments. Every one of those is a place
where two implementations of the same idea can drift apart while both look correct and
every test passes.

**This file is the list.** Before changing anything on it, change both sides.
[Checking for divergence](check-for-divergence.md) is the obligation to consult it, and
applies to ideas not yet listed here.

## The preferred shape: delegate, don't re-implement

`cli.resolve_universe` is four lines that call `services.data.resolve_universe`. That is
the shape to aim for — one definition, reached two ways. A parity point only earns its
existence when delegation is impossible: the trade clock genuinely cannot import the
research clock, and an argparse namespace genuinely is not a function signature.

If you are about to write a second implementation, check first whether a shared helper
would do. `limits_key` exists because that check was skipped once and a trial recorded
over the CLI stopped being found over MCP.

## The list

**Research clock ↔ trade clock** — `engine/backtest.py` and `execution/live_trader.py`.
Position limits, sizing, exit ordering, and signal causality all exist in both. See
[cross-clock parity](cross-clock-parity.md), which is the long form of this entry.
*Guarded by* `tests/test_signal_causality.py`, `tests/test_net_exposure.py`.

**CLI ↔ service dedup identity** — `cli._cost_key` and `services.analysis._cost_key` are
two implementations that must produce the same shape, because a trial recorded over one
surface has to be found by the other. The CLI additionally folds a bar-cache vintage.
*Guarded by* `tests/test_surface_parity.py`.

**A walk-forward's memoization recipe** — *converged*. It was three copies of one dict
(CLI, service, draft-service). `limits_key` was folded into `run_backtest`'s key and into
none of them, so two configs differing only in `max_positions` hashed alike and the
second was served the first's result — reporting a one-position validation as an
eight-position book, which is the single thing a walk-forward exists to rule out. Now one
definition in `services.analysis.walk_forward_recipe`; `cli._walkforward_recipe` is the
namespace adapter over it, the shape `_dedup_params` already had.
*Guarded by* `tests/test_surface_parity.py`.

**CLI flags ↔ MCP tool parameters** — anything a run can be configured with should be
reachable from both, and an MCP argument the service does not accept fails only at call
time. An agent cannot notice a stale description; it acts on one. The direction that
actually bit: `walkforward --benchmark` had no MCP equivalent and no service parameter,
so over MCP every fold reported `benchmark_available: False` and the benchmark-relative
promotion prerequisites were never *evaluated* — a gate that cannot be configured on a
surface is not stricter there, it simply does not run, and nothing says so.
*Guarded by* `tests/test_surface_parity.py`.

**Config ↔ what actually got validated** — a config's `position_limits` is not a tunable
param, so anything reconstructing a strategy from params alone drops it. A config asking
for eight positions was walk-forward validated at one.
*Guarded by* `tests/test_surface_parity.py`.

**Installed copy ↔ checkout** — every instruction printed to a user. `make`, `python
main.py`, `uv sync`, and `.env.example` do not exist for an installed reader. Use the
`_invocation` helper rather than a literal.
*Guarded by* `tests/test_setup.py`, `tests/test_surface_parity.py`.

**Strategy convention ↔ engine execution** — `generate_signals` keys a signal at the bar
whose close produced it, and the engine must execute it on the bar after. Neither side
can see the other's assumption.
*Guarded by* `tests/test_signal_causality.py`.

**Ledger write ↔ ledger replay** — what `record_fill` means by a quantity (`basis`) and
what `_replay` does with it. A cumulative quantity summed as if incremental turned an
order that filled 8 into 21. It then diverged a second time on the *other* reader:
`lifecycles()` ignored `basis` entirely, so an order filled incrementally as 3+3+2
reported a filled quantity of 2 against a submitted 8 — counted as a short fill, with its
notional understated. `filled_quantity()` is now the one rule; `_replay` still applies it
inline because it also carries reset sequencing, so the two are held together by a test
that reads one ledger both ways rather than by a shared call.
*Guarded by* `tests/test_ledger_fill_accounting.py`.

**Engine behaviour ↔ `ACCOUNTING_VERSION`** — any change to what the engine computes
must bump it, or results from two different models compare as though they were one.
*Guarded by* the trial store's accounting-scoped lookups.

**The journal's location** — *converged*. It was two constants holding one path, kept
in step by a comment; had they diverged the store would have indexed a different file
from the one being written and the multiple-testing correction would have deflated
against half its evidence, with nothing erroring. Now one definition in
`settings.trial_journal_path()`, which is the layer both depend on.
*Guarded by* `tests/test_surface_parity.py` — kept, because the constants still exist
and could be re-pointed.

Still parallel and **unguarded**: `cli._find_cached_trial` / `services._find_cached_trial`,
`cli._open_trial_store` / `services._open_trial_store`, `cli._worker_data_spec` /
`services._worker_data_spec`, `parallel._build_cost_model` /
`services._build_cost_model`. Each is a candidate for delegation.

## Finding a new one

```
rg -N '^\s*def ([a-zA-Z_]\w*)' -or '$1' tradeflow --sort path | sort | uniq -d
```

A name defined in two modules is either a delegate (fine) or a parity point (add it
here). The MCP server's tool names deliberately mirror the service's — those are the
transport, not duplication.

## Testing a parity point

Two tests that each pass do not establish parity. The test has to **compare the two**:
build the same thing both ways and assert the results are equal. A parity bug looks like
two green tests, which is precisely why it survives.
