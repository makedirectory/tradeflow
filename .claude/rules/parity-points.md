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

The *defaults* then turned out to be a second, quieter half of the same point:
`--folds` defaulted to `None` on the CLI and `4` in the service. Both build four folds
— `build_folds` falls back to `n_folds or 4` — so a default run over each surface
validated identically and keyed differently, and a walk-forward recorded over one was
never found again over the other. Converged on `None`: it is what the recorded history
carries, and the honest value when `--train-days`/`--test-days` derive the fold count
and this parameter has no effect at all. Identical construction is not parity if the
two callers reach it with different arguments.
*Guarded by* `tests/test_surface_parity.py`, which now compares a *default* run over
each surface and reads every default from its own signature rather than restating it.

**CLI flags ↔ MCP tool parameters** — anything a run can be configured with should be
reachable from both, and an MCP argument the service does not accept fails only at call
time. An agent cannot notice a stale description; it acts on one. The direction that
actually bit: `walkforward --benchmark` had no MCP equivalent and no service parameter,
so over MCP every fold reported `benchmark_available: False` and the benchmark-relative
promotion prerequisites were never *evaluated* — a gate that cannot be configured on a
surface is not stricter there, it simply does not run, and nothing says so.

`screen` is guarded the stronger way: its CLI flags are enumerated *from the parser*
and every one must have a counterpart in the MCP tool's signature, rather than a
hand-written list of the flags somebody remembered. Two knobs are spelled differently
because argparse cannot take a mapping (`--range` → `param_ranges`, `--max-positions` →
`position_limits`); the rename table that permits this is itself the loophole, so a
further test follows both all the way to the service argument. A rename may only record
that two surfaces reach the same argument — never excuse a knob one surface lacks.
*Guarded by* `tests/test_surface_parity.py`.

**Config ↔ what actually got validated** — *converged*. A config's `position_limits` is
not a tunable param, so anything reconstructing a strategy from params alone drops it. A
config asking for eight positions was walk-forward validated at one. Every sweep over a
parameter space needs the same three lines, and each one that wrote its own was a place
the book could go missing again, so it is now
`strategies.base.build_with_limits` — used by the walk-forward validator, the parameter
optimizer, and the screen.
*Guarded by* `tests/test_surface_parity.py`.

**Scanner registry ↔ the driver's class attribute** — `services.registry.SCANNERS` and
`SymbolScanner.SCANNERS` are two dicts holding one answer, kept in step by
`refresh_registries()`. The class attribute used to be seeded from the scanner package's
own literal; once the example scanner moved to `tradeflow.demo` that literal went empty,
so the attribute was empty too and a bare `import symbol_scanner` gave
`available() == []` — the class worked or not depending on whether some *other* module
had been imported first. `SymbolScanner._registry()` now delegates (lazily, because
`registry` imports it), which also means a discovery failure leaves the registry's
seeded reserved names in play rather than failing every name the CLI still advertises.
*Guarded by* `tests/test_extension_registry.py`, which compares the two and runs the
bare import in a subprocess — every other test has already imported the registry and
would mask the ordering entirely.

**What the sdist ships ↔ what the instructions reference** — `init --example-pack` copies
whatever the distribution carried, and the CLI's next printed line and the pack README's
runbook both name `configs/breakout.json`. A `pyproject.toml` exclude is not a list of
directories: the patterns are gitignore-style, so an unanchored `configs` also matched
`example/configs/`. `example_pack_source()` keys on `example/pyproject.toml`, which still
shipped — so the scaffold *succeeded* and printed a next step that did not exist.

The environment is an unpacked sdist tree, not a pip install: the wheel omits the pack
deliberately and `--example-pack` refuses there with a clear message. That is precisely
the environment the sdist carries the pack *for*, and the one nothing was looking at.

Anchor anything naming repo-root state, and test by building the file list rather than
reading the manifest — reading it is exactly what missed this.
*Guarded by* `tests/test_packaging.py`.

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
