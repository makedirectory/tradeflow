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

The **execution floor** was the case where one side simply had nothing: `min_notional`
was enforced in the backtest and absent from the live path entirely, so a config
validated with a floor traded without one — the same declared book admitting different
orders depending on which clock was asking. It was worse than a silent difference,
because the live preflight *printed* the floor on every run, so the surface said the
limit was in force while nothing applied it. Now `execution.sizing.below_min_notional`
is the one definition both clocks call, and both the sub-floor refusal and the
size-rounds-to-zero one carry a `reason_code` — the two ways a book can be too small to
express a position in a name are the signature of trading below the size something was
validated at, and they only show up as a number if they can be grouped.
*Guarded by* `tests/test_min_notional_parity.py`, which puts one order to both
admission paths and compares the verdicts either side of the boundary, rather than
asserting each clock separately — the backtest's own test passed throughout.

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

**The validated contract ↔ the contract a small run actually trades** — one book
written twice, and the only permitted relation between them is the scale.
`services.smallreal` is the single rule (fractions unchanged, counts unchanged, dollar
ceilings scaled, venue floors absolute), and the hazard is not two implementations but
one implementation whose result never arrives: the scaled book has to be *applied* to
the strategy, recorded in the telemetry session header, and printed in the preflight,
and those are three separate statements of it. A mutation deleting the line that applies
it to the strategy passed every test, because the tests asserted the recorded contract —
the arithmetic — rather than the effect. The guard now captures the strategy the engine
is handed and asks it to size a position.

Small-real is also the one command deliberately **absent from MCP** rather than mirrored
there: see the CLI↔MCP entry below, and `mcp.server.OPERATOR_ONLY`. That absence is the
exception the next entry's rule would otherwise forbid, so it is written down in both
places.
*Guarded by* `tests/test_small_real_command.py`, `tests/test_small_real_contract.py`.

**CLI flags ↔ MCP tool parameters** — anything a run can be configured with should be
reachable from both, and an MCP argument the service does not accept fails only at call
time. An agent cannot notice a stale description; it acts on one. The direction that
actually bit: `walkforward --benchmark` had no MCP equivalent and no service parameter,
so over MCP every fold reported `benchmark_available: False` and the benchmark-relative
promotion prerequisites were never *evaluated* — a gate that cannot be configured on a
surface is not stricter there, it simply does not run, and nothing says so.

**MCP intentionally does not expose evidence-gated construction knobs** whose own specs
say they have not cleared adoption gates. This is not a parity miss; it is the evidence
gate applying to an agent surface. Revisit only when the gated feature is promoted.

Conditional risk, the Black-Litterman posterior and the multi-period aim policy ship
*off* because their adoption gates do not clear, and the CLI makes a human read that
before using them. An agent reads a description as fact and acts on it at machine speed,
so a surface that offered the same knob as a neutral argument would become the easier way
to switch on a feature nothing has validated. Being reachable is not being validated.
`mcp.server.DEFERRED_PARAMS` is the list with a reason per parameter — `trade_rate` is
there for a second reason worth keeping distinct: the service passes it only when
`policy` is `"aim"`, so exposing it alone would be a knob that reaches nothing, which is
the defect `_refuse_inert_flags` exists to stop on the CLI.

Everything *ungated* must still be reachable, and the guard enumerates the **service
signature** rather than a remembered list: every parameter is exposed or in
`DEFERRED_PARAMS` with a reason. `construct_portfolio` exposed nine of the service's
parameters while `allocate` carried about thirty, and the check found two more
(`neutralize`, `scanner`) that reading the code had missed.

Presence in the schema is not reach. The defect that started this was a tool advertising
a `neutralized_against` field it could never populate, because no parameter existed to
pass — and a mutation leaving the parameter in the signature while dropping it from the
forwarding dict is indistinguishable from the outside. So a further test calls each tool
and asserts the *service* received the value, and that an omitted knob is not forwarded
at all, since a restated default is a second definition to keep in step.
*Guarded by* `tests/test_mcp_surface.py`.

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
main.py`, `uv sync`, and `.env.example` do not exist for an installed reader. Use
`services.setup.invocation` (the CLI's `_invocation` delegates to it) rather than a
literal — services print instructions too.
*Guarded by* `tests/test_setup.py`, `tests/test_surface_parity.py`.

**Strategy convention ↔ engine execution** — `generate_signals` keys a signal at the bar
whose close produced it, and the engine must execute it on the bar after. Neither side
can see the other's assumption.
*Guarded by* `tests/test_signal_causality.py`, and now also by
`tests/test_causality_probes.py`, which restores the one-bar look-ahead deliberately and
requires the probes to say so. The two are different things: the first pins the property
on this engine, the second pins the *detector* — a probe asserted only against correct
code is exactly the probe that passed for three days.

**Probe class ↔ what a probe actually tests** — the feed-shift leakage probe tests for
future data; the causality probes test intra-bar causality and the as-of clock. Neither
can see what the other looks for, and the shift probe cleared a candidate whose every
signal executed a bar early. The hazard is not code drift but a reader conflating them,
so the distinction is stated in the module docstring, the tool description, the CLI help,
the usage guide and the walk-forward wiki — and a test asserts the report carries it.

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

The bump has a second half that is easy to miss: those lookups are scoped to the current
version, so on the day it changes a campaign's entire history stops matching the default
listing. An empty table reads as "nothing was ever run here", which is the most alarming
possible way to learn a bump happened. `trials list` now counts what the filter hid and
says so, and `trials archive` is the command for actually retiring the era.

**The journal's location** — *converged*. It was two constants holding one path, kept
in step by a comment; had they diverged the store would have indexed a different file
from the one being written and the multiple-testing correction would have deflated
against half its evidence, with nothing erroring. Now one definition in
`settings.trial_journal_path()`, which is the layer both depend on.
*Guarded by* `tests/test_surface_parity.py` — kept, because the constants still exist
and could be re-pointed.

**The declared schema ↔ the tables actually on disk** — `store/trials._SCHEMA` says what
the index is; the SQLite file says what it *is*, and the two are separate objects that
were kept in step by a version stamp that could not see either one. `CREATE TABLE IF NOT
EXISTS` is a no-op on a table that already exists with the wrong columns, and `rebuild()`
emptied the tables rather than recreating them, so a schema change reached every fresh
database and no existing one — while stamping the new version onto the old shape. A store
written before the quarantine columns therefore reported the current version and raised
`no such column: contaminated_at` the moment anyone quarantined anything — which is to say
`trials mark-contaminated` was dead on every store that already existed, and it was
shipped that way. `best()` was worse: it filters quarantine in Python, so it silently
ranked rows somebody had quarantined.

Now every open compares the file's real shape against a database built from `_SCHEMA`
itself — never a hand-written list of columns, which would be a third statement of the
same thing to forget — and a mismatch drops the derived tables, recreates them and
replays. The version stamp is one of three triggers, not the trigger. Dropping is safe
for exactly one reason, and it is the precondition to check before ever extending this:
**every column of every table here is reconstructable from the journal**, quarantine
flags included. A rebuild refuses rather than replacing real rows with an empty index
when the journal cannot be read.
*Guarded by* `tests/test_trial_store.py`, which builds a store from the *shipped* v4 DDL
rather than from today's schema minus a column, and covers the case a version comparison
cannot see: a current stamp on a stale table.

**The exit-reason split over a live result ↔ over a recorded one** — *converged*. The
backtest's "Where the P&L came from" block grouped a pandas frame in the CLI; asking the
same question of a *recorded* trial reads a stored `{columns, rows}` table. Two
implementations of one idea, one printed under every backtest and one reached from the
trial browser, differing in the thing that matters most about them — whether the rows
they were given are all of the run's trades. `analytics.trade_analytics` is the one
definition and the CLI block is a renderer over it, reaching it through
`trades_payload(frame, max_rows=None)` so the live path declares itself complete rather
than being assumed so.
*Guarded by* `tests/test_trade_analytics.py`.

**Trial-analytics knobs across CLI and MCP** — `trials analyze` / `analyze_trial` and
`trials compare` / `compare_trials`. Both surfaces enumerate their flags from the parser
and must have a counterpart in the MCP signature, and both *defaults* are compared too:
`allow_partial` and `min_overlap`. A capped trade table summed silently on one surface
and refused on the other is the same trial answering one question two ways depending on
who asked, which is what a shared default exists to prevent.
*Guarded by* `tests/test_surface_parity.py`.

**The book a run validated at ↔ the book its identity records** — a walk-forward folded
the limits it was *overridden* with into its dedup key, not the book it actually ran at.
A run with no `--config` therefore recorded no `_limits` while still having a book, so a
class default moving from one position to eight changed the experiment without touching
params, universe or window — two such runs hashed alike and the second was answered from
the first, which is exactly the failure `limits_key` was created to prevent, surviving in
the one case it did not cover.

`walk_forward_recipe` now takes the strategy class and resolves the book itself through
`strategies.base.resolve_book`, rather than accepting one: a caller handed a raw override
can pass it straight through, and that is how this got missed. The same resolver writes
the saved config, so the recipe and the config cannot disagree about the book for one run.
Resolution is deliberately params-independent — a search's identity is fixed before its
winner is known.

**This changed the identity of every walk-forward that previously omitted `_limits`**, so
their memos miss once. Accepted: recomputing is cheaper than serving a one-position result
to an eight-position question. Replay is unaffected — a rebuild reads each journal line's
own recorded recipe, so historical rows keep their original hashes.
*Guarded by* `tests/test_surface_parity.py`, whose old assertion that an un-overridden run
keys as it did before limits existed is now reversed on purpose, with the reason recorded.

**A saved config's runnable contract ↔ the identity it was validated under** — a
config's `position_limits` and the `_limits` folded into the trial's dedup identity are
the same book written twice, and they were not checked against each other. Omitting the
first does not leave the book unspecified: it resolves at load to the strategy class's
default of one position, so a config whose own provenance recorded eight silently traded
one — the file disagreeing with itself, in the direction that costs money.

Four writers, each losing it a different way: `trials promote` discarded it with the
reserved `_` keys; `walkforward --save-config` wrote `create_with_defaults()` — the
*class* book rather than the run's — so round-tripping a config through it shrank the
book every time; the research agent wrote none; and the MCP service had no parameter to
put one in, so an agent holding campaign material could not write a runnable config at
all. `services.analysis.recorded_book` is the one resolver, and it takes several sources
because the book is not in the same place for every kind: a walk-forward keeps it in its
recipe, a backtest in its params, since a backtest's dedup identity *is* its params.

*Guarded by* `tests/test_saved_config_book.py`, which enumerates the `save_config` call
sites from the AST rather than from a list somebody remembered, and promotes a trial then
reloads the file and builds the strategy to check the number that arrives at the far end.

**One provenance format, wherever a config is written** — `save_config` is the
portability format and campaign material is a *field* of it (`provenance.campaign`),
never a second artifact beside it. A campaign export living somewhere else would be a
second provenance schema, which is the hazard this list exists for: two files claiming
to say how a config was produced, drifting, with neither one wrong enough to notice.
The MCP `save_config` tool builds its `Provenance` from the same dataclass, so a block
the dataclass does not accept fails there while `trials promote` succeeds — two schemas
by accident rather than by decision.
*Guarded by* `tests/test_campaign_material.py`, which writes a config over each surface
and reads the same block back, and checks a pre-campaign config still loads.

**How to invoke this copy** — *converged*. `cli._invocation` was the helper the
installed-copy-vs-checkout entry names, and it lived in the CLI, so a *service* that
needed to print an instruction — a config's campaign block naming the command that
reads its stored trades — had nowhere to get the answer but a second copy. Now
`services.setup.invocation`, with the CLI delegating.
*Guarded by* `tests/test_setup.py`, `tests/test_surface_parity.py`,
`tests/test_campaign_material.py`.

**Opening the trial store** — *converged*. It was three copies, not the two listed here:
`cli._open_trial_store`, `services.analysis._open_trial_store` and
`mcp.server._trial_store`, each deciding for itself which journal to index — the one
decision they must never disagree about, since the multiple-testing correction rests on
there being one journal. One definition in `services.audit.open_trial_store`, which is
where it has to live: the default must be `audit.default_trial_journal()` rather than
`store.trials.default_journal_path()`, because those two resolve alike in production and
differ under a redirected journal, which is every test in the suite.
*Guarded by* `tests/test_surface_parity.py`, which opens all three under a redirect and
compares the paths they reach, rather than three tests that each pass.

Still parallel and **unguarded**: `cli._find_cached_trial` / `services._find_cached_trial`,
`cli._worker_data_spec` / `services._worker_data_spec`, `parallel._build_cost_model` /
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
