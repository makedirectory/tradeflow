---
sidebar_position: 99
title: Changelog
---

# Changelog

QuantRed (TradeFlow until the rename below) started in June 2023 as a personal project and has been rebuilt more than
once since. It is not a product, and it has never managed real money — it exists
partly for the pleasure of building it, and partly to demonstrate something worth
demonstrating: **the machinery serious quantitative research runs on is buildable by
one person.** Structural risk models, cost-aware portfolio construction, walk-forward
validation with honest multiple-testing correction, point-in-time data discipline —
none of it requires a hedge fund's resources. It requires knowing which parts matter
and being unwilling to fool yourself about the results.

That second goal is why so many entries below describe *refusals* rather than
features: a gate that would not pass, a default left off because its evidence did not
clear, a number that turned out to be an artifact. The hard part of this domain is
not computing a Sharpe ratio. It is building something that will tell you your idea
does not work.

Every release, newest first. The project's status is **Experimental** — interfaces
and gate thresholds may still change, and it has no production users or live capital.

Dates are release dates. Versions before 2.0.0 were never published to a package
index; they are tagged in the repository.

---

## Unreleased

TradeFlow is now **QuantRed**, and a first session works from install to a result
without guessing. The history below keeps the old name as it was written.

### Accounting v6

`ACCOUNTING_VERSION` is **6**. Window boundaries computed by adding days to a market
date - walk-forward folds, the holdout, warmup starts, scanner and beta lookbacks, the
research agent's holdout - kept January's offset, so from March to November they sat at
01:00 New York instead of midnight. Daily bars are stamped at midnight, so a window
starting on a summer day skipped that day's bar: every such out-of-sample fold began a
bar late. There was no look-ahead (the gap between in-sample and out-of-sample only grew),
but the windows were not the ones asked for, so v5 walk-forwards are not comparable with
v6 ones. Earlier trials drop out of the default `trials list` (it says how many);
`trials archive` retires the era.

### Honest labels

- **Capacity** says which bound it hit instead of a sentinel: `capacity_capital` is
  null with a `capacity_status` (`no_edge`, `below_search_floor`,
  `above_search_ceiling`) when no capital was solved for, and the CLI, HTML report
  and MCP payload say it in words rather than `$1,000` or `0`.
- **Attribution** counts why rebalances were skipped (too few names, covariance not
  buildable, no benchmark weights, no spread) and names the dominant reasons, rather
  than blaming insufficient history. `info --attribution` prints the IC report and
  then the attribution result instead of replacing the first with the second.
- **HTML metrics table**: an undeflated Sharpe is labelled `PSR, undeflated (1 trial)`
  as the terminal already did; percent, money, day and hour metrics carry their units
  (declared beside the gate-value table). `run_optimization` states that its rows'
  `deflated_sharpe_ratio` is not deflated.
- **A walk-forward with no out-of-sample trades** no longer passes the drawdown gate
  vacuously (it fails with a note), and the payload carries `no_oos_trades_note`.
- **Verdict and alpha output** says how many names a list cut, shows tiny alphas in
  scientific notation instead of `+0.0000`, carries the thin-universe reason in the
  payload (`low_confidence_reason`), and no longer journals a `trial:alpha` for a
  run that scored no names.

### Rename

- The distribution is `quantred` (was `tradeflow-engine`) and the main command is
  `quantred`. The `tradeflow` command remains as an alias; the import package, plugin
  groups, `TRADEFLOW_HOME` and `~/.tradeflow` are unchanged, so private packs and all
  recorded history carry over.
- MCP clients now see the server as `quantred`. Existing entries keep launching.
- **Upgrading in place:** uninstall `tradeflow-engine` first, then install `quantred`;
  in a shared virtual environment, the other order breaks it. A pack
  that still depends on `tradeflow-engine` needs installing without it. Steps,
  symptoms and recovery are in [migrating to QuantRed](usage/migrating-to-quantred).

### Getting started

- `init --check` says what is ready for which task (demo, data, research, MCP) instead
  of one pass/fail and names the missing capability. `init --check --for <task>` prints
  the single install command for that task, and the command that needs an extra prints
  it when it is missing.
- `init` can set up a model provider for autonomous research and test the connection
  (`--probe-model`); `.env` is written readable by you only.
- A new [extras guide](usage/extras) says which optional capability each command needs.
- Install hints match how you installed (uv tool, virtual environment, checkout),
  and keep the packs you added with `--with`.
- `--offline` no longer asks for broker keys: a cache-only run never reaches the
  vendor, and research commands no longer build a trading client at all. Without keys,
  an uncached window now says which bars are missing instead of asking for credentials.
- Common mistakes are refused up front with what to type instead: a reversed or
  future `--start`/`--end` (over MCP too, where a verdict over a reversed window used
  to be recorded as a trial), a mistyped `--scanner` (with a suggestion), a malformed
  date (the expected format), and a zero or negative `--capital` or `--max-positions`.
  Rejected keys end the command with their fix instead of a stack trace.

### Verdicts and examples

- A book that cannot be built under its own constraints is reported as **infeasible**
  with the binding constraint named, not as "needs more data". Too little history is
  still reported as a data problem.
- The documented verdict examples use a starter universe that is feasible with its
  defaults, and a test keeps every documented example that way.
- A verdict that came back **incomplete** (a step failed, for example an offline run
  before the cache was warm) is no longer served back as "reused" when you repeat the
  command after fixing the cause; it runs again. Complete verdicts are reused as before.
- An incomplete verdict in which no step measured an IC no longer counts as a trial, so
  a failed attempt does not raise the campaign's multiple-testing total. The report says
  `Not recorded as a trial` and why, and its `campaign trials` count leaves the run out.
- Deflated and probabilistic Sharpe and PBO print as probabilities with three decimals
  everywhere (terminal, chart, HTML report, `trials list/show/best`), and a very small one
  as `<0.001`. Before, the terminal printed `0.00` (which reads as a probability of zero),
  the chart `4.4e-04`, and a failing 0.496 printed the same as the 0.5 gate. An
  unrecorded metric on a reused walk-forward prints `—`, not `0`. A one-trial backtest
  labels its undeflated value `PSR, undeflated (1 trial)` - it is a probability, not the
  Sharpe ratio the old `Sharpe (undeflated)` label suggested.
- `quantred demo` prints its scorecard without the hundred progress lines above it, and
  an offline cache miss names `quantred cache warm`, not a script an installed copy
  does not have.
- An offline cache miss names every missing symbol (the benchmark included) and one
  `cache warm` command with the dates and the `--cache-dir` that cover the request that
  missed. Before, it named only the first symbol and no window, and the rerun failed on
  the next one. When `backtest`, `optimize`, `walkforward` or `screen` cannot even run
  its scanner from the cache, the one command covers the scan and the run for every
  candidate, so a single warm is enough; before, it missed on the scan's short lookback
  and then again on its own window. The same goes for reads past the request's own
  window: `walkforward` also reads an indicator warmup before `--start` and its
  `--benchmark`, and `backtest` its `--benchmark` (which offline used to drop alpha and
  beta with only a warning), and `backtest --beta-sizing` the trailing lookback before
  `--start`; a cold miss now asks for those too, the lookback only when the flag is on.
  `--causality`, `--cost-stress` and the walk-forward's `--leakage-probe`,
  `--param-sensitivity` and `--monte-carlo` were checked and read nothing beyond the
  run's own reads. Still a second round: a scanner on a different timeframe from the
  strategy (one warm covers one timeframe). The note about the
  market-date change appears only for the few-hour gap that change leaves (coverage
  recorded through UTC midnight, asked for through New York midnight), not on every miss
  or on any symbol that happens to have some coverage.
  An offline `verdict` whose steps miss prints one warm command covering every step,
  instead of one per step with a different window each.
- `quantred --help`, and the bare `quantred`, show the commands grouped by what you are
  trying to do - start here, evaluate an idea, history and data, agents, trading - with
  `init` and `demo` first. They were one long list in definition order with live trading
  near the top and every name repeated in a brace above it; the bare command was an
  error.
- The metrics glossary states which equity curve a result's risk figures come from: a
  backtest and each walk-forward fold are marked to market at bar closes, and only a
  walk-forward's aggregate across folds is rebuilt from closed-trade P&L. It used to say
  every curve was closed-trade, so a backtest's drawdown read as missing open-position
  losses it actually includes.
- `trials best` / `best_trials` no longer rank a plain backtest by default. It is one pass
  over its window, never validated out of sample, yet it was ranked under "validated runs
  only" with its full-window Sharpe in the out-of-sample column. `--include-in-sample`
  shows it, labelled.
- An HTML backtest report shows the execution-viability check the terminal prints, with
  a warning above the metrics when the book is not executable at its capital. A reused
  result, which does not carry the check, says it was not recorded.
- A backtest is reused only for the same capital, benchmark, beta sizing and take-profit
  margin, and a walk-forward only for the same capital and benchmark. None of these was
  part of what makes two runs "the same", so a run at $100,000 answered a request at
  $4,000. Backtests and walk-forwards recorded before this change are re-run once rather
  than reused.
- A memoized backtest, over the CLI or MCP,
  now carries `not_recorded` (total cost, execution checks, excursion, trades CSV,
  resolved config, and so on), `from_request` and `derived` lists, and how to recover
  the missing fields (`--force` / `force=true`). The HTML Run card shows "not
  recorded" instead of a blank or a zero, and `low_sample` is derived from the stored
  trade count. Both surfaces build the payload through one function.
- A verdict step that fails is reported in the verdict's step list; its traceback is
  logged only at debug level instead of filling the screen above the report.
- A request that Alpaca refuses for the credentials now says so ("Alpaca rejected these
  credentials", where the keys come from, and `quantred init`) in place of printing the
  gateway's raw HTML page.
- The MCP allocation tool exposes the construction knobs that are not gated on
  evidence, so an agent can state the same book the command line does.

### Trial history

- `trials list --since/--until` read a date as the New York market date and include
  the whole of it. Before, `--until` dropped trials recorded on the day it named and
  `--since` could admit the previous evening's. `trials mark-contaminated --before`
  had the same fault and could miss a trial recorded the evening before the cut-off.
- `trials list` and `trials show` print when a trial was recorded in New York time, the
  zone the date filters read, instead of its UTC text.
- Trade tables can be recorded over MCP: `run_backtest` and `run_walk_forward` take
  `record_trades`, the counterpart of `--record-trades`. `analyze_trial` answered "re-run
  it with --record-trades" to an agent that had no way to pass it. A prior identical trial
  that kept no table no longer answers a request for one, on either surface: the run is
  repeated and a new trial recorded, instead of reporting success and leaving the caller
  without the table. That trial counts toward the multiple-testing total; for a
  walk-forward it adds the whole inner search again, so ask on the first run.
- A unique trial-id prefix (four or more characters) works wherever an id is taken, on
  both surfaces; an ambiguous one is refused with the candidates. `trials status` on a
  fresh setup no longer reports the journal as unreadable, `compare` refuses a single
  id, `mark-contaminated --id` refuses an id that matches nothing (and `--id` beside a
  filter, which used to ignore the filter), and `archive --dry-run` names what it would
  move.

### MCP

- The `mcp` extra needs the SDK at 1.22 or later; earlier releases do not import
  under current pydantic.
- The server reports the application's version to a client, not the MCP SDK's.
- A date passed to any MCP tool is read in the exchange's time zone, exactly as the
  CLI reads `--start`/`--end`. Before, the same verdict over the CLI and then over MCP
  was recorded as two trials and the second was not served from the first, which
  counted one experiment twice toward the multiple-testing total. MCP-recorded trials
  from before this change will be re-run once rather than found.
- `list_strategies` and `list_scanners` work in a real client; they failed output
  validation before. `mcp --self-test` now exercises discovery over a real connection.
- Ctrl-C, closing the client and SIGTERM stop the server promptly with a one-line
  message; before, the first Ctrl-C could hang and repeated ones could abort with a
  traceback.
- The printed client configuration carries `TRADEFLOW_HOME` when you set it, and the
  self-test says when credentials came only from your shell.
- `run_walk_forward` returns the same promotion prerequisites, evidence completeness and
  per-fold excess return the CLI prints, in an `evidence` block beside `gate_report`,
  and echoes the `benchmark`. Before, an agent saw `promotable` with no sign that
  prerequisites existed or had failed. `promotable` is the statistical gates only;
  `evidence.clear` is the one to read. The tool takes `cost_stress` (on by default, as
  on the CLI); the family bootstrap is not run over MCP and stays unevaluated.
  `oos_aggregate` still carries no benchmark figures by design: the benchmark is scored
  per fold. The walk-forward and information HTML reports name the benchmark, timeframe
  and universe in their header and carry the prerequisites.
- The server starts without broker keys: catalogue and trial-history tools work, and a
  tool that needs bars says keys are missing and how to add them, where a client used to
  see only "Connection closed". `mcp --offline` serves from the local bar cache with no
  keys; the self-test and the printed client configuration carry it.
- Bad input to the trial and config tools (an unknown or ambiguous trial id, fewer than
  two trials to compare, an unknown config name) is a tool error naming the tool to use,
  instead of a successful response carrying an `error` field.
- `save_config` and `load_config` take a plain name and stay inside the configs
  directory (a name like `../x` wrote outside it), and an existing config is not
  replaced unless `overwrite=true`. `save_config` checks the strategy,
  scanner, parameters and limits before writing, so a config that could not run is
  never saved; an unknown provenance field lists the accepted ones.
- Arguments no run can mean are refused before anything is read or journaled, on both
  surfaces: an unknown `mode`, `method` or `objective` (an unknown objective scored every
  candidate equally, so the "best" was arbitrary; the search always maximises, so a
  lower-is-better objective such as `max_drawdown` picked the worst candidate and is now
  refused too), negative `holdout_days`,
  `max_evals` below one, a non-positive `capital`, an empty universe, a misspelt
  parameter name (it used to run at the default) and a position limit nothing enforces.
  A backtest that read no bars for any symbol is refused instead of reported as all
  zeros. Each of these used to cost a journaled trial.
- Walk-forward gates say what they are computed over (median across folds, total
  across folds, or the stitched out-of-sample aggregate), so a gate value no longer
  looks like it contradicts the aggregate metric of the same name.
- Calmar and Sterling ratios are negative for a losing run; they took the magnitude of
  the whole ratio, so a loss read as a reward. Values already stored for losing runs
  carry the old sign, so neither is accepted as a search objective for now (a search
  reuses stored per-candidate results); neither feeds a gate or a leaderboard.

### Model providers

- Provider failures are classified (authentication, credit exhausted, rate limited,
  model not found, unavailable, unreachable, timeout) and reported in plain words.
- Rate-limited, unavailable and unreachable calls are retried a bounded number of
  times; authentication, credit and model errors never are.

---

## 2.3.0 — 2026-09-16

Research affordances, the second half of the causality defect, and two telemetry modes
kept deliberately apart.

### The headline

**The same look-ahead again, one layer down.** The book was marked to a bar's close
before entries were considered, and the exposure caps are tested against `equity × cap` —
so whether an entry filling at that bar's *open* was admitted could depend on where the
bar finished, hours later. Same class as the v4 signal defect, in the admission gate
rather than the signal, and just as invisible to a feed shift.

It was found by the new causality probes rather than by a user, which is the point of
building them. Positions are now marked at the open for the decisions that transact
there and re-marked at the close for the equity curve. `ACCOUNTING_VERSION` is 5.

Narrower in effect than v4 — it changes a result only where a cap was close to binding,
and the default sizer is untouched because it sizes off cash, which is not marked. But
"usually identical" is not comparable. The promotion gates are deliberately **not**
recalibrated: unlike the mark-to-market change at v2, this moves no metric
systematically, and rescaling a threshold with no measured shift behind it is
gate-fitting.

**Two telemetry modes, kept apart on purpose.** Observing what execution actually does
used to require shrinking a book's caps by hand until fills happened, which biases the
sample toward low-priced names and means the strategy under observation is not the
strategy that was validated. That is now two separate answers to two separate questions.

`live --dry-run` drives the real decision path against a stated capital with a broker
that *cannot* trade, and reports what would have been sent. Its vocabulary is
deliberate — `would_submit`, `would_bind`, `would_skip`, never *submitted* or *filled* —
because a report that borrows execution's words gets read as execution's evidence. It
journals nothing: a rehearsal must not spend a family's statistical budget.

`small-real` trades the validated contract at **reduced capital** rather than reduced
caps. Fractions scale because capital did, counts stay counts, dollar ceilings scale,
and venue floors stay absolute — a broker's minimum does not get smaller because a run
chose to. The floor not scaling is a cost taken deliberately, and it is measured rather
than hidden: refusals carry reason codes so the bias is a number in the report. Shrink
far enough and it stops being a bias and becomes the whole result, so a scale at which
no order could clear the floor is refused outright.

### Added

- **`tradeflow screen`** — sweep a parameter space cheaply and journal nothing. One
  process, one data fetch, N evaluations. It leads with the distribution rather than a
  winner, because the best of N is the maximum of N draws: a positive number even when
  nothing searched has any edge, growing with N. Beside any best point it reports what
  the best of that many draws is worth under the null, refuses to compute one for an
  objective whose null is not zero, and shows how the result moves across each parameter
  — a coherent gradient being different evidence from scattered positive points.
  `--confirm` re-runs exactly one point as a real trial.
- **`backtest --causality`** — four probes asking whether each decision could have been
  made when it was made: the execution clock, same-bar ranking, benchmark alignment, and
  the scanner's as-of clock. A different class from the leakage probe, which tests for
  future data and cannot see a one-bar look-ahead at all. Each probe reports pass, fail,
  or *not exercised*, and the third is not the first.
- **`PARAM_CONSTRAINTS`** — relationships between parameters, declared beside the ranges
  so a sampler can read them. Enforced by construction: the grid never contains an
  invalid point and the random sampler never draws one, because an evaluated invalid
  combination is a journaled trial that raises the deflation bar for its whole family.
- **`trials archive` and `trials mark-contaminated`** — retiring an era and quarantining
  a suspect subset, deliberately not one command. Archive moves the journal and its index
  together; quarantine appends an event and rewrites nothing. Neither deletes anything.
- **`trials list`** now says how many rows the accounting filter is hiding. The day the
  version is bumped, an empty table would otherwise read as "nothing was ever run here".

- **`trials analyze` and `trials compare`** — the trade table, exit-reason split and
  excursion of a recorded trial, and two trials read side by side. The browser could
  list a run and not answer the first question anyone asks of one.
- **Portfolio excursion** — how far the *book* went against itself, not just each trade,
  so a drawdown can be attributed rather than only reported.
- **Campaign materialisation and export** — a validated config carries how it was
  produced as a field of its own provenance, never as a second file beside it: the
  recipe, the resolved book, the seed, the universe and the command that reads its
  trades back. Two artifacts claiming to say how a config was made is the failure this
  avoids.
- **`live --dry-run`** — what a contract would do, against a broker with no trading
  capability. See the headline.
- **`small-real`** — the validated contract at reduced capital, with a mandatory
  preflight that shows every limit beside the value it was validated at. Deliberately
  absent from MCP: it is the one command an agent must not reach.
- **MCP agent attachment** — the supported construction surface widened from nine
  parameters to the service's full ungated set, with `compute_attribution` added.
  Evidence-gated families stay off: conditional risk, the Black–Litterman posterior and
  the multi-period aim policy ship disabled because their adoption gates do not clear,
  and an agent reads a description as fact and acts on it at machine speed. Being
  reachable is not being validated.
- **`reconcile --small-real`** and **`flatten --ledger` / `--small-real`**, mirroring
  `execution-report`. Both commands now print the ledger they opened, the live default
  included — a reconciliation that does not say which book it compared reads as a
  verdict on *the* book, and there are two on purpose.

### Fixed

- **A saved config traded a book it was never validated at.** A config's
  `position_limits` and the limits folded into its trial's identity are the same book
  written twice and were not checked against each other. Omitting the first does not
  leave the book unspecified — it resolves to the strategy class default of one
  position, so a config whose own provenance recorded eight silently traded one. Four
  writers each lost it a different way.
- **`min_notional` was enforced in the backtest and absent from the live path.** A
  config validated with a venue floor traded without one, while the live preflight
  *printed* the floor on every run — the surface said the limit was in force and nothing
  applied it. One definition both clocks call, and both the sub-floor refusal and the
  size-rounds-to-zero one now carry reason codes.
- **`flatten` reported a submitted close as a closed position.** Exit zero meant the
  broker had accepted the request, not that anything was flat. It now requires a broker
  read that observed no positions and no resting orders, and reports `pending`
  otherwise — a queued close fills at the next open and a refused one never will, and
  nothing could tell them apart.
- **A flatten left the ledger permanently unreconcilable.** It goes straight to the
  broker by design, so the ledger never saw those exits and reported every flattened
  position as a possibly-dropped fill, for good. A confirmed-flat flatten now records a
  terminal event per position — no invented fill, no price, nothing rewritten.
- **A session header said a run happened when it never began.** The header is written
  when the contract is committed to, before anything can fill, which is what keeps a run
  that dies on its first bar from losing the capital its fills were measured against.
  It now carries an outcome as well; a header written before outcomes existed reads as
  *not recorded*, never inferred from the absence of a later row.
- **MCP silently discarded unknown arguments.** An agent passing a withheld knob got a
  normal successful result with no signal it was dropped. Every unknown argument is now
  refused with a did-you-mean suggestion.
- **A halted refusal grouped by its timestamped message**, so two halts counted as two
  refusal families — the fragmentation reason codes were introduced to fix for the
  exposure caps, surviving in the one refusal that never got a code.
- **A config this build cannot honour crashed instead of refusing.** Missing, not JSON,
  no strategy recorded, or naming a strategy that has since moved into a package that is
  not installed — all four raised a traceback out of commands that place orders.
- The long/short decomposition never rendered: `log_backtest_report` accepted `legs` and
  dropped it before the formatter, so the block was fully tested and dead at the surface.
- Execution checks were all rendered as percentages, so `book_breadth` — a position count
  — read as "a maximum of 1.00% positions", and its remedy line was hardcoded to say the
  value was 1 whatever it was.
- Importing the trial store created the state root, so an unwritable one made
  `tradeflow --help` fail with a traceback before argparse ran.
- A warm-up the data feed could not answer ended a live session with a raw broker
  traceback instead of the refusal written for exactly that situation.
- `--folds` defaulted differently on the CLI and in the service: same validation,
  different memoization key, so a walk-forward run on one surface was never found on the
  other.
- `costs/base.py` claimed to be research-clock only long after the live path began
  importing it.

### Durable records

`ACCOUNTING_VERSION` is **5** and `LEDGER_VERSION` is **4**. Both are stamped so a
reader can ask what shape it is looking at rather than inferring it from which keys
happen to be present — two compatibility decisions in this file were already improvised
per-field and each cost a live session before anyone noticed.

The ledger gained two event kinds: `session_outcome`, what became of a run that
committed to a contract, and `flattened`, a position an operator closed at the broker
outside the engine. Older files simply lack them, and that reads as exactly what it is.
A book flattened before the event existed still reconciles as missing, because inferring
a flatten from a divergence would excuse the dropped fill the divergence exists to
surface.

### A note on how these were found

Most of the fixes above came from running the software rather than reading it, and
several were found while attempting something else entirely. A venue-floor evidence
check that never completed — the universe produced no signal that day — nonetheless
surfaced four separate defects on the way to not completing. The recurring shape is a
green test suite sitting on top of a real break because the test exercised something
adjacent to the thing that ships.

## 2.2.0 — 2026-09-02

*Tagged and published on 2026-09-16. The version and this entry were written on the date
above and the tag was never pushed, so the release sat in the repository for two weeks
without reaching the package index. Tagged at the commit that set the version, so what
was published is what this entry describes rather than the month of work that followed.*

Live-path validation against a real paper account, and the defect it eventually found.
Everything here came from *running* the thing — the preflight, the ledger, the execution
report and finally a trade table each surfaced something the full test suite agreed was
fine.

### The headline

**The backtest transacted one bar before it could have known.** A signal at bar `i` comes
from scores computed on bar `i`'s *close*, and the engine executed it against bar `i`'s
*open* — for entries and signal exits alike. That is a one-bar look-ahead applied to
every trade in every result this project had recorded.

It survived because nothing could see it. A feed shift moves signal and price together,
so the leakage probe passed over it; and it made the backtest structurally impossible to
match live, where a closed bar produces a signal and a market order fills afterwards.
Three days of narrowing — a fill-assumption stress, a position-size cap, a per-trade
excursion study — each eliminated a hypothesis and left the anomaly intact, until the
question became "is this information causally available at all".

`ACCOUNTING_VERSION` is 4. **Every number recorded under 1–3 overstates what a deployment
could achieve**, and the trial store keeps the two apart. Causality is now asserted on
both clocks, in one file, because the two implement it separately and nothing in the
codebase connects them.

### Added

- **`tradeflow execution-report`** — what the live path actually did, reconstructed from
  the ledger: slippage in basis points, decision-to-fill latency, submitted-versus-filled
  notional, modelled cost beside observed fees, and refusals grouped by kind. Ungraded on
  purpose: what counts as bad slippage is not knowable from one session.
- **A live preflight**, printed on every run and exiting early under `--preflight`. It
  states the contract before any order logic: broker mode, account balance beside the
  capital this run may deploy, data feed, every book limit in the units it is enforced in,
  telemetry destinations, and how many symbols actually warmed up — and whether they
  warmed up *enough*, which is a different question.
- **`--fill-stress`** re-runs a backtest requiring the price to trade progressively
  further *through* each take-profit before it counts as filled. The default fills a
  target the moment a bar touches it, which models a resting limit always first in the
  queue; for a strategy whose gain concentrates in target exits, that assumption is the
  result rather than a detail.
- **Book limits on both clocks** — `--max-positions`, `--max-position-size`,
  `--max-gross-exposure`, `--max-net-exposure`, `--max-total-risk`, `--min-notional`.
  `max_net_exposure` is new and bounds directional tilt: gross bounds long + short and
  cannot see direction, so a book inside a gross cap can be entirely one-directional. A
  backtest of a long/short book now derives a cap from the tilt it actually carried, and
  says when the gross cap already subsumes it.
- **`--capital`**, so a run deploys what a config was validated at rather than whatever
  equity a paper account was handed. **`--feed`**, pinning the historical and streaming
  halves to one data feed. **`walkforward --config`**, so the run type that produces a
  config can consume one.
- Every backtest prints net P&L by exit reason, and says so when one winning exit carries
  over 90% of the gain.

### Fixed

- **The ledger counted fills twice and recorded every short as a long.** A venue
  re-reports an order's cumulative filled quantity on each partial fill and again on the
  final one, and those events were summed; separately the trade-update type had no `side`
  field, so a defaulted `"buy"` reached every record. An order that filled 8 arrived as
  21, and a 31-share short as +31.
- **A resumed session disagreed with its own ledger**, and a reconciliation sweep landing
  between an entry's submission and its fill could erase a position the strategy held —
  leaving it flat in a symbol it owned, and a strategy that believes it is flat cannot
  emit an exit.
- **Shutdown hung until a second interrupt, and `SIGTERM` was unhandled entirely.** Four
  passes, each correct and each still wrong one level up, ending at a synchronous SDK
  `stop()` that blocks the very loop that would have to run its close. A blocked loop
  defeats every loop-scheduled bound, signal handlers included, so the one that must
  always hold now lives on a daemon thread.
- **Blocking broker calls ran on the event loop**, stalling other symbols' bars, fill
  delivery and reconciliation for the length of every entry. They run in a worker thread
  now, with the order path serialized behind one semaphore — a correctness requirement,
  not a speed-up.
- **A failed market-data fetch returned "no bars" instead of raising**, so an unreachable
  provider produced a zero-trade backtest that was journaled as an evaluated trial and
  then served from cache to the next identical run. Absent is not zero, and unreachable
  is not absent.
- **Risk-limit flags never reached the live book**, and a flag that cannot reach anything
  now stops the run instead of being silently ignored.
- Position limits went through no cache identity at all, so two runs differing only in
  `max_gross_exposure` hashed alike and the second was answered from the first.

---

## 2.1.0 — 2026-08-10

Trade-clock hardening. The live path is the smallest and least-tested part of this
project, which is backwards — it is the only part that can lose money. Most of what
follows was found by *running* it rather than reading it.

### Fixed

- **The live path could open a position but never close one.** A strategy decides
  whether an exit is legitimate by looking itself up in its own position book, and
  nothing in live mode ever wrote that book — so every `CLOSE_BUY`/`CLOSE_SELL` was
  rewritten to `HOLD` before execution saw it, and a position could only be closed by
  its broker-side bracket legs, if it had any. The book is now rebuilt from broker
  truth: before the first bar, on every sweep, and when an entry is placed. The unit
  tests had been passing because they fed `handle_signal` directly, exercising the
  layer below the break.
- **A strategy could go permanently silent with nothing logged.** Warm-up history is
  timezone-aware and a feed may stream naive timestamps; mixing them made every later
  comparison raise inside `process_bar`, whose blanket `except` turned that into a
  strategy that emitted nothing at all, indefinitely. Timestamps are aligned before
  append, and the `except` now says what it swallowed.
- **Warm-up fetched a fraction of the history it asked for.** The lookback converted
  bars to wall-clock time directly, which counts the overnight gap, the weekend and
  every holiday as tradeable: fifty one-minute bars became "100 minutes ago", so a
  09:35 start warmed a fifty-bar indicator with five bars and ran anyway. Daily
  under-fetched too, less visibly. The window is now measured in sessions, and a
  short warm-up is reported rather than absorbed.
- **Missed signal edges were lost for good.** Entries fire on the crossing bar and
  never again, so a bar rejected by a quality guard, a dropped stream, or a restart
  left the score saying "should be long" while every bar emitted `HOLD`. Live mode
  now compares the direction the score implies against the position book and
  re-states the difference. Exits always; entries under `reaffirm_entries`, on by
  default.

### Added

- **A kill switch.** `tradeflow halt | resume | halts` records a durable decision to
  stop, and `tradeflow flatten --confirm` halts, cancels every order, and closes
  every position — going straight to the broker, so it works when the engine is
  wedged. Halt state is a file, not a database: the order path must hold no
  connection to anything that can be down. Halts block entries and never exits, so
  the switch cannot trap the book and flatten cannot deadlock against its own gate.
- **Deterministic client order ids.** The guard against double-submitting was a
  question asked of the broker immediately before placing an order — a check-then-act
  race with no memory across a restart. Each order now carries an id derived from the
  decision behind it, so a venue that has already accepted it rejects the duplicate.
- **Typed broker failures.** Every broker call used to fail as `None` or `False`, so
  a rate limit, an expired token, insufficient funds and a deliberate rejection all
  produced the same non-answer. Two cases were actively wrong: a duplicate order read
  as a failed submission, and `list_positions` returning `[]` — the claim that the
  account is flat — when the broker could not be reached.
- **A decision record for every signal.** "No order" had a dozen causes and one
  representation. Execution now returns what it decided, why, and which guards it
  consulted — the guards that *ran*, not just the one that fired — and the ledger
  records the declines, which are the case that leaves no other trace.

### Changed

- **An unreadable market clock no longer uniformly means "assume open".** That is
  right for a transient blip, but it was also being applied to revoked credentials,
  which are never transient. Authentication failures now fail closed.
- **A strategy started while the score already implies a position takes it** on the
  first live bar, rather than waiting for the next crossing. Turn it off with
  `--no-reaffirm-entries`.
- **Internal trade-clock interfaces changed shape.** `Broker.submit_*` take a
  `client_order_id`; the order-path methods raise `BrokerError` rather than
  returning `None`/`False`; `LiveTrader.handle_signal` returns a `Decision` rather
  than an optional order. No supported API moved — `tradeflow.services.*` is the
  supported surface and is untouched — but anyone who had reached past it into the
  broker or execution layer will need these. Hence a minor, not a major.

## 2.0.4 — 2026-08-07

### Fixed

- **`trials best` ranked in-sample search results as if they were track records.**
  An `optimize` row is the winner of a parameter search — best-of-N by construction,
  which is the selection bias the whole evaluation stack exists to correct for — and
  the leaderboard ranked them alongside validated results without even showing the
  kind. On a real campaign, four of the top five rows were search artifacts and
  nothing said so. In-sample kinds are now excluded by default (with the count of
  exclusions reported, and `--include-in-sample` to opt back in), and every row names
  its kind. Found by using the tool rather than reading it.

### Added

- `CLAUDE.md` and `.claude/rules/` — the conventions this project is actually built
  to, as path-scoped rules that load only when a matching file is touched: the
  two-clocks invariant, the honesty rules for anything the tool prints, the trade
  clock's reject-never-repair discipline, and the verify-don't-assert habit that has
  caught most of the real defects here.

## 2.0.3 — 2026-08-07

Found by walking the getting-started flow end to end against real paper credentials,
using the published package rather than a checkout.

### Fixed

- **`tradeflow-engine[mcp]` installed a broken server.** The MCP SDK released 2.0.0,
  which removed the server class this integration is built on, and the dependency was
  an unconstrained `mcp>=1.0` — so a fresh install resolved to a version that could
  not be imported. It was invisible locally because the lockfile held a working 1.x.
  Now constrained to `<2`, with tests that check the *resolved* SDK rather than
  trusting the lockfile.
- **`init --check` suggested `uv sync --extra …`**, which an installed copy cannot
  run — the third place this same checkout-only assumption turned up.
- **`PAPER_TRADE` was masked as `****`.** It is not a secret, and hiding it cost the
  reader the one value they most want to confirm.
- **Reports said `git unknown`** when run from an installed copy. A provenance block
  exists so a result can say what produced it; it now falls back to the package
  version, which is the honest answer when there is no repository.

### Changed

- **The documentation site now deploys itself** on every push to `main`. It was
  published by hand, so it silently fell behind the repository — a page could be
  written, reviewed, merged, and linked from the README while 404ing for every
  reader.
- **The home page leads with `uv tool install tradeflow-engine` and `tradeflow
  demo`**, rather than the clone-and-`uv sync` instructions it still carried from
  before the package existed.

## 2.0.2 — 2026-08-07

Onboarding fixes, all found by installing the published package and following the
path a new user would.

- **The missing-credentials message sent installed users to dead ends.** It told them
  to copy a `.env.example` they do not have and run a `make` target that does not
  exist, and never mentioned `tradeflow init` — the one command that fixes it. It is
  now phrased for the copy that is running, and names the exact path the `.env` goes
  to.
- **`tradeflow mcp` crashed with a traceback** when the `mcp` extra was not
  installed, instead of saying so. The guard wrapped only the import, but the SDK is
  pulled in lazily further down, so the real failure escaped — at exactly the moment
  someone is first trying to connect Claude. The install hint it prints is now
  correct for an installed copy too.
- **Added [Getting started](usage/getting-started)** — one walkthrough from install
  to a real result with Claude connected, in six steps, replacing a set of correct
  pages with no thread between them.
- The README's MCP registration snippet assumed a checkout; it now shows the
  installed form first, and its tool list is no longer two releases stale.

## 2.0.1 — 2026-08-07

- **Fixed the broken screenshot on the PyPI project page.** The README referenced the
  demo image by a repository-relative path, which GitHub resolves and PyPI cannot —
  PyPI renders a README standalone, with no repository context. It is now an absolute
  URL, which works in both places.
- Bumped the GitHub Actions used by CI and the release workflow off Node 20, which
  GitHub is deprecating.

No functional changes.

---

## 2.0.0 — 2026-08-06

The first release published to PyPI, as **`tradeflow-engine`**. Also the release that
made the research machinery usable by someone who did not write it: before this,
every capability existed but assumed you had the repository, the flags, and the
context in your head.

### Breaking

- **The package is now `tradeflow`, not `src`.** Every import moves:
  `from src.services.analysis import …` → `from tradeflow.services.analysis import …`.
  A package named `src` cannot be installed — it would collide with every other
  project that shipped one — so this was the precondition for distributing anything
  at all.
- **The distribution is `tradeflow-engine`.** The bare `tradeflow` name on PyPI
  belongs to an unrelated project that also imports as `tradeflow`; installing both
  into one environment would collide. The command and the importable package remain
  `tradeflow`.
- **The Docker image no longer defaults to `live`.** `docker run tradeflow` with no
  arguments used to start a paper-trading loop; it now prints help. Turning the
  machine on must not turn trading on.

### Added

- **`tradeflow` as an installed command.** `uv tool install tradeflow-engine`, no
  clone required. State resolves to `TRADEFLOW_HOME`, else a checkout, else
  `~/.tradeflow`; `tradeflow --version` and `init --check` both print which copy is
  running and where its state lives.
- **`verdict`** — the whole cross-sectional pipeline as one command over one
  universe, one window, and one cost model, ending in a single gate-derived verdict
  with every check shown. Running the steps by hand produced reports that looked
  joined-up while each re-resolved its own universe.
- **`init`** — guided first-run setup: hidden prompts, fully masked output, credential
  validation through the data-only client, and `--check`, a doctor that writes
  nothing and makes no network call.
- **`--html`** on `verdict`, `backtest`, `walkforward`, and `info` — one
  self-contained report file that issues zero network requests when opened, with
  provenance as a mandatory header and honesty labels as banners rather than
  footnotes.
- **`trials list` / `show` / `best`** — the campaign's memory, browsable: filters,
  SQL-level paging, per-trial detail (including which later trials reused it), and a
  leaderboard ranked by *deflated* Sharpe with every row's family trial count
  attached.
- **`--workers N`** on `optimize` and `walkforward` — parallel candidate evaluation
  with the single-writer invariant intact. Measured 1.85× on four workers, with an
  identical winner and an identical trial count.
- **MCP research surface completed** — `run_verdict`, `render_report`, and read-only
  trial-store tools, plus an audit of every tool description against current
  behavior, anchored to the shared metric glossary.
- **Live-path hardening** — bar-quality guards that reject (never repair) a bad bar,
  an append-only position ledger reconciled against the broker, a `reconcile` verb,
  and the loop-level test fence the live engine never had.
- **`docker compose`** for a local development stack, with state on named volumes
  that survives container replacement.
- **`tradeflow.services.*` documented as the supported library API** for embedding.

### Fixed

- An HTML-escaping hole in the report renderer: a symbol list could inject script
  through the provenance header.
- A crash in the live loop when a data feed switched timestamp awareness mid-stream.
- Optimizer ranking was not a total order, so a tie's winner depended on evaluation
  order — in the sequential path too, not only under parallelism.
- `construct_portfolio` hardcoded its cost model, so cost flags could never reach it.
- The engineering docs claimed the portfolio tool over MCP solved a cost-blind book;
  it had been cost-aware since the default changed.

### Notes

- The compatibility posture remains **clean-slate**. Live-path hardening is the
  *precondition* for trading real capital, built ahead of that decision; the decision
  itself is a separate, deliberate change.
- Every research run journals a trial and counts toward the campaign's
  multiple-testing total. Embedding this in a loop that runs thousands of backtests
  will correctly make every later result harder to clear.

---

## 1.0.1 — 2026-08-05

- **The Alpaca SDK is confined to the broker layer**, for real. A broker factory
  (`build_broker` / `build_market_data`) means entry points and services construct
  vendor-backed objects through the abstraction instead of importing the SDK
  directly — making the architecture page's "the only place `import alpaca` appears"
  claim true rather than aspirational.
- Regenerated the dependency lockfile, which still recorded a stale version.

---

## 1.0.0 — 2026-08-05

The portfolio layer, the research infrastructure that makes a campaign's history
count, and an honest capital model.

### Portfolio construction

- **Benchmark as a held portfolio** — active weights, true tracking error in active
  space, alpha neutralized against the benchmark, and reverse optimization for the
  consensus returns that make the benchmark itself optimal.
- **Black–Litterman posterior** blending consensus with our views, so coverage holes
  get a real propagated posterior instead of an implicit zero-view. Defaults off
  until validated out-of-sample.
- **Long/short books** — market-neutral construction with a mandatory gross-leverage
  cap and short-side borrow carry, plus a report pricing what the long-only
  constraint actually costs.
- **Multi-period trading policy** — aim in front of the target: alphas discounted by
  their measured decay, then a partial adjustment composed with the cost-aware
  no-trade band. Defaults off; the net-of-cost A/B does not clear on this
  repository's own demo data, which is a legitimate outcome rather than unfinished
  work.

### Risk and analytics

- **Regime-conditioned volatilities** on slow correlations, with a built-in
  predictive-accuracy gate and a net-of-cost A/B that decide adoption rather than
  preference. Default off — the gate does not clear on the demo data.
- **Return attribution** splitting realized active return into benchmark timing,
  risk factors, signals, and stock-picking, by an exact regression identity.
- **Bootstrap inference** — a block bootstrap for a single track record's own
  zero-alpha null, and a joint reality check across every trial ever run.

### Research infrastructure

- **The trial store.** A SQLite index over the research journal, so the
  multiple-testing correction counts every configuration tried across a whole
  campaign rather than resetting each session. Identical trials are served from it
  instead of re-run, unmistakably labeled with the original run's age.
- **A local bar cache** with gap-filling and an offline mode, so a repeated study
  reuses data instead of re-fetching it.
- **`demo-agent`** — a narrated research session on real market data, deterministic
  by default, with the LLM proposer able to author strategy code that a sandbox
  admits or rejects.

### The capital model

**One clock, one capital pool.** Every symbol now simulates on a merged timeline
against shared cash with portfolio-level position limits and a per-bar
mark-to-market equity curve. Previously each symbol ran its own full-capital
backtest and the results were summed — which meant return and Sharpe scaled with
universe size, an artifact rather than a finding. Transaction costs are charged
through optimization and validation, and an unrunnable backtest is now
distinguishable from a strategy with no edge.

---

## 0.3.0 — 2026-07-03

The active-management spine: the classical quantitative-equity framework, built end
to end on a point-in-time data substrate.

- **Continuous alphas.** A per-name signal becomes a cross-sectionally comparable
  residual-return forecast (`α = σ·IC·z`) through a pure pipeline — winsorize,
  standardize, neutralize, scale, cap. Strategies migrated to score-first: each
  defines one continuous conviction score, and both the discrete signal and the
  alpha derive from it.
- **The feature panel** — a cross-sectional, point-in-time data spine with an
  as-of seam that everything above runs on, so leakage is prevented structurally
  rather than by discipline.
- **Risk models** — Ledoit–Wolf shrinkage, then a structural factor model with a
  factor/specific split.
- **Transaction costs** as a first-class model (commission, spread, square-root
  impact, borrow carry), with backtests net of cost by default.
- **Mean-variance portfolio construction** in pure numpy, with the transfer
  coefficient and risk aversion calibrated to a target tracking error — then the
  cost term moved *inside* the objective, so a no-trade band emerges from the cost
  itself.
- **Information analysis** — measured information coefficient, effective breadth
  deflated by average correlation, and predicted-versus-realized information ratio,
  with the research-integrity guardrails that keep a lucky backtest honest.
- **Multi-signal combination** by information coefficient and correlation, so
  redundant signals split a weight rather than double-counting.
- **Information horizon** — signal decay and half-life, driving rebalance cadence.
- **Out-of-core storage and lazy compute** — a date-partitioned columnar bar store
  behind the as-of seam, with streaming covariance bounded by universe size rather
  than by history length.
- **Forecast refinement** — score-scaling case selection, an
  estimation-uncertainty haircut on the alpha level, and an equal-risk diagnostic.

---

## 0.2.0 — 2026-06-24

- **Surfaced the two-clocks model.** The research-clock / trade-clock split is the
  single idea that makes the safety story cohere, and it was buried. Promoted, with
  a diagram, to the top of the README and the architecture documentation.
- Quickstarts now lead with `make demo` — try it before signing up for anything.

---

## 0.1.0 — 2026-06-19

The rebuild into a layered, broker-agnostic engine — and a deliberate discarding of
most of what came before it (see below). The earlier code could place orders; it had
no way to tell you whether it should.

- **Strict separation of concerns** across brokers, market data, indicators,
  strategies, scanners, execution, analytics, engine, optimization, portfolio, and
  utilities — with dependencies pointing downward and the engine orchestrating
  rather than owning.
- **Broker and market-data interfaces**, isolating vendor SDK usage inside adapters
  so another venue means writing one adapter and nothing else.
- **No TA-Lib.** Indicators are pure pandas and numpy, so installation needs no
  compiler and the Docker image carries no build toolchain.
- **Honest evaluation from the start** — risk, tail, and trade-level metrics
  including the probabilistic and deflated Sharpe ratios; walk-forward validation
  with a sacred holdout and promotion gates; an MCP server that is structurally
  incapable of trading; and an autonomous research loop where only candidates that
  survive out-of-sample, leakage-checked, gated validation ever reach a human.

---

## Before the rebuild — 2023 to 2024

The first two and a half years, kept here because the shape of the project came out
of what these versions got wrong.

**June to October 2023 — the original.** Market-data endpoints, order placement, the
first attempts at a scanner, and a README rewritten more times than the code. It
could talk to a broker and do something in response to a price. What it could not do
was answer whether that something was a good idea, which turned out to be the entire
problem.

**October 2024 — the streaming rewrite.** Live market-data and account-action
streams, websocket lifecycle handling that closed cleanly on exit, subscription
management, a file watchdog for live reloads, and paper trading moved behind
configuration rather than a code edit. This is where the live path grew up, and
several of its ideas survived the rebuild intact — the streaming reconnect logic in
particular.

**What did not survive:** everything above the venue. There was no separation between
deciding and doing, no way to test a strategy without a network, and no evaluation at
all — so every result was in-sample by construction, and nothing could distinguish a
real edge from a lucky window. The 2026 rebuild kept the broker abstraction's
lessons, threw out the rest, and started from the question the earlier versions could
not answer.
