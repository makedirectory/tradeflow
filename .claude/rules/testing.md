---
paths:
  - "tests/**/*.py"
---

# Testing rules

- **Offline and deterministic, always.** Everything runs through `tests/fakes.py` —
  no API keys, no network, no clock dependence. A test that needs the real world
  belongs in a script recorded in the PR, not in the suite.
- **A fixture that agrees with the test proves nothing.** A checkout-detection test
  wrote its own `pyproject.toml` and kept passing after the real one was renamed.
  Build fixtures from the same constant the code reads.
- **Test the resolved environment, not the declaration.** A lockfile agreeing with
  itself is how a broken dependency shipped. Assert the version actually installed.
- **Isolate state.** Point `ARTIFACT_DIR`, the journal, and `TRADEFLOW_HOME` at
  `tmp_path`; a test that memoizes against the real journal depends on what someone
  ran yesterday. The state root is deliberately *one directory for the whole session*
  — module-level path constants make a per-test root unsafe — so anything written
  there outlives the test that wrote it. Most of it is a record and harms nothing;
  **halt state is not**, because the live path consults it before every entry. A test
  that builds `HaltState()` with no argument writes to the shared root and changes the
  rules for everything after it, which once cost two dozen failures in two unrelated
  files, none reproducible alone. `tests/state_leak_guard.py` now clears such a leak
  and fails the test that caused it; build `HaltState(tmp_path / "halts.json")` and it
  never comes up.
- **Name the property, not the mechanics.** `test_a_rejected_bar_does_not_become_the_baseline`
  says what breaks if it fails; `test_check_2` does not.
- **Every bug found in review gets a regression test**, and the docstring says what
  the failure actually was.
- **Cover both directions.** A guard that rejects the bad case must also accept the
  boundary case, or it is indistinguishable from one that rejects everything.
- **Verify the test fails without the fix.** A test written after the fact usually
  passes either way, and then proves only that the code runs. Revert the change, watch
  it fail, restore it. Repeatedly in this project a green test has sat over a real
  break: a total that was 20 whichever arrangement ran, a shutdown that "returned"
  because an outer timeout cancelled the wait, sequential awaits that pass whatever a
  lock does. If reverting is awkward, that is a signal the test is asserting the wrong
  thing.
- **A test must fail when the property is violated on the surface a user actually
  reaches.** The strongest version of the rule above, and the one the last four changes
  each taught in a different outfit. Every time, a green suite sat on top of a real
  break because the test exercised something adjacent to the thing that ships:

  - a guard installed on a convenience method while every real client went through a
    protocol handler captured earlier — the wrapper existed, and nothing reached it;
  - a scaled book computed correctly and never applied, because the test asserted the
    *recorded* contract rather than the strategy the engine was handed;
  - a report whose formatter dropped a whole vocabulary the payload carried, because
    the tests asserted the JSON and never the rendered text;
  - a rejected argument that was unknown to *every* tool, which proves the global
    reject path and never that the schema consulted belonged to the tool being called.

  So: drive the real path, read the artifact back, and pick inputs that can tell the
  right implementation from a plausibly-wrong one. "Does the field exist" is rarely
  that test; "does a wrong implementation fail" always is.

- **Read the schema before writing against it.** Field names guessed from memory
  compile, pass, and report a confident wrong answer — an invented cost key made the
  model silently unconfigured, and `trades` instead of `total_trades` reported 0 for a
  1952-trade run. Print the real object once; it costs a line.
- **A parity point needs a test that compares the two.** Two tests that each pass do not
  establish that two implementations agree — a parity bug looks exactly like two green
  tests. Build the same thing both ways and assert equality. See
  [parity points](parity-points.md) for the list of places this applies.

## The math contract (`tests/math/`)

Every formula that produces a number someone reads has a reference test, and the
end-to-end numbers are pinned. Together they are what lets one fix land without quietly
moving another.

- **Reference tests** (`test_returns_and_ratios.py`, `test_tail_risk.py`,
  `test_multiple_testing.py`, `test_risk_models.py`, `test_portfolio_math.py`,
  `test_costs_math.py`, `test_information_math.py`, `test_engine_accounting.py`,
  `test_sizing_math.py`, `test_walk_forward_math.py`) compare the code with a computation
  written **in the test**: a hand-derived literal with its derivation in a comment, a plain
  loop, `scipy`, or a brute-force search. A reference never calls another function of the
  package - a fix that bends a formula would bend its own check. Existing one-mechanic
  tests stay where they are and are named in each module's docstring.
- **Each defect gets its input.** The docstring says which defect a test guards, and the
  input is one a plausibly-wrong implementation fails (a gap through a stop, a book that
  holds its own benchmark, a cap that binds). Properties and identities (`tc * IR*` is
  the predicted IR, Σ trade P&L is the book change, a blend beats a grid) sit beside the
  known answers. Degenerate inputs assert the documented "unavailable" value, never a
  silent `NaN` or `inf` in a printed metric.
- **Golden pins** (`test_golden_pins.py`, files in `tests/math/golden/`) store the full
  numeric output of a few seeded scenarios. Each scenario is first checked against the
  reference identities, so a pin never freezes a wrong number. Tolerance is relative
  `1e-9` / absolute `1e-12`; a looser one (an iterative solver) is declared in the test
  with its reason.
- **A pin that moves is a question, not a nuisance.** The failure lists every changed key
  as `old -> new`. Regenerate only after confirming the new value is *correct* and, if
  engine output changed, bumping `ACCOUNTING_VERSION`:
  `QUANTRED_REGEN_GOLDEN=1 uv run --extra dev pytest tests/math -k golden`. Each file
  records the version it was generated under, and a mismatch fails until you regenerate
  on purpose. Never regenerate to make a red run green; read the diff first.
- **Known defects are `xfail(strict=True)`** with the reason, so fixing one turns the test
  red until the marker is removed.
