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
  ran yesterday.
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
- **Read the schema before writing against it.** Field names guessed from memory
  compile, pass, and report a confident wrong answer — an invented cost key made the
  model silently unconfigured, and `trades` instead of `total_trades` reported 0 for a
  1952-trade run. Print the real object once; it costs a line.
- **A parity point needs a test that compares the two.** Two tests that each pass do not
  establish that two implementations agree — a parity bug looks exactly like two green
  tests. Build the same thing both ways and assert equality. See
  [parity points](parity-points.md) for the list of places this applies.
