"""The guard that fails a test for leaving the kill switch thrown.

Every case here runs a **nested pytest session in its own process**, because the thing
under test is a session-level hook and the assertions are about which test failed - not
something the running session can say about itself. The nested session points
``TRADEFLOW_HOME`` at its own directory, so a leak staged here is invisible to the suite
that stages it.

Two of these tests are mutations rather than requirements: they install a plausibly
wrong guard and pin the case that separates it from the right one. Without them, an
implementation that merely checked whether ``halts.json`` exists, or one that reported a
leak without clearing it, would satisfy everything else in this file.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The real thing, loaded the same way `tests/conftest.py` loads it - by module, so this
#: file cannot drift from what the suite actually runs by restating the hook.
REAL_GUARD = 'pytest_plugins = ["tests.state_leak_guard"]'

#: Wrong guard 1: presence of the file taken for presence of a halt. `clear()` rewrites
#: rather than unlinks, so a cleared halt leaves `{}` on disk and this refuses a test
#: that tidied up correctly.
EXISTENCE_GUARD = """
import pytest

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    from tradeflow.execution.halt import default_halt_path

    if default_halt_path().exists():
        raise AssertionError("halt state left behind")
"""

#: Wrong guard 2: correct detection, no clearing. Blames the right test and then lets
#: the halt stand, so the cascade it exists to prevent happens anyway.
NO_CLEAR_GUARD = """
import pytest

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    from tradeflow.execution.halt import HaltState

    leaked = HaltState().list()
    if leaked:
        raise AssertionError("halt state left behind: %s" % leaked)
"""

_CONFTEST = """
import os
import tempfile

{guard}

def pytest_configure(config):
    # The same one-root-per-session shape the real suite uses, so the nested run
    # reproduces the condition under test instead of a tidier version of it.
    os.environ["TRADEFLOW_HOME"] = tempfile.mkdtemp(prefix="guard-under-test-")
"""

#: A test that writes to the shared root, followed by one that asserts the root is clean.
#: The second is what makes this a test of the *cascade*, not only of the blame.
LEAK_THEN_OBSERVE = """
def test_leaves_a_halt_set():
    from tradeflow.execution.halt import HaltState

    HaltState().set("rehearsal", actor="the-leaking-test", scope="all")


def test_runs_under_a_clean_root():
    from tradeflow.execution.halt import HaltState

    assert HaltState().list() == []
"""


@pytest.fixture
def guarded(pytester, monkeypatch):
    """Run a nested session under a chosen guard, with this checkout importable."""
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))

    def run(body, guard=REAL_GUARD):
        pytester.makeconftest(_CONFTEST.format(guard=guard))
        pytester.makepyfile(body)
        return pytester.runpytest_subprocess()

    return run


def test_a_test_that_leaves_a_halt_set_fails_and_the_next_test_does_not(guarded):
    """The whole contract in one run: the leaker is blamed, and the test after it still
    sees a clean root. Either half alone is satisfiable by a wrong guard — blaming
    without clearing leaves the cascade, and clearing without blaming loses the defect."""
    result = guarded(LEAK_THEN_OBSERVE)

    # The leaker's *body* passes — nothing it asserted was wrong — and the blame arrives
    # at teardown, which is why `errors` rather than `failed` is the count that moves.
    # `failed=0` is the load-bearing half: no innocent test was harmed.
    result.assert_outcomes(passed=2, failed=0, errors=1)
    assert any(line.startswith("ERROR") and "test_leaves_a_halt_set" in line for line in result.outlines), (
        "the error was not attributed to the test that caused it"
    )


def test_a_halt_on_a_path_the_test_owns_is_ignored(guarded):
    """The boundary case. Tests whose subject is halt behaviour build their own state on
    a temporary path; a guard that failed those would be indistinguishable from one that
    refuses everything, and would have to be switched off to write a halt test at all."""
    result = guarded(
        """
        def test_halts_on_its_own_path(tmp_path):
            from tradeflow.execution.halt import HaltState

            state = HaltState(tmp_path / "halts.json")
            state.set("rehearsal", actor="a-test-that-owns-its-state")
            assert state.is_halted()
        """
    )

    result.assert_outcomes(passed=1)


def test_a_test_that_sets_and_clears_a_halt_passes(guarded):
    """The other boundary: the guard must distinguish *was* halted from *is* halted."""
    result = guarded(
        """
        def test_sets_then_clears():
            from tradeflow.execution.halt import HaltState

            state = HaltState()
            state.set("rehearsal", actor="a-tidy-test")
            assert state.clear()
        """
    )

    result.assert_outcomes(passed=1)


def test_the_failure_names_the_halt_rather_than_only_its_existence(guarded):
    """A message reading "durable state was left behind" sends the reader back to the
    bisect this guard exists to remove. Asserted against the rendered output, which is
    what a reader actually gets, rather than against a payload nobody sees."""
    result = guarded(LEAK_THEN_OBSERVE)

    output = "\n".join(result.outlines)
    assert "rehearsal" in output, "the reason is missing"
    assert "the-leaking-test" in output, "the actor is missing"
    assert "[all]" in output, "the scope is missing"
    # `set_at` is a UTC timestamp; asserting the year keeps this from pinning a clock.
    assert "T" in output and "+00:00" in output, "the time it was set is missing"


def test_a_strategy_scoped_halt_is_a_leak_too(guarded):
    """`active()` with no scope answers only about the global halt, so a guard built on
    it would let a strategy-scoped leak straight through — while that halt still refuses
    every later entry for the strategy it names. Reading every recorded halt is the
    difference, and this is the case that shows it."""
    result = guarded(
        """
        def test_leaves_one_strategy_halted():
            from tradeflow.execution.halt import HaltState

            HaltState().set("rehearsal", actor="the-leaking-test", scope="breakout")


        def test_runs_under_a_clean_root():
            from tradeflow.execution.halt import HaltState

            assert HaltState().list() == []
            assert not HaltState().is_halted("breakout")
        """
    )

    # Same shape as the global case: blamed at teardown, and the scoped halt cleared
    # rather than left for whatever runs that strategy next.
    result.assert_outcomes(passed=2, failed=0, errors=1)
    assert any(
        line.startswith("ERROR") and "test_leaves_one_strategy_halted" in line for line in result.outlines
    ), "a scoped halt was not treated as a leak"


def test_a_guard_keyed_on_the_file_existing_refuses_a_test_that_tidied_up(guarded):
    """Mutation. `clear()` rewrites the file rather than unlinking it, so a cleared halt
    leaves `{}` on disk — meaning "the file exists" and "a halt is in force" are
    genuinely different questions, and the cheap implementation answers the wrong one."""
    body = """
        def test_sets_then_clears():
            from tradeflow.execution.halt import HaltState

            state = HaltState()
            state.set("rehearsal", actor="a-tidy-test")
            assert state.clear()
        """
    assert guarded(body, guard=EXISTENCE_GUARD).ret != 0, (
        "the existence check passed this, so it is not separated from the real guard"
    )
    assert guarded(body).ret == 0


def test_a_guard_that_reports_without_clearing_lets_the_cascade_happen(guarded):
    """Mutation, and the half most likely to be lost in a later refactor: nothing
    user-facing depends on the clearing, so it reads like tidiness rather than the
    mechanism that keeps one bad test to one failure."""
    leaky = guarded(LEAK_THEN_OBSERVE, guard=NO_CLEAR_GUARD)
    # Both tests are now implicated: the leaker errors, and the innocent test that
    # follows it fails on a halt it never set — which is the original defect.
    leaky.assert_outcomes(passed=1, failed=1, errors=2)

    # The real guard, same scenario: the innocent test still passes, and the only
    # difference between these two lines is the one thing the mutation removed.
    guarded(LEAK_THEN_OBSERVE).assert_outcomes(passed=2, failed=0, errors=1)
