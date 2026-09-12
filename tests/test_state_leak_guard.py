"""The guard that fails a test for leaving the kill switch thrown.

Most cases here run a **nested pytest session in its own process**, because the thing
under test is a session-level hook and the assertions are about which test failed - not
something the running session can say about itself. The nested session points
``TRADEFLOW_HOME`` at its own directory, so a leak staged here is invisible to the suite
that stages it.

Two things this file learned the hard way, both from review:

**Assert against the guard's own line, not the whole output.** ``halt.py`` logs
``HALT SET [all] ...`` and ``Halt cleared: ...`` at warning level, and pytest captures
both into the same failure report. A test reading the rendered output for a scope, a
reason and an actor therefore passed against a guard whose entire message had been
replaced with "durable state was left behind" - the strings it asserted were coming from
the logger, and the guard's message was never consulted. Every such assertion now matches
an ``E``-prefixed line, which only the assertion text produces.

**One case has to test the real session.** The nested sessions prove the module behaves;
they say nothing about whether the suite that ships loads it. It did not, at first - the
registration could be deleted with every test here still green.
"""

from pathlib import Path

import pytest

from tests import state_leak_guard as _guard

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Taken from the module object rather than spelled out, so the dotted path this file
#: hands to a nested session cannot drift from the one that is actually importable.
GUARD_MODULE = _guard.__name__

#: The real thing, loaded the way the root conftest loads it.
REAL_GUARD = f'pytest_plugins = ["{GUARD_MODULE}"]'

#: Wrong guard 1: presence of the file taken for presence of a halt. `clear()` rewrites
#: rather than unlinks, so a cleared halt leaves `{}` on disk and this refuses a test
#: that tidied up correctly.
EXISTENCE_GUARD = """
import pytest

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    from tradeflow.execution.halt import default_halt_path

    if default_halt_path().exists():
        raise AssertionError("the existence check refused this test")
"""

#: Wrong guard 2: correct detection, no clearing. Blames the right test and then lets
#: the halt stand, so the cascade it exists to prevent happens anyway.
NO_CLEAR_GUARD = """
import pytest

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    from tradeflow.execution.halt import HaltState

    if HaltState().list():
        raise AssertionError("the no-clear guard refused this test")
"""

#: Wrong guard 3: reads the halt path at teardown instead of pinning it at session
#: start, so a test that unsets the variable sends it at whatever root is left.
UNPINNED_GUARD = """
import pytest

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    from tradeflow.execution.halt import HaltState

    state = HaltState()
    leaked = state.list()
    for halt in leaked:
        state.clear(halt.scope)
    if leaked:
        raise AssertionError("the unpinned guard refused this test")
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

TIDY = """
def test_sets_then_clears():
    from tradeflow.execution.halt import HaltState

    state = HaltState()
    state.set("rehearsal", actor="a-tidy-test")
    assert state.clear()
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


def test_the_guard_is_installed_in_the_session_that_actually_runs(pytestconfig):
    """Every other test here installs the guard into a session it builds itself, which
    proves the module works and nothing about whether the suite loads it. It did not:
    the registration could be deleted and all of these stayed green while the original
    cascade came back in full. This is the only assertion about the real surface."""
    manager = pytestconfig.pluginmanager

    assert manager.hasplugin(GUARD_MODULE), f"{GUARD_MODULE} is not registered"
    # Registered is not the same as wired in. Ask the hook this session actually calls
    # at every teardown whose implementations it holds.
    wired = [impl.plugin for impl in manager.hook.pytest_runtest_teardown.get_hookimpls()]
    assert _guard in wired, "the guard is registered but not on pytest_runtest_teardown"


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
    guarded(TIDY).assert_outcomes(passed=1)


def test_the_failure_names_the_halt_rather_than_only_its_existence(guarded):
    """A message reading "durable state was left behind" sends the reader back to the
    bisect this guard exists to remove.

    Matched against an ``E``-prefixed line, which is the assertion text and nothing else.
    Asserting against the whole rendered output is what made the first version of this
    test vacuous: `halt.py` logs the same scope, reason and actor at warning level, so
    every string here was available from the logger whether the guard said anything or
    not."""
    result = guarded(LEAK_THEN_OBSERVE)

    # Scope, reason, actor and the time it was set, all on one line of the guard's own
    # message. `[` opens a character class in fnmatch, so the scope is matched by its
    # closing bracket rather than escaped.
    result.stdout.fnmatch_lines(["E*all] rehearsal (set by the-leaking-test at *+00:00)*"])
    result.stdout.fnmatch_lines(["E*halt state was set in the shared state root*"])


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
    genuinely different questions, and the cheap implementation answers the wrong one.

    The mutant's own message is asserted, not just a non-zero exit: an ImportError at
    teardown also exits non-zero, so a rename inside the embedded guard would otherwise
    satisfy this while proving nothing about existence-versus-in-force."""
    wrong = guarded(TIDY, guard=EXISTENCE_GUARD)
    wrong.assert_outcomes(passed=1, errors=1)
    wrong.stdout.fnmatch_lines(["E*the existence check refused this test*"])

    guarded(TIDY).assert_outcomes(passed=1, errors=0)


def test_a_guard_that_reports_without_clearing_lets_the_cascade_happen(guarded):
    """Mutation, and the half most likely to be lost in a later refactor: nothing
    user-facing depends on the clearing, so it reads like tidiness rather than the
    mechanism that keeps one bad test to one failure."""
    leaky = guarded(LEAK_THEN_OBSERVE, guard=NO_CLEAR_GUARD)

    # Both tests are now implicated: the leaker errors, and the innocent test that
    # follows it fails on a halt it never set — which is the original defect.
    leaky.assert_outcomes(passed=1, failed=1, errors=2)
    leaky.stdout.fnmatch_lines(["E*the no-clear guard refused this test*"])

    # The real guard, same scenario: the innocent test still passes, and the only
    # difference between these two runs is the one thing the mutation removed.
    guarded(LEAK_THEN_OBSERVE).assert_outcomes(passed=2, failed=0, errors=1)


def test_a_guard_that_resolves_the_root_at_teardown_reaches_state_it_never_pinned(guarded):
    """The guard resolves the halt path once, at session start, under the root the suite
    pinned. Resolving it per teardown instead reads whatever `TRADEFLOW_HOME` says at
    that instant — so a test that unsets it sends the guard at `~/.tradeflow`, where the
    next thing it does is *clear* a real operator's halt, because clearing comes before
    reporting.

    Staged against a stand-in home, and the assertion is that the halt is still there
    afterwards: blame alone would not catch the destructive half. The environment is
    changed with `os.environ` rather than `monkeypatch`, because monkeypatch undoes
    itself before a trylast teardown hook runs — which is the only reason nothing in the
    real suite reaches this today."""
    body = """
        import json
        import os

        HOME = None


        def test_leaves_a_halt_outside_the_pinned_root(tmp_path):
            global HOME
            from tradeflow.execution.halt import HaltState

            HOME = tmp_path / "home"
            (HOME / ".tradeflow" / "logs").mkdir(parents=True)
            os.environ["HOME"] = str(HOME)
            os.environ.pop("TRADEFLOW_HOME", None)

            HaltState().set("a real operator stopped trading", actor="a-real-human")
            assert json.loads((HOME / ".tradeflow" / "logs" / "halts.json").read_text())


        def test_the_operator_halt_is_untouched():
            recorded = json.loads((HOME / ".tradeflow" / "logs" / "halts.json").read_text())
            assert recorded, "a halt was cleared in a root the guard never pinned"
        """

    unpinned = guarded(body, guard=UNPINNED_GUARD)
    unpinned.stdout.fnmatch_lines(["E*the unpinned guard refused this test*"])
    unpinned.assert_outcomes(passed=1, failed=1, errors=1)

    # The real guard: state under a root it did not pin is none of its business. It
    # neither blames the test nor touches the file.
    guarded(body).assert_outcomes(passed=2, failed=0, errors=0)


def test_the_guard_contains_its_own_failure_instead_of_erroring_every_later_test(guarded):
    """A guard that raises something unexpected from teardown fails every test collected
    after it — the cascade wearing a different hat, and arriving through the mechanism
    that exists to stop one. The read and the clear are wrapped so the blast radius is
    the one test, and the message names the file to delete rather than pointing the
    reader at a stack in `halt.py`."""
    result = guarded(
        """
        def test_one():
            pass


        def test_two():
            pass
        """,
        guard=REAL_GUARD
        + """

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    import tradeflow.execution.halt as halt

    class Exploding(halt.HaltState):
        def list(self):
            raise RuntimeError("the state root went away")

    halt.HaltState = Exploding
""",
    )

    # Every test errors — the failure is real — but each error is the guard's own
    # contained report, and the session runs to the end rather than aborting.
    result.assert_outcomes(passed=2, errors=2)
    result.stdout.fnmatch_lines(["E*could not read or clear*"])
    result.stdout.fnmatch_lines(["E*Delete that file to continue*"])
