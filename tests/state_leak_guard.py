"""Fail the test that leaves the kill switch thrown, and clear it before moving on.

The state root is one directory for the whole session (see ``tests/conftest``), so
anything a test writes there is still there for every test after it. Most of what lives
there is a *record* - a journal line, a ledger entry - and a record changes nothing about
what runs next. Halt state is different: the live path consults it before every entry, so
a test that sets one and forgets to clear it does not leave a mess behind, it changes the
rules.

That happened. A ``flatten`` test built ``HaltState()`` with no argument, which resolves
to the session root rather than to the test's own directory, and every later test that
tried to open a position was refused by a switch a different test had thrown. Two dozen
failures across two unrelated files, none of them reproducible alone, and nothing broken:
every component did its job under a halt somebody else had set.

Three properties, and the last two are the ones that look like housekeeping:

**Fail the test that caused it.** The failure belongs to the leaker, not to the first
innocent test that trips over the consequence.

**Clear it before failing.** A guard that reports without clearing turns one leak into
the same cascade it exists to prevent.

**Contain its own failure.** Anything unexpected while reading or clearing is reported
against the one test and does not propagate, because a guard that raises from teardown
for the rest of the session is the cascade wearing a different hat.

Deliberately narrow. A test that supplies its own path - ``HaltState(tmp_path / ...)`` -
is already isolated and is invisible here, which is correct. Journal, ledger and cache
writes are not guarded: they are records, and a blanket rule over every durable file in
this suite would need an allowlist for the many tests whose subject *is* persistence.
"""

import os

import pytest

#: The halt file, resolved **once**, while the session's own state root is in force.
#: Resolving it per teardown instead would read whatever ``TRADEFLOW_HOME`` happened to
#: say at that instant - and a test that unsets the variable would point the guard at the
#: developer's real state root, where the next thing it does is clear a live operator
#: halt. Pinning it means the guard can only ever touch the root this session created.
_HALT_PATH = None


def pytest_sessionstart(session):
    """Resolve the halt path under the pinned root, after conftest has set it."""
    global _HALT_PATH

    # No root means nothing pinned this session to a throwaway directory, and the guard
    # would be operating on real state. Staying off is the only safe reading.
    if not os.environ.get("TRADEFLOW_HOME"):
        _HALT_PATH = None
        return

    # Asked of the real resolver rather than rebuilt from parts, so this cannot drift
    # from where halt state actually lives.
    from tradeflow.execution.halt import default_halt_path

    _HALT_PATH = default_halt_path()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Run after fixture finalisation, so a test that cleans up properly is not blamed."""
    if _HALT_PATH is None:
        return

    from tradeflow.execution.halt import HaltState

    state = HaltState(_HALT_PATH)
    try:
        # Every recorded halt, not `active()`: a strategy-scoped halt never satisfies a
        # global query and would leak straight past a guard built on the narrower read,
        # while still refusing every later entry for that strategy.
        leaked = state.list()
        if not leaked:
            return
        for halt in leaked:
            state.clear(halt.scope)
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        raise AssertionError(
            f"the halt-state leak guard could not read or clear {_HALT_PATH}: "
            f"{exc!r}. Delete that file to continue. This is reported against one test "
            "rather than raised onward, because a guard that fails every remaining "
            "teardown is the cascade it exists to prevent."
        ) from exc

    described = "\n".join(f"  - {halt}" for halt in leaked)
    raise AssertionError(
        f"halt state was set in the shared state root when this test finished "
        f"({_HALT_PATH}):\n"
        f"{described}\n"
        "It has been cleared so the rest of the session runs unaffected — without that, "
        "every later test that opens a position fails for a reason that has nothing to "
        "do with it. Build HaltState(tmp_path / 'halts.json'), or point TRADEFLOW_HOME "
        "at this test's own directory, rather than writing to the shared root. A fixture "
        "that holds a halt across several tests needs its own path for the same reason: "
        "this guard clears the shared root between every test."
    )
