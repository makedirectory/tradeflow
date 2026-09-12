"""Fail the test that leaves the kill switch thrown, and clear it before moving on.

The state root is one directory for the whole session (see ``conftest``), so anything a
test writes there is still there for every test after it. Most of what lives there is a
*record* — a journal line, a ledger entry — and a record changes nothing about what runs
next. Halt state is different: the live path consults it before every entry, so a test
that sets one and forgets to clear it does not leave a mess behind, it changes the rules.

That happened. A ``flatten`` test built ``HaltState()`` with no argument, which resolves
to the session root rather than to the test's own directory, and every later test that
tried to open a position was refused by a switch a different test had thrown. Twenty-four
failures across two unrelated files, none of them reproducible alone, and nothing broken:
every component did its job under a halt somebody else had set.

Two properties, and the second is the one that is easy to drop:

**Fail the test that caused it.** The failure belongs to the leaker, not to the first
innocent test that trips over the consequence.

**Clear it before failing.** A guard that reports without clearing turns one leak into
the same cascade it exists to prevent. The clearing is what keeps a bad test to one
failure.

Deliberately narrow. A test that supplies its own path - ``HaltState(tmp_path / ...)`` -
is already isolated and is invisible here, which is correct. Journal, ledger and cache
writes are not guarded: they are records, and a blanket rule over every durable file in
this suite would need an allowlist for the many tests whose subject *is* persistence.
"""

import pytest


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Run after fixture finalisation, so a test that cleans up properly is not blamed."""
    from tradeflow.execution.halt import HaltState

    state = HaltState()
    # Every recorded halt, not `active()`: a strategy-scoped halt never satisfies a
    # global query and would leak straight past a guard built on the narrower read,
    # while still refusing every later entry for that strategy.
    leaked = state.list()
    if not leaked:
        return

    for halt in leaked:
        state.clear(halt.scope)

    described = "\n".join(f"  - {halt}" for halt in leaked)
    raise AssertionError(
        f"this test left halt state set in the shared state root ({state.path}):\n"
        f"{described}\n"
        "It has been cleared so the rest of the session runs unaffected — without that, "
        "every later test that opens a position fails for a reason that has nothing to "
        "do with it. Build HaltState(tmp_path / 'halts.json'), or point TRADEFLOW_HOME "
        "at this test's own directory, rather than writing to the shared root."
    )
