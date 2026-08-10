"""Suite-wide isolation.

The state root is where the journal, the trial store, the position ledger, and halt
state all live. A test that reads or writes it unpointed reads the *developer's* —
so a suite run would depend on what someone ran yesterday, and, now that a halt is
durable state the live path consults, a real halt left set on a machine would make
execution tests fail for a reason that has nothing to do with the code.

Redirecting ``TRADEFLOW_HOME`` for every test closes both.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_state_root(tmp_path, monkeypatch):
    """Point every state path at this test's own directory."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    return tmp_path / "state"
