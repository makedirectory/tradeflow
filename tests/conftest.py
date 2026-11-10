"""Suite-wide isolation of the state root.

The state root is where the journal, the trial store, the position ledger, and halt
state all live. Tests used to resolve it to the developer's own checkout, so a run
depended on what someone had run yesterday — and now that a halt is durable state the
live path consults, a real halt left set on a machine would fail execution tests for a
reason that has nothing to do with the code.

Set in ``pytest_configure`` rather than in a fixture, and once for the whole session
rather than per test. Both are forced by the same thing: several modules capture their
default paths in module-level constants, evaluated once when the module is first
imported. A fixture runs too late — collection has already imported them against the
real root — and a per-test root would freeze each constant to whichever test imported
it first, leaving them disagreeing with every later ``state_root()``. One root, set
before anything imports it, keeps the constants and the function telling the same
story.

Tests whose subject *is* how the root resolves from a real environment delete the
variable themselves.
"""

import os
import tempfile

_PREVIOUS = {}


def pytest_configure(config):
    """Point every state path at one throwaway directory, before anything imports it."""
    _PREVIOUS["QUANTRED_HOME"] = os.environ.get("QUANTRED_HOME")
    os.environ["QUANTRED_HOME"] = tempfile.mkdtemp(prefix="quantred-tests-")


def pytest_unconfigure(config):
    previous = _PREVIOUS.get("QUANTRED_HOME")
    if previous is None:
        os.environ.pop("QUANTRED_HOME", None)
    else:
        os.environ["QUANTRED_HOME"] = previous
