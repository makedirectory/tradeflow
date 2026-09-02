"""Where evidence lands, and why the default cannot depend on being configured.

A checkout used to be its own state root, so a contributor's `logs/` kept working the
way it always had. That was convenient and wrong in two directions: a private
strategy's trials, saved configs and live position ledger accumulated inside a git
working tree protected by nothing but `.gitignore`, and `git clean -xd` would delete
the lot, live ledger included.

A user arriving through PyPI or over MCP may never learn there is a working tree, a
`logs/` directory or an ignore file — so a warning cannot carry this, and neither can
`init`, which may never be run. The safe root has to be the one you get without knowing
to ask.
"""

from pathlib import Path

import pytest

from tradeflow.settings import (
    PROJECT_ROOT,
    git_worktree_containing,
    orphaned_checkout_state,
    running_from_checkout,
    state_root,
)


@pytest.fixture(autouse=True)
def _no_inherited_home(monkeypatch):
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)


def test_a_checkout_is_not_its_own_state_root(monkeypatch, tmp_path):
    """The change. Standing in a working copy must not send evidence into it."""
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert state_root() == tmp_path / ".tradeflow"


def test_the_default_is_the_same_wherever_you_stand(monkeypatch, tmp_path):
    """It used to depend on the working directory, so the answer changed as you moved."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    monkeypatch.chdir(PROJECT_ROOT)
    from_checkout = state_root()
    monkeypatch.chdir(tmp_path)
    from_elsewhere = state_root()

    assert from_checkout == from_elsewhere


def test_the_environment_still_overrides(monkeypatch, tmp_path):
    """The escape hatch, what a container mounts, and how a contributor opts back in."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "elsewhere"))

    assert state_root() == tmp_path / "elsewhere"


def test_how_the_copy_was_reached_is_a_separate_question(monkeypatch, tmp_path):
    """Messaging depends on it; state does not. Conflating them is what sent a
    contributor's instructions to an installed user, and would now do the reverse."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(PROJECT_ROOT)

    assert running_from_checkout() is True
    assert state_root() != PROJECT_ROOT


# --- the hazard the default removes ------------------------------------------------
def test_a_state_root_inside_a_repository_is_detected(tmp_path):
    """One ignore-file edit from disclosure, and `git clean -xd` deletes it outright."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "logs" / "deep"
    nested.mkdir(parents=True)

    assert git_worktree_containing(nested) == tmp_path


def test_a_state_root_outside_a_repository_reports_nothing(tmp_path):
    """Both directions: a warning that always fires is one nobody reads."""
    assert git_worktree_containing(tmp_path) is None


def test_a_missing_directory_is_not_an_error(tmp_path):
    """The check runs during setup, before anything has been created."""
    assert git_worktree_containing(tmp_path / "not-yet") is None


# --- the contributor whose evidence is already somewhere else ----------------------
def test_earlier_checkout_state_is_reported_rather_than_ignored(monkeypatch, tmp_path):
    """The failure the fix could have caused: a contributor's journals stayed in their
    checkout while the next campaign deflated against none of them, with nothing
    erroring — the exact split the state root exists to prevent."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    found = orphaned_checkout_state()

    assert found == PROJECT_ROOT  # this repo has logs/ with a ledger in it


def test_nothing_is_reported_when_the_checkout_is_the_active_root(monkeypatch):
    """A contributor who opted back in has no split to warn about."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(PROJECT_ROOT))

    assert orphaned_checkout_state() is None


# --- the diagnostic says all of it -------------------------------------------------
def test_the_setup_check_states_where_evidence_lands(monkeypatch, tmp_path):
    """A user who arrived over MCP or from PyPI has no reason to know the file model.
    This is the one command that exists to explain it."""
    from tradeflow.services.setup import run_checks

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    names = {check.name for check in run_checks(path=tmp_path / ".env")}

    assert "state root" in names
    assert "research journal" in names
    assert "position ledger" in names
    assert "state outside version control" in names


def test_the_setup_check_fails_when_state_sits_in_a_repository(monkeypatch, tmp_path):
    """It is a real hazard, so it is a failed check rather than a note."""
    from tradeflow.services.setup import run_checks

    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))

    check = next(c for c in run_checks(path=tmp_path / ".env") if c.name == "state outside version control")

    assert check.passed is False
    assert "git clean" in check.detail  # names the destructive half, not only disclosure
