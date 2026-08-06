"""The installable command: entry point, and where state lives.

The state-root tests carry the weight here. Every multiple-testing correction in
this project rests on **one** journal accumulating every trial, so a campaign split
across two roots deflates its Sharpe against half the evidence it should — and
nothing errors. Resolution has to be one function with tested branches, not a rule
each caller reimplements.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tradeflow import settings

#: Built from the real distribution name rather than hand-written. A literal here
#: kept passing after the distribution was renamed, while checkout detection had
#: silently broken — a fixture that agrees with the test instead of with the code
#: proves nothing.
PYPROJECT = f'name = "{settings.DISTRIBUTION_NAME}"\n'


# --- the entry point --------------------------------------------------------
def test_the_console_script_target_exists_and_is_callable():
    from tradeflow.cli import main

    assert callable(main)


def test_the_checkout_shim_and_the_installed_command_are_the_same_code():
    """Two behaviors reachable two ways is the confusion this is meant to avoid."""
    import main as shim
    from tradeflow.cli import main as packaged

    assert shim.main is packaged


def test_the_declared_entry_point_matches_reality():
    """A console script that names a function nobody kept is only discovered at
    install time, by a user."""
    import tomllib

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    assert manifest["project"]["scripts"]["tradeflow"] == "tradeflow.cli:main"
    assert manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["tradeflow"]


def test_the_repo_itself_is_recognized_as_a_checkout():
    """The regression guard for the rename that broke this once. Running from the
    repo must resolve state to the repo — the alternative is a developer's journal
    silently moving to ~/.tradeflow while their old one sits in the checkout."""
    assert settings._looks_like_checkout(Path.cwd())
    assert settings.state_root() == Path.cwd()


def test_the_distribution_name_matches_what_is_packaged():
    """Checkout detection reads this out of pyproject.toml; if the two disagree,
    every state path silently relocates."""
    import tomllib

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    assert manifest["project"]["name"] == settings.DISTRIBUTION_NAME


def test_the_version_has_exactly_one_source():
    """A packaged version and a printed version that can disagree will."""
    import tomllib

    import tradeflow

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    assert "version" in manifest["project"].get("dynamic", []), "version must be dynamic"
    assert manifest["tool"]["hatch"]["version"]["path"] == "tradeflow/__init__.py"
    assert tradeflow.__version__


def test_help_exits_clean_through_the_shim():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "verdict" in result.stdout and "backtest" in result.stdout


def test_version_reports_which_copy_is_running_and_where_its_state_is():
    result = subprocess.run(
        [sys.executable, "main.py", "--version"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "running from" in result.stdout
    assert "state root" in result.stdout
    # Multi-line, not re-wrapped into one — the layout is what makes it readable.
    assert result.stdout.count("\n") >= 3


def test_the_package_root_imports_nothing_heavy():
    """`tradeflow --help` must not pay for pandas, the vendor SDK, or an optional
    extra it is not using."""
    source = Path("tradeflow/__init__.py").read_text()
    for heavy in ("import pandas", "import numpy", "from tradeflow.", "import alpaca"):
        assert heavy not in source


# --- the state root ---------------------------------------------------------
def test_an_explicit_home_wins_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.chdir(tmp_path)
    assert settings.state_root() == tmp_path / "elsewhere"


def test_a_checkout_keeps_using_itself(tmp_path, monkeypatch):
    """A developer's logs/ and configs/ must keep working exactly as before."""
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert settings.state_root() == tmp_path


def test_anywhere_else_resolves_to_one_predictable_home(tmp_path, monkeypatch):
    """An installed command must not scatter journals across the filesystem."""
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path / "not-a-checkout")
    assert settings.state_root() == Path.home() / ".tradeflow"


def test_someone_elses_project_is_not_mistaken_for_ours(tmp_path, monkeypatch):
    """Every Python project has a pyproject.toml; putting our journal in one of
    theirs because the filename matched would be worse than not finding a root."""
    (tmp_path / "pyproject.toml").write_text('name = "some-other-tool"\n')
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path / "not-a-checkout")
    assert settings.state_root() != tmp_path


def test_an_unreadable_manifest_is_not_a_checkout(tmp_path, monkeypatch):
    assert not settings._looks_like_checkout(tmp_path / "does-not-exist")


def test_state_path_creates_its_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "home"))
    path = settings.state_path("logs", "research_journal.jsonl")
    assert path.parent.is_dir()
    assert path == tmp_path / "home" / "logs" / "research_journal.jsonl"


def test_every_kind_of_state_lands_under_one_root():
    """One campaign, one root — the split-journal failure has no error to catch it,
    so the only defense is that nothing resolves its own path."""
    from tradeflow.optimization.config_store import DEFAULT_CONFIG_DIR
    from tradeflow.services.analysis import ARTIFACT_DIR
    from tradeflow.services.audit import DEFAULT_AUDIT_PATH, DEFAULT_TRIAL_JOURNAL
    from tradeflow.store.bars import DEFAULT_CACHE_ROOT, DEFAULT_COVERAGE_DB
    from tradeflow.store.trials import DEFAULT_DB_PATH, DEFAULT_JOURNAL_PATH

    root = settings.state_root()
    for path in (
        DEFAULT_TRIAL_JOURNAL,
        DEFAULT_AUDIT_PATH,
        DEFAULT_DB_PATH,
        DEFAULT_JOURNAL_PATH,
        DEFAULT_CACHE_ROOT,
        DEFAULT_COVERAGE_DB,
        DEFAULT_CONFIG_DIR,
        ARTIFACT_DIR,
        settings.ENV_PATH,
    ):
        assert Path(path).is_absolute()
        assert root in Path(path).parents or Path(path).parent == root


def test_the_journal_and_the_trial_store_agree_on_where_the_journal_is():
    """Two modules naming the same file separately is how a campaign silently
    becomes two campaigns."""
    from tradeflow.services.audit import DEFAULT_TRIAL_JOURNAL
    from tradeflow.store.trials import DEFAULT_JOURNAL_PATH

    assert Path(DEFAULT_TRIAL_JOURNAL) == Path(DEFAULT_JOURNAL_PATH)


# --- the zero-configuration first impression --------------------------------
@pytest.mark.slow
def test_demo_runs_against_an_empty_state_root(tmp_path):
    """`install` then `demo` is the whole first impression: no keys, no network, no
    .env, no writable repo."""
    import os

    env = dict(os.environ, TRADEFLOW_HOME=str(tmp_path / "fresh"))
    env.pop("APCA_API_KEY_ID", None)
    env.pop("APCA_API_SECRET_KEY", None)
    result = subprocess.run(
        [sys.executable, "main.py", "demo"], capture_output=True, text=True, timeout=600, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TradeFlow demo" in result.stdout
