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

import os
import sys
from pathlib import Path

import pytest

from tradeflow import settings
from tradeflow.settings import (
    DISTRIBUTION_NAME,
    PROJECT_ROOT,
    git_worktree_containing,
    running_from_checkout,
    state_root,
)


@pytest.fixture(autouse=True)
def _no_inherited_home(monkeypatch):
    # Modules bind their default state paths at first import. Importing them here,
    # while the suite's throwaway TRADEFLOW_HOME is still set, keeps a later first
    # import from happening with it removed and binding a module to the real home
    # for every test file that runs after this one.
    import tradeflow.cli  # noqa: F401
    import tradeflow.mcp.server  # noqa: F401
    import tradeflow.services.audit  # noqa: F401
    import tradeflow.services.setup  # noqa: F401

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
def _checkout_with(tmp_path, *evidence) -> Path:
    """A working copy as `_looks_like_checkout` recognises one, holding `evidence`.

    Built rather than borrowed. This asserted against the real `PROJECT_ROOT` and the
    untracked `logs/` a developer's own checkout happens to have, so it passed on the
    machine that wrote it and failed on every clean one — CI included. What the check
    reports must not depend on whose disk it runs from.
    """
    checkout = tmp_path / "checkout"
    (checkout / "logs").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(f'[project]\nname = "{DISTRIBUTION_NAME}"\n')
    for name in evidence:
        (checkout / "logs" / name).write_text("")
    return checkout


@pytest.mark.parametrize("evidence", ["research_journal.jsonl", "trials.db", "position_ledger.jsonl"])
def test_earlier_checkout_state_is_reported_rather_than_ignored(monkeypatch, tmp_path, evidence):
    """The failure the fix could have caused: a contributor's journals stayed in their
    checkout while the next campaign deflated against none of them, with nothing
    erroring — the exact split the state root exists to prevent.

    Every kind of evidence counts, because any one of them alone is a split campaign.
    """
    checkout = _checkout_with(tmp_path, evidence)
    monkeypatch.setattr(settings, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert settings.orphaned_checkout_state() == checkout


def test_a_checkout_with_no_evidence_of_its_own_is_not_reported(monkeypatch, tmp_path):
    """Both directions: a fresh clone has a `logs/` and nothing in it, and a warning
    that fires for everyone is one nobody reads."""
    checkout = _checkout_with(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert settings.orphaned_checkout_state() is None


def test_a_directory_that_is_not_a_checkout_is_not_reported(monkeypatch, tmp_path):
    """An installed copy has no working tree to have left evidence in."""
    stray = tmp_path / "elsewhere"
    (stray / "logs").mkdir(parents=True)
    (stray / "logs" / "trials.db").write_text("")
    monkeypatch.setattr(settings, "PROJECT_ROOT", stray)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert settings.orphaned_checkout_state() is None


def test_nothing_is_reported_when_the_checkout_is_the_active_root(monkeypatch, tmp_path):
    """A contributor who opted back in has no split to warn about — and this has to be
    asserted against a checkout that *does* hold evidence, or it passes for the wrong
    reason on any machine that has none."""
    checkout = _checkout_with(tmp_path, "research_journal.jsonl")
    monkeypatch.setattr(settings, "PROJECT_ROOT", checkout)
    monkeypatch.setenv("TRADEFLOW_HOME", str(checkout))

    assert settings.orphaned_checkout_state() is None


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


# --- reaching the state root is a use, never an import ----------------------------
def _import_probe(body: str, home: Path):
    """Run `body` in a fresh interpreter under a state root that does not exist yet.

    A subprocess because every other test in this suite has already imported these
    modules, and an import that already happened creates nothing the second time —
    which would mask exactly the thing being asserted.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "TRADEFLOW_HOME": str(home)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


@pytest.mark.parametrize(
    "module",
    ["tradeflow.store.trials", "tradeflow.services.audit", "tradeflow.research.agent"],
)
def test_importing_a_journal_module_does_not_create_the_state_root(tmp_path, module):
    """It did. The journal's location converged onto `settings.trial_journal_path()`,
    which goes through `state_path()`, which does `mkdir(parents=True)` — so a
    module-scope constant naming the journal made importing that module a filesystem
    write. Every command imports it, so an unwritable or read-only state root began
    failing at *import* rather than at first use: a broken environment bricking the
    CLI, which is the case the registry code goes out of its way to protect against.

    Before the two definitions converged this was pure path construction and the
    constant was free. The convergence is worth keeping; paying for it at import is
    not.
    """
    home = tmp_path / "unwritten"

    out = _import_probe(f"import {module}; import os; print(os.path.isdir(r'{home}'))", home)

    assert out == "False"


def test_the_journal_constants_still_resolve_on_access(tmp_path):
    """The other direction: lazy must not mean absent. Both constants still exist,
    still name one file, and reaching either one creates the directory it needs."""
    home = tmp_path / "on-demand"
    probe = (
        "from tradeflow.store.trials import DEFAULT_JOURNAL_PATH;"
        "from tradeflow.services.audit import DEFAULT_TRIAL_JOURNAL;"
        "import os;"
        "assert DEFAULT_JOURNAL_PATH == DEFAULT_TRIAL_JOURNAL, (DEFAULT_JOURNAL_PATH, DEFAULT_TRIAL_JOURNAL);"
        f"print(os.path.isdir(os.path.dirname(str(DEFAULT_JOURNAL_PATH))), str(DEFAULT_JOURNAL_PATH).startswith(r'{home}'))"
    )

    assert _import_probe(probe, home) == "True True"


def test_an_unknown_attribute_still_raises_attribute_error():
    """A module `__getattr__` that answers every name turns a typo into a silent
    wrong value, which is worse than the import-time cost it replaced."""
    from tradeflow.services import audit
    from tradeflow.store import trials

    for module in (trials, audit):
        with pytest.raises(AttributeError):
            module.NO_SUCH_CONSTANT


# --- how the copy was reached: where the package lives, not where you stand --------
def _installed_copy(monkeypatch, tmp_path):
    """Make the running package look like it was imported from site-packages."""
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path / "site-packages")


def test_an_installed_copy_run_inside_a_checkout_is_still_installed(monkeypatch, tmp_path):
    """The working directory says nothing about which copy is running. An installed
    command typed inside a clone of this repository used to be told to `uv sync` and
    `python main.py`, neither of which is the thing that is running."""
    _installed_copy(monkeypatch, tmp_path)
    monkeypatch.chdir(PROJECT_ROOT)

    assert running_from_checkout() is False
    assert settings.invocation("demo") == f"{settings.CLI_NAME} demo"


def test_a_checkout_is_detected_from_anywhere_it_is_run(monkeypatch, tmp_path):
    """`python main.py` imports the package from the working copy, wherever the shell is."""
    monkeypatch.chdir(tmp_path)

    assert running_from_checkout() is True
    assert settings.invocation("demo") == "python main.py demo"


@pytest.mark.parametrize("started", ["/venv/bin/{cli}", "/venv/bin/{cli}.exe", "/venv/bin/tradeflow"])
def test_a_checkout_started_as_the_installed_command_is_told_that_command(started, monkeypatch, tmp_path):
    """A checkout's environment carries the installed command too, and someone who typed
    it was told `python main.py cache warm ...`, which fails outside the repository root.
    The instruction now echoes how the process started; the alias is answered with the
    current name."""
    from tradeflow import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [started.format(cli=settings.CLI_NAME), "verdict"])

    assert running_from_checkout() is True
    assert settings.invocation("demo") == f"{settings.CLI_NAME} demo"
    assert settings.invocation("init", make_target="init") == f"{settings.CLI_NAME} init"
    assert "make " not in settings._missing_credentials_message(["APCA_API_KEY_ID"])
    assert "make " not in cli._next_step_hint()


@pytest.mark.parametrize("started", ["main.py", "/repo/main.py"])
def test_a_checkout_started_through_the_shim_keeps_make_and_main_py(started, monkeypatch):
    """The other direction: `python main.py` and every make target start here."""
    from tradeflow import cli

    monkeypatch.setattr(sys, "argv", [started, "verdict"])

    assert settings.invocation("demo") == "python main.py demo"
    assert settings.invocation("init", make_target="init") == "make init"
    assert "make init" in settings._missing_credentials_message(["APCA_API_KEY_ID"])
    assert "make init" in cli._next_step_hint()


def test_messages_still_render_when_the_working_directory_is_gone(monkeypatch, tmp_path):
    """The flatten report, the halt line and the missing-credentials message all name a
    command, and naming it used to read the working directory - which raises once it
    has been deleted. That is the state a shell is in after `git clean`, and these
    messages are printed after the halt and the closes have already happened."""
    if sys.platform == "win32":
        pytest.skip("a process cannot delete its own working directory on Windows")
    from tradeflow import cli
    from tradeflow.execution.flatten import CLOSED, FlattenReport

    _installed_copy(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "home"))
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    monkeypatch.chdir(doomed)
    doomed.rmdir()
    with pytest.raises(OSError):
        Path.cwd()  # the premise: this really is a deleted working directory

    report = FlattenReport(
        started_at="t",
        halted=True,
        orders_cancelled=True,
        close_submitted=True,
        observed=CLOSED,
        checked_at="t",
        open_orders=0,
        recorded_closed=[],
    )
    assert report.complete
    assert "resume all" in report.summary()
    assert "resume" in cli._invocation("resume all")
    assert "init" in settings._missing_credentials_message(["APCA_API_KEY_ID"])


def test_another_tables_name_line_is_not_a_checkout(tmp_path):
    """Only `[project]` names the project: an index or a tool table can carry a
    `name = "..."` that is merely ours."""
    own = settings.DISTRIBUTION_NAME
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "someone-else"\n\n[[tool.uv.index]]\nname = "{own}"\nurl = "https://x"\n'
    )
    assert not settings._looks_like_checkout(tmp_path)

    (tmp_path / "pyproject.toml").write_text(
        f'[build-system]\nrequires = []\n\n[ project ]\nversion = "1"\nname = "{own}"\n'
    )
    assert settings._looks_like_checkout(tmp_path)


def test_the_project_image_is_told_it_is_not_a_checkout(monkeypatch, tmp_path):
    """The image carries main.py and pyproject.toml, so it read as a checkout and told a
    container user to run `make init` and copy `.env.example` - neither is in the image -
    and `init` could not write the .env compose mounts read-only."""
    from tradeflow import cli

    monkeypatch.setenv(settings.CONTAINER_ENV, "1")
    monkeypatch.setattr(sys, "argv", ["main.py", "demo"])
    assert settings.invocation("init", make_target="init") == f"{settings.CLI_NAME} init"
    message = settings._missing_credentials_message(["APCA_API_KEY_ID"])
    assert "make " not in message and ".env.example" not in message
    assert "read-only" in message
    hint = cli._next_step_hint()
    assert "make " not in hint and ".env.example" not in hint and "read-only" in hint


def test_the_dockerfile_declares_the_container_and_stops_runtime_syncs():
    """The two settings the image depends on, read from the Dockerfile itself: without
    UV_NO_SYNC every `uv run` re-synced the environment at start (an offline container
    could not start), and without the container flag it printed checkout instructions."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "ENV UV_NO_SYNC=1" in dockerfile
    assert f"ENV {settings.CONTAINER_ENV}=1" in dockerfile


def test_the_build_context_is_an_allowlist_that_keeps_secrets_out():
    """There was no .dockerignore, and the Dockerfile copies the whole context: an image
    built from a checkout carried `.env`, `specs/`, `configs/` and `logs/`."""
    ignore = (PROJECT_ROOT / ".dockerignore").read_text().splitlines()
    assert ignore[[line.strip() for line in ignore].index("*")] == "*"
    allowed = {line[1:].rstrip("/") for line in ignore if line.startswith("!")}
    assert not allowed & {".env", "specs", "configs", "logs", "cache", ".git", ".venv"}
    assert {"tradeflow", "main.py", "pyproject.toml"} <= allowed
