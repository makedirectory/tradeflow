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

from quantred import settings

#: Built from the real distribution name rather than hand-written. A literal here
#: kept passing after the distribution was renamed, while checkout detection had
#: silently broken — a fixture that agrees with the test instead of with the code
#: proves nothing.
PYPROJECT = f'name = "{settings.DISTRIBUTION_NAME}"\n'


# --- the entry point --------------------------------------------------------
def test_the_console_script_target_exists_and_is_callable():
    from quantred.cli import main

    assert callable(main)


def test_the_checkout_shim_and_the_installed_command_are_the_same_code():
    """Two behaviors reachable two ways is the confusion this is meant to avoid."""
    import main as shim
    from quantred.cli import main as packaged

    assert shim.main is packaged


def test_the_declared_entry_point_matches_reality():
    """A console script that names a function nobody kept is only discovered at
    install time, by a user."""
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the package still supports 3.10

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = manifest["project"]["scripts"]
    assert scripts[settings.CLI_NAME] == "quantred.cli:main"
    # The old command is the transition package's. Two distributions installing the same
    # script means one's upgrade or uninstall deletes the other's copy.
    for legacy in settings.LEGACY_CLI_NAMES:
        assert legacy not in scripts
    assert manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["quantred"]


def test_the_repo_itself_is_recognized_as_a_checkout(monkeypatch):
    """The regression guard for the rename that broke detection once. It no longer
    decides where state goes — it decides which *instructions* make sense, `make demo`
    against `quantred demo`. Getting it wrong now sends a contributor an install
    command instead of a Makefile target.

    Drops the suite's own TRADEFLOW_HOME: the override is what every other test wants
    and the one thing this test must not have."""
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)

    assert settings._looks_like_checkout(Path.cwd())
    assert settings.running_from_checkout() is True
    # And deliberately *not* the state root, which is the whole point of the split.
    assert settings.state_root() != Path.cwd()


def test_the_distribution_name_matches_what_is_packaged():
    """Checkout detection reads this out of pyproject.toml; if the two disagree,
    every state path silently relocates."""
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the package still supports 3.10

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    assert manifest["project"]["name"] == settings.DISTRIBUTION_NAME


def _manifest_dir(tmp_path, name: str, line: str = "") -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{name}"{line}\nversion = "1"\n')
    return tmp_path


def test_an_older_checkout_is_still_recognised(tmp_path):
    """A working copy from before the rename declares the old distribution name.

    Failing to recognise it would put that checkout's state in the wrong place without
    a word - the split-campaign failure the state root exists to prevent."""
    # The one place the historical name is spelled out: it is a fact about what was
    # published, not something the code can derive, and every old checkout declares it.
    assert "tradeflow-engine" in settings.LEGACY_DISTRIBUTION_NAMES
    for legacy in settings.LEGACY_DISTRIBUTION_NAMES:
        assert settings._looks_like_checkout(_manifest_dir(tmp_path, legacy))


def test_a_similarly_named_project_is_not_a_checkout(tmp_path):
    """The name line is matched whole: a project merely starting with ours is not ours."""
    for name in (f"{settings.DISTRIBUTION_NAME}-extras", f"{settings.LEGACY_DISTRIBUTION_NAMES[0]}-tools"):
        assert not settings._looks_like_checkout(_manifest_dir(tmp_path, name))
    assert settings._looks_like_checkout(_manifest_dir(tmp_path, settings.DISTRIBUTION_NAME, "  # ours"))


@pytest.mark.parametrize("spelling", ['  name = "{n}"', "name = '{n}'", 'name="{n}"'])
def test_other_valid_spellings_of_the_name_line_still_count(tmp_path, spelling):
    (tmp_path / "pyproject.toml").write_text(f"[project]\n{spelling.format(n=settings.DISTRIBUTION_NAME)}\n")
    assert settings._looks_like_checkout(tmp_path)


def test_the_wheel_installs_only_the_quantred_executable(tmp_path):
    """Read from the built artifact, not from pyproject.toml: the declaration is not
    what an installer reads."""
    import shutil
    import zipfile

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available to build the wheel")
    built = subprocess.run(
        [uv, "build", "--wheel", "-o", str(tmp_path), str(settings.PROJECT_ROOT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if built.returncode != 0:
        pytest.skip(f"the wheel could not be built here (offline?): {built.stderr[-200:]}")
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        entry_points = next(n for n in archive.namelist() if n.endswith("entry_points.txt"))
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        scripts = archive.read(entry_points).decode()
        name_line = next(
            line for line in archive.read(metadata).decode().splitlines() if line.startswith("Name:")
        )
    assert name_line == f"Name: {settings.DISTRIBUTION_NAME}"
    assert f"{settings.CLI_NAME} = quantred.cli:main" in scripts
    for legacy in settings.LEGACY_CLI_NAMES:
        assert f"{legacy} =" not in scripts


def test_the_version_has_exactly_one_source():
    """A packaged version and a printed version that can disagree will."""
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the package still supports 3.10

    import quantred

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    assert "version" in manifest["project"].get("dynamic", []), "version must be dynamic"
    assert manifest["tool"]["hatch"]["version"]["path"] == "quantred/__init__.py"
    assert quantred.__version__


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
    """`quantred --help` must not pay for pandas, the vendor SDK, or an optional
    extra it is not using."""
    source = Path("quantred/__init__.py").read_text()
    for heavy in ("import pandas", "import numpy", "from quantred.", "import alpaca"):
        assert heavy not in source


# --- the state root ---------------------------------------------------------
def test_an_explicit_home_wins_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.chdir(tmp_path)
    assert settings.state_root() == tmp_path / "elsewhere"


def test_a_checkout_is_never_its_own_state_root(tmp_path, monkeypatch):
    """A checkout used to be its own root, which put a private strategy's trials,
    configs and live ledger inside a git working tree — protected by nothing but an
    ignore file, and deleted outright by `git clean -xd`. Contributors opt back in
    explicitly with TRADEFLOW_HOME; nobody gets it by standing somewhere."""
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)

    assert settings.state_root() != tmp_path
    assert settings.state_root() == Path.home() / ".tradeflow"


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
    from quantred.optimization.config_store import DEFAULT_CONFIG_DIR
    from quantred.services.analysis import ARTIFACT_DIR
    from quantred.services.audit import DEFAULT_AUDIT_PATH, DEFAULT_TRIAL_JOURNAL
    from quantred.store.bars import DEFAULT_CACHE_ROOT, DEFAULT_COVERAGE_DB
    from quantred.store.trials import DEFAULT_DB_PATH, DEFAULT_JOURNAL_PATH

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
    from quantred.services.audit import DEFAULT_TRIAL_JOURNAL
    from quantred.store.trials import DEFAULT_JOURNAL_PATH

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
    assert f"{settings.BRAND} demo" in result.stdout


# --- what the source distribution actually carries --------------------------
def _sdist_paths() -> set:
    """The sdist's real file list, from the build backend rather than the manifest.

    Built rather than asserted about. The exclude list *read* correctly while it was
    silently dropping half the example pack, because the patterns are gitignore-style
    and a bare directory name matches at any depth.
    """
    hatchling_sdist = pytest.importorskip("hatchling.builders.sdist")

    builder = hatchling_sdist.SdistBuilder(str(Path(__file__).resolve().parent.parent))
    return {f.relative_path for f in builder.recurse_included_files()}


def test_the_sdist_carries_the_example_pack_with_its_configs():
    """The scaffolded pack minus its configs is worse than no pack.

    `init --example-pack` copies whatever shipped, so a pack without `configs/`
    scaffolds *successfully* and the next line the CLI prints is a
    `backtest --config .../configs/breakout.json` that does not exist. Steps 1, 2 and
    7 of the pack's own README are the same path. An unanchored `configs` exclude did
    exactly this, and nothing in this file looked at an sdist.
    """
    paths = _sdist_paths()

    assert "example/pyproject.toml" in paths, "the pack must be installable"
    for config in ("example/configs/breakout.json", "example/configs/reversion_longshort.json"):
        assert config in paths, f"{config} is a path the CLI and the pack README tell a user to run"


def test_the_sdist_leaves_repo_local_state_behind():
    """The other half of the anchoring: root state must still be excluded."""
    paths = _sdist_paths()
    roots = {p.split("/", 1)[0] for p in paths}

    assert not roots & {"configs", "logs", "cache", "specs", "dist", ".claude", ".vscode", ".venv"}


def test_a_receipt_or_entry_point_under_the_old_name_is_this_project():
    """Evidence written while the project had its previous name describes this copy,
    not a pack someone installed; counting it as one would advise re-adding the engine."""
    assert settings.is_own_distribution(settings.DISTRIBUTION_NAME)
    for legacy in settings.LEGACY_DISTRIBUTION_NAMES:
        assert settings.is_own_distribution(legacy.upper().replace("-", "_"))
    assert not settings.is_own_distribution("some-pack")


def test_the_example_pack_fetch_command_is_the_one_the_docs_show(monkeypatch):
    """An installed copy has no example pack, so the refusal prints a clone-and-copy
    command and the private-strategies guide prints it too. Two hand-written copies of
    one command drift with nothing failing, and a renamed repository would leave a `cp`
    of a directory the clone never made. The guide's command must be the CLI's, and
    the directory it copies must be one the repository actually has."""
    import re

    from quantred import cli
    from quantred.services import setup

    monkeypatch.setattr(setup, "example_pack_source", lambda: None)
    with pytest.raises(SystemExit) as refused:
        cli._scaffold_example_pack("./my-signals")
    command = next(line.strip() for line in str(refused.value.code).splitlines() if "git clone" in line)

    guide = (Path(__file__).resolve().parents[1] / "docs/content/usage/private-strategies.md").read_text()
    assert command in guide

    copied = re.search(r"cp -R (\S+)/(\S+) ", command)
    assert copied.group(1) == settings.REPOSITORY_URL.rstrip("/").rsplit("/", 1)[-1]
    assert (Path(__file__).resolve().parents[1] / copied.group(2) / "pyproject.toml").is_file()


def test_the_example_pack_fetch_command_follows_a_renamed_repository(monkeypatch):
    """The copied directory is the one the clone creates, whatever the repository is called."""
    from quantred import cli
    from quantred.services import setup

    monkeypatch.setattr(setup, "example_pack_source", lambda: None)
    monkeypatch.setattr(cli, "REPOSITORY_URL", "https://example.invalid/acme/renamed")
    with pytest.raises(SystemExit) as refused:
        cli._scaffold_example_pack("./my-signals")
    assert "git clone https://example.invalid/acme/renamed && cp -R renamed/example " in str(
        refused.value.code
    )
