"""The final `tradeflow-engine` release: the `tradeflow` import and command, over `quantred`.

The shim cannot be tested by importing it into this process (the suite runs with
`quantred` only), so these run it in a subprocess, and the slow test builds both wheels and
installs them into a fresh virtual environment, which is the only way to see that the
console script, the import package and the entry-point metadata land where an upgrade needs
them.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from quantred import settings

PACKAGE_DIR = settings.PROJECT_ROOT / "packaging" / "tradeflow-engine"

pytestmark = pytest.mark.skipif(not PACKAGE_DIR.is_dir(), reason="not a source checkout")


def _run_python(code: str, *extra_args: str) -> subprocess.CompletedProcess:
    env_path = f"{PACKAGE_DIR}{':' if sys.platform != 'win32' else ';'}{settings.PROJECT_ROOT}"
    return subprocess.run(
        [sys.executable, "-W", "always", *extra_args, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": env_path, "PATH": "", "TRADEFLOW_HOME": "/nonexistent-state"},
    )


def _toml(path: Path) -> dict:
    tomllib = pytest.importorskip("tomllib")  # 3.11+
    return tomllib.loads(path.read_text())


# --- the shim, with quantred taken from this checkout --------------------------------
def test_importing_tradeflow_warns_and_gives_back_quantreds_own_modules():
    """Identity, not equality: a pack subclassing the old path must pass `issubclass`
    against the registry's `Strategy`, and a second copy of a module would not."""
    done = _run_python(
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import tradeflow\n"
        "assert any(issubclass(w.category, DeprecationWarning) and 'quantred' in str(w.message) for w in caught), caught\n"
        "import tradeflow.strategies.base as old, quantred.strategies.base as new\n"
        "assert old is new\n"
        "from tradeflow.strategies.base import Strategy\n"
        "assert Strategy is new.Strategy\n"
        "assert new.__spec__.name == 'quantred.strategies.base', new.__spec__.name\n"
        "import tradeflow.demo.strategies as d1, quantred.demo.strategies as d2\n"
        "assert d1 is d2 and tradeflow.__version__ == __import__('quantred').__version__\n"
        "print('ok')\n"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_a_missing_old_submodule_is_still_an_import_error_and_a_real_one_is_not_swallowed():
    done = _run_python(
        "import tradeflow\n"
        "try:\n"
        "    import tradeflow.no_such_module\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert 'no_such_module' in str(exc), exc\n"
        "else:\n"
        "    raise SystemExit('expected ModuleNotFoundError')\n"
        "assert not hasattr(tradeflow, 'no_such_module')\n"
        "print('ok')\n"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_importing_the_shim_twice_installs_one_finder():
    done = _run_python(
        "import sys, importlib, tradeflow\n"
        "importlib.reload(tradeflow)\n"
        "n = [f for f in sys.meta_path if type(f).__name__ == '_AliasFinder']\n"
        "assert len(n) == 1, n\n"
        "assert sys.meta_path[0] is n[0]\n"
        "print('ok')\n"
    )
    assert done.returncode == 0, done.stderr


def test_python_dash_m_tradeflow_cli_runs_the_quantred_cli():
    """Old MCP client configs launch `python -m tradeflow.cli ...`; runpy needs a real file."""
    done = subprocess.run(
        [sys.executable, "-m", "tradeflow.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(settings.PROJECT_ROOT),
        env={
            "PYTHONPATH": f"{PACKAGE_DIR}:{settings.PROJECT_ROOT}",
            "PATH": "",
            "TRADEFLOW_HOME": "/nonexistent-state",
        },
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(f"{settings.CLI_NAME} "), done.stdout


# --- the declared metadata ----------------------------------------------------------
def test_the_transition_package_offers_exactly_the_engines_extras():
    """`uv tool upgrade` re-resolves from the recorded request (`tradeflow-engine[mcp]`).
    An extra missing from the new release is dropped, and an exact tool environment then
    removes what it pulled in. Compared by reading both manifests so they cannot drift."""
    root = _toml(settings.PROJECT_ROOT / "pyproject.toml")["project"]
    mine = _toml(PACKAGE_DIR / "pyproject.toml")["project"]
    expected = set(root["optional-dependencies"]) - {"dev"}
    assert set(mine["optional-dependencies"]) == expected
    main_spec = next(d for d in mine["dependencies"] if d.startswith("quantred")).removeprefix("quantred")
    for extra, requirements in mine["optional-dependencies"].items():
        assert requirements == [f"quantred[{extra}]{main_spec}"]


def test_the_transition_package_owns_the_old_command_and_declares_no_entry_points():
    mine = _toml(PACKAGE_DIR / "pyproject.toml")["project"]
    assert mine["scripts"] == {legacy: "quantred.cli:main" for legacy in settings.LEGACY_CLI_NAMES}
    assert "entry-points" not in mine
    assert mine["name"] in settings.LEGACY_DISTRIBUTION_NAMES
    root_scripts = _toml(settings.PROJECT_ROOT / "pyproject.toml")["project"]["scripts"]
    assert not set(root_scripts) & set(mine["scripts"]), "two distributions would own one file"


# --- built and installed ------------------------------------------------------------
@pytest.mark.slow
def test_both_wheels_installed_together_give_the_old_import_and_command(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available to build the wheels")
    dist = tmp_path / "dist"
    for source in (settings.PROJECT_ROOT, PACKAGE_DIR):
        built = subprocess.run(
            [uv, "build", "--wheel", "-o", str(dist), str(source)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if built.returncode != 0:
            pytest.skip(f"a wheel could not be built here (offline?): {built.stderr[-200:]}")
    new_wheel = next(dist.glob("quantred-*.whl"))
    old_wheel = next(dist.glob("tradeflow_engine-*.whl"))

    # No two distributions own one file.
    with zipfile.ZipFile(new_wheel) as a, zipfile.ZipFile(old_wheel) as b:
        shared = {n for n in a.namelist() if not n.split("/")[0].endswith(".dist-info")} & {
            n for n in b.namelist() if not n.split("/")[0].endswith(".dist-info")
        }
        assert not shared, shared
        assert not any(n.startswith("tradeflow/") for n in a.namelist())
        assert (
            "tradeflow = quantred.cli:main"
            in b.read(next(n for n in b.namelist() if n.endswith("entry_points.txt"))).decode()
        )

    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True, timeout=120)
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    installed = subprocess.run(
        [uv, "pip", "install", "-q", "--python", str(python), "--no-deps", str(new_wheel), str(old_wheel)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert installed.returncode == 0, installed.stderr
    # The third-party dependencies are the test environment's own (the build is offline);
    # the venv's quantred and tradeflow come first on the path.
    site = next(venv.glob("lib/python*/site-packages"), None) or venv / "Lib" / "site-packages"
    import sysconfig

    (site / "_deps.pth").write_text(sysconfig.get_paths()["purelib"] + "\n")

    env = {"PATH": str(python.parent), "TRADEFLOW_HOME": str(tmp_path / "state"), "HOME": str(tmp_path)}
    check = subprocess.run(
        [
            str(python),
            "-W",
            "error::DeprecationWarning:__main__",
            "-c",
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import tradeflow\n"
            "assert any(issubclass(w.category, DeprecationWarning) for w in caught)\n"
            "import tradeflow.strategies.base as a, quantred.strategies.base as b\n"
            "assert a.Strategy is b.Strategy\n"
            "from tradeflow.strategies.base import Strategy\n"
            "import quantred\n"
            "assert quantred.__file__.startswith(%r), quantred.__file__\n" % str(venv),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
    )
    assert check.returncode == 0, check.stderr

    banners = []
    for argv in (
        [str(python.parent / "tradeflow"), "--version"],
        [str(python.parent / settings.CLI_NAME), "--version"],
        [str(python), "-m", "tradeflow.cli", "--version"],
    ):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
        assert done.returncode == 0, (argv, done.stderr)
        banners.append(done.stdout.strip())
    assert all(b.startswith(f"{settings.CLI_NAME} ") for b in banners), banners
    assert len(set(banners)) == 1, banners
