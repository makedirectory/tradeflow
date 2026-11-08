"""The transition package must not be published before the engine release it requires.

PyPI versions cannot be replaced, so a `tradeflow-engine-v*` tag pushed first would
leave a permanent release depending on a version that does not exist. The workflow runs a
script that reads the requirement from the transition package's own manifest.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from quantred import settings

SCRIPT = settings.PROJECT_ROOT / ".github" / "scripts" / "require_published_dependency.py"
MANIFEST = settings.PROJECT_ROOT / "legacy" / "tradeflow-engine" / "pyproject.toml"
WORKFLOW = settings.PROJECT_ROOT / ".github" / "workflows" / "release.yml"

pytest.importorskip("packaging")
if sys.version_info < (3, 11):  # the script reads TOML with tomllib, as the CI runner has it
    pytest.skip("tomllib is Python 3.11+", allow_module_level=True)


def _script():
    spec = importlib.util.spec_from_file_location("require_published_dependency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file(name="x.whl", yanked=False):
    return [{"filename": name, "yanked": yanked}]


def test_the_check_refuses_when_the_required_engine_is_not_published():
    script = _script()
    text = MANIFEST.read_text()
    with pytest.raises(SystemExit, match="not on PyPI yet"):
        script.satisfied(text, {})
    with pytest.raises(SystemExit):  # only releases below the required floor exist
        script.satisfied(text, {"2.3.0": _file()})
    with pytest.raises(SystemExit):  # a yanked release is not a release to depend on
        script.satisfied(text, {"2.4.0": _file(yanked=True)})
    with pytest.raises(SystemExit):  # nor one the dependency's upper bound excludes
        script.satisfied(text, {"3.0.0": _file()})


def test_the_check_accepts_a_published_release_that_meets_the_requirement():
    script = _script()
    text = MANIFEST.read_text()
    assert script.satisfied(text, {"2.3.0": _file(), "2.4.0": _file(), "2.4.1": _file()}) == "2.4.1"


def test_the_workflow_runs_it_on_the_transition_path_only():
    text = WORKFLOW.read_text()
    assert "require_published_dependency.py" in text
    step = text[text.index("The engine release this requires is on PyPI") :]
    assert step.splitlines()[1].strip() == "if: startsWith(github.ref, 'refs/tags/tradeflow-engine-v')"


def test_the_script_exits_non_zero_from_the_command_line(tmp_path):
    releases = tmp_path / "releases.json"
    releases.write_text('{"releases": {}}')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(MANIFEST), str(releases)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "not on PyPI yet" in result.stderr
