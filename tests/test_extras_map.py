"""The optional-extras map and the one recovery text every missing-dependency message uses.

The hazard is two-sided: a message that sends an installed reader to a checkout command
they cannot run, and a message whose command *works* but rebuilds the tool environment
with only the extra it names - silently removing every other extra and every pack the
reader had added. Both read correctly in the source.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from quantred import __version__, cli, settings
from quantred.utils import extras

PYPROJECT = Path(settings.PROJECT_ROOT) / "pyproject.toml"

#: Where the import name differs from the distribution name. Everything else is the same.
IMPORT_NAME = {"scikit-learn": "sklearn"}


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement.strip(), maxsplit=1)[0].lower()


def _published_extras() -> dict:
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the package still imports on 3.10
    manifest = tomllib.loads(PYPROJECT.read_text())
    return manifest["project"]["optional-dependencies"]


def _as_installed_copy(monkeypatch, installed=()):
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    # Every probe the doctor and the commands use, not only the summary: the rows ask
    # `installed(name)` one extra at a time, so stubbing the list alone passed exactly
    # where the SDKs were absent and failed for a contributor who had installed one.
    monkeypatch.setattr(extras, "installed_extras", lambda: sorted(installed))
    monkeypatch.setattr(extras, "installed", lambda name: name in installed)
    monkeypatch.setattr(cli, "extras_installed", lambda name: name in installed)
    monkeypatch.setattr(extras, "_pack_distributions", lambda: [])
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: False)
    monkeypatch.setattr(extras, "_tool_receipt", lambda: None)


def test_the_extras_map_matches_the_published_extras():
    """The map is read against the declaration file, so adding an extra to
    `pyproject.toml` and not here (or the reverse) fails instead of drifting."""
    published = {name: reqs for name, reqs in _published_extras().items() if name != "dev"}
    assert set(extras.EXTRAS) == set(published)
    for name, requirements in published.items():
        provided = {IMPORT_NAME.get(_requirement_name(r), _requirement_name(r)) for r in requirements}
        assert set(extras.EXTRAS[name].modules) == provided, name


def test_store_is_missing_if_any_of_its_modules_is_missing(monkeypatch):
    """`store` is three packages. Probing only the first reported it installed with
    the Parquet store or the lazy compute path still failing at first use."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "polars" else real(name, *a, **k)
    )
    assert extras.installed("store") is False
    assert "store" not in extras.installed_extras()


def test_an_extra_is_installed_when_every_module_is_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    assert extras.installed("store") is True


def test_an_installed_copy_hint_keeps_the_extras_already_installed(monkeypatch):
    """`uv tool install --force` replaces the environment: naming only `mcp` would
    remove the `viz` and `store` the reader already had."""
    _as_installed_copy(monkeypatch, installed={"viz", "store"})
    hint = extras.install_hint("mcp")
    assert f"[mcp,store,viz]=={__version__}" in hint
    assert f"'{settings.DISTRIBUTION_NAME}[mcp,store,viz]" in hint  # quoted for zsh


def test_the_pip_alternative_adds_only_what_was_asked_for(monkeypatch):
    """pip extras are additive; restating the installed ones would be noise."""
    _as_installed_copy(monkeypatch, installed={"viz"})
    hint = extras.install_hint("mcp")
    assert f"python -m pip install '{settings.DISTRIBUTION_NAME}[mcp]=={__version__}'" in hint


def test_a_detected_uv_tool_environment_is_labelled_and_the_other_form_still_shown(monkeypatch):
    _as_installed_copy(monkeypatch)
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: True)
    hint = extras.install_hint("viz")
    assert "With uv tool (detected):" in hint
    assert "pip install" in hint
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: False)
    assert "(detected)" not in extras.install_hint("viz")


def test_uv_tool_environments_are_recognised_by_where_they_live(monkeypatch, tmp_path):
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "share" / "uv" / "tools" / "pkg"))
    assert extras._in_uv_tool_env() is True
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "project" / ".venv"))
    assert extras._in_uv_tool_env() is False
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "custom"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "custom" / "pkg"))
    assert extras._in_uv_tool_env() is True


def test_packs_installed_with_with_are_named_so_they_can_be_re_added(monkeypatch):
    _as_installed_copy(monkeypatch)
    monkeypatch.setattr(extras, "_pack_distributions", lambda: ["acme-signals"])
    hint = extras.install_hint("viz")
    assert "re-add any packs you installed with --with: acme-signals" in hint
    monkeypatch.setattr(extras, "_pack_distributions", lambda: [])
    # Outside a uv tool environment there is no --with to re-add; the line would be noise.
    assert "none detected" not in extras.install_hint("viz")
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: True)
    assert "none detected" in extras.install_hint("viz")


def test_this_distributions_own_entry_points_are_not_reported_as_packs():
    """The demo pair registers through the same groups; it is the engine, not a pack."""
    assert settings.DISTRIBUTION_NAME.lower() not in {n.lower() for n in extras._pack_distributions()}


def test_a_checkout_hint_adds_without_removing(monkeypatch):
    """Plain `uv sync` is exact: it uninstalled the dev tools and an editable pack when
    the printed command was run. `--inexact` only adds, so the extras already present
    need not be restated either."""
    monkeypatch.setattr(settings, "running_from_checkout", lambda: True)
    monkeypatch.setattr(extras, "installed_extras", lambda: ["store", "viz"])
    hint = extras.install_hint("mcp")
    assert "uv sync --inexact --extra mcp" in hint
    assert "--extra store" not in hint
    assert "uv tool" not in hint


def _has_tomllib() -> bool:
    try:
        import tomllib  # noqa: F401
    except ImportError:
        return False
    return True


#: The product reads a uv receipt with tomllib and, by design, falls back to advice
#: without it (Python 3.10), so a test of what it reads from a receipt cannot run there.
_needs_tomllib = pytest.mark.skipif(not _has_tomllib(), reason="tomllib is Python 3.11+")

# The receipt `uv tool install --with ... --with-editable ...` really wrote (uv 0.13),
# paths shortened. Asserting against the real shape, not one imagined for the test.
RECEIPT = """
[tool]
requirements = [
    { name = "tradeflow-engine", extras = ["viz"], specifier = "==2.3.0" },
    { name = "tqdm" },
    { name = "requests", specifier = ">=2" },
    { name = "packdir", directory = "/work/packdir" },
    { name = "packdir", editable = "/work/packdir" },
]
entrypoints = [
    { name = "tradeflow", install-path = "/bin/tradeflow", from = "tradeflow-engine" },
]

[tool.options]
find-links = ["file:///wheels"]
"""


@_needs_tomllib
def test_the_uv_tool_command_restates_what_the_receipt_was_built_from(monkeypatch, tmp_path):
    """The command replaces the environment, so every extra and every `--with` package
    has to be in it - named in a parenthetical, the reader loses them on copy-paste."""
    (tmp_path / "uv-receipt.toml").write_text(RECEIPT)
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(extras, "installed_extras", lambda: [])
    hint = extras.install_hint("mcp")
    command = next(line for line in hint.splitlines() if "uv tool install" in line)
    assert f"'{settings.DISTRIBUTION_NAME}[mcp,viz]=={__version__}'" in command
    assert "--with tqdm" in command
    assert "--with 'requests>=2'" in command
    assert "--with /work/packdir" in command
    assert "--with-editable /work/packdir" in command
    assert "(detected)" in hint
    assert "re-add any packs" not in hint  # nothing left for the reader to remember


@_needs_tomllib
def test_the_uv_tool_command_restates_the_exclusions_it_was_built_with(monkeypatch, tmp_path):
    """A pack that still declares the old engine only installs cleanly because the tool
    was built with that package excluded. The rebuilt environment loses the exclusion
    unless the printed command says it, and pasting it would pull the old engine back."""
    receipt = RECEIPT.replace(
        "entrypoints = [",
        'excludes = ["tradeflow-engine"]\nentrypoints = [',
        1,
    )
    (tmp_path / "uv-receipt.toml").write_text(receipt)
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(extras, "installed_extras", lambda: [])
    hint = extras.install_hint("mcp")
    command = next(line for line in hint.splitlines() if "uv tool install" in line)
    assert f"--excludes {extras.EXCLUDES_FILE}" in command
    # The file the flag names is created by the same pasted line, so it runs from any
    # directory; actually run that part and read the file back.
    import subprocess

    create = command[command.index("printf") :].split(" && ")[0].strip()
    subprocess.run(create, shell=True, check=True, cwd=tmp_path)
    assert (tmp_path / extras.EXCLUDES_FILE).read_text() == "tradeflow-engine\n"
    assert " && uv tool install" in command

    (tmp_path / "uv-receipt.toml").write_text(RECEIPT)
    plain = extras.install_hint("mcp")
    assert "--excludes" not in plain


def test_without_a_receipt_the_hint_falls_back_to_naming_the_packs(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))  # no receipt here
    monkeypatch.setattr(extras, "installed_extras", lambda: [])
    monkeypatch.setattr(extras, "_pack_distributions", lambda: ["acme-signals"])
    assert "re-add any packs you installed with --with: acme-signals" in extras.install_hint("viz")


@_needs_tomllib
def test_a_pack_install_command_is_additive(monkeypatch, tmp_path):
    (tmp_path / "uv-receipt.toml").write_text(RECEIPT)
    monkeypatch.setattr(settings, "running_from_checkout", lambda: True)
    checkout = extras.pack_install_hint("/p")
    assert checkout.splitlines()[0] == "uv pip install -e /p"
    assert "--no-deps" in checkout and "old tradeflow-engine" in checkout
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    command = extras.pack_install_hint("/p")
    assert "--with-editable /p" in command and "--with tqdm" in command


def test_the_pack_install_fallback_says_when_not_to_skip_dependencies(monkeypatch, tmp_path):
    """With no receipt the command is a plain pip one, and pip resolves a pack's
    dependency on the old distribution name by installing it over this one. The cure is
    in the pack; --no-deps is for one that cannot be edited, and the hint says it also
    skips the pack's other dependencies."""
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))  # no receipt here
    lines = extras.pack_install_hint("/p").splitlines()
    assert lines[0] == "python -m pip install -e /p"
    note = " ".join(lines[1:])
    assert "--no-deps" in note and "cannot edit" in note and "other dependencies" in note
    assert "change that to quantred" in note


@_needs_tomllib
def test_the_uv_tool_pack_command_creates_the_exclusions_it_names(monkeypatch, tmp_path):
    receipt = RECEIPT.replace("entrypoints = [", 'excludes = ["tradeflow-engine"]\nentrypoints = [', 1)
    (tmp_path / "uv-receipt.toml").write_text(receipt)
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    command = extras.pack_install_hint("/p")
    assert command.startswith("printf ") and "> excludes.txt && uv tool install" in command
    assert "--with-editable /p" in command and "--excludes excludes.txt" in command


def test_installed_hint_never_names_an_interpreter_path(monkeypatch):
    """A recreated tool environment loses whatever was installed by hand into a
    particular interpreter, so a path to one is not a durable instruction."""
    _as_installed_copy(monkeypatch)
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: True)
    hint = extras.install_hint("mcp")
    assert "/bin/python" not in hint and "Scripts" not in hint
    assert sys.executable not in hint


def test_the_extras_guide_url_has_no_docs_prefix():
    """The site is served from its root; a `/docs/` prefix is a 404."""
    assert "/docs/" not in settings.DOCS_URL + "/usage/extras"
    assert (settings.PROJECT_ROOT / "docs" / "content" / "usage" / "extras.md").is_file()


def test_nothing_recommends_uv_tool_inject():
    """That subcommand does not exist, and being told to run it was a dead end."""
    root = settings.PROJECT_ROOT
    needle = "uv tool " + "inject"
    hits = []
    for base in (root / "quantred", root / "docs" / "content"):
        for path in base.rglob("*"):
            if path.suffix in {".py", ".md", ".mdx"} and needle in path.read_text():
                hits.append(str(path))
    assert needle not in (root / "README.md").read_text()
    assert hits == []


# --- the surface a user reaches ------------------------------------------------
def _every_module_missing(monkeypatch):
    for module in ("sklearn", "ortools", "matplotlib", "anthropic", "openai", "mcp"):
        # An already-imported submodule is found in sys.modules before its parent is
        # consulted, so blocking only the parent passes or fails depending on which
        # other test file happened to import `sklearn.gaussian_process` first.
        for loaded in [name for name in sys.modules if name.startswith(module + ".")]:
            monkeypatch.delitem(sys.modules, loaded)
        monkeypatch.setitem(sys.modules, module, None)


def _assert_installed_reader_message(text: str):
    assert "uv sync" not in text, text
    assert settings.DISTRIBUTION_NAME in text, text
    assert "uv tool install --force" in text, text


def test_no_missing_dependency_message_sends_an_installed_reader_to_a_checkout(monkeypatch, capsys):
    """Drive every guard that reports a missing extra, as an installed copy with all of
    them absent. A guard that still prints its own `uv sync` fails here, whichever one."""
    from quantred.analytics import charts, htmlreport
    from quantred.optimization.optimizer import ParameterOptimizer
    from quantred.portfolio.allocator import PortfolioAllocator

    _as_installed_copy(monkeypatch)
    _every_module_missing(monkeypatch)

    for guard in (ParameterOptimizer._make_gp, PortfolioAllocator._make_solver, charts._mpl):
        with pytest.raises(RuntimeError) as caught:
            guard()
        _assert_installed_reader_message(str(caught.value))

    # The HTML placeholder deliberately carries no command at all (see its own test).
    assert "uv sync" not in htmlreport._chart(None, "equity curve")

    for provider in ("anthropic", "openai"):
        with pytest.raises(SystemExit) as caught:
            cli.cmd_research(cli.parse_cli(["research", "--provider", provider]))
        _assert_installed_reader_message(str(caught.value.code))

    import quantred.mcp.server as server

    def no_sdk(**_):
        raise ImportError("No module named 'mcp'")

    monkeypatch.setattr(server, "serve", no_sdk)
    with pytest.raises(SystemExit) as caught:
        cli.cmd_mcp(cli.parse_cli(["mcp"]))
    _assert_installed_reader_message(str(caught.value.code))


def test_the_html_placeholder_never_carries_this_machines_install_text(monkeypatch):
    """A report is attached and forwarded. The install command names the installer, the
    version and every pack the reader added with `--with`, so it stays in the terminal."""
    from quantred.analytics import htmlreport

    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(extras, "_pack_distributions", lambda: ["acme-signals"])
    monkeypatch.setattr(extras, "_tool_receipt", lambda: tomllib_loads(RECEIPT))
    rendered = htmlreport._chart(None, "equity curve")
    assert "viz extra is not installed" in rendered
    assert f"{settings.DOCS_URL}/usage/extras" in rendered
    for leaked in ("acme-signals", "packdir", "uv tool", "pip install", "--with", "/work"):
        assert leaked not in rendered, leaked


def tomllib_loads(text):
    return pytest.importorskip("tomllib").loads(text)


def _doctor_output(argv, monkeypatch, capsys, tmp_path, installed):
    _as_installed_copy(monkeypatch, installed=installed)
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    args = cli.parse_cli(argv)
    try:
        args.func(args)
    except SystemExit:  # not ready when no keys resolve; the report printed either way
        pass
    return capsys.readouterr().out


def test_the_doctor_prints_one_install_command_scoped_to_what_was_asked(monkeypatch, capsys, tmp_path):
    """Each `uv tool install --force` replaces the environment, so two of them leave
    only the second's extras: one command. And it carries what the asked-about task and
    provider need - not a solver, a learning library and the other vendor's SDK for
    someone who asked about anthropic research. The ai extra, reported by two rows,
    appears once."""
    out = _doctor_output(
        ["init", "--check", "--for", "research", "--provider", "anthropic"],
        monkeypatch,
        capsys,
        tmp_path,
        {"viz", "mcp"},
    )
    assert out.count("uv tool install --force") == 1, out
    command = next(line for line in out.splitlines() if "uv tool install --force" in line)
    assert "ai" in command.split("[")[1].split("]")[0].split(",")
    for name in ("mcp", "viz"):  # already installed: kept, not dropped
        assert name in command
    for name in ("openai", "optimize", "portfolio"):
        assert name not in command, command
    # The rest is named on one line instead of folded into the command.
    assert "Also not installed (optional)" in out and "openai" in out.split("Also not installed")[1]


def test_a_task_with_no_missing_extra_gets_no_install_command(monkeypatch, capsys, tmp_path):
    out = _doctor_output(["init", "--check"], monkeypatch, capsys, tmp_path, {"viz", "mcp"})
    assert "uv tool install --force" not in out
    assert "Also not installed (optional)" in out


def test_the_extras_page_lists_every_extra():
    """The guide is where the install hint sends a reader, so an extra it omits is a
    capability whose recovery command points at a page that never mentions it."""
    page = (Path(settings.PROJECT_ROOT) / "docs" / "content" / "usage" / "extras.md").read_text()
    rows = [line for line in page.splitlines() if line.startswith("| `")]
    listed = {line.split("`")[1] for line in rows}
    assert set(extras.EXTRAS) <= listed, sorted(set(extras.EXTRAS) - listed)
