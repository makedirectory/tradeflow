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
import tomllib

from tradeflow import __version__, cli, settings
from tradeflow.utils import extras

PYPROJECT = Path(settings.PROJECT_ROOT) / "pyproject.toml"

#: Where the import name differs from the distribution name. Everything else is the same.
IMPORT_NAME = {"scikit-learn": "sklearn"}


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement.strip(), maxsplit=1)[0].lower()


def _published_extras() -> dict:
    manifest = tomllib.loads(PYPROJECT.read_text())
    return manifest["project"]["optional-dependencies"]


def _as_installed_copy(monkeypatch, installed=()):
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    monkeypatch.setattr(extras, "installed_extras", lambda: sorted(installed))
    monkeypatch.setattr(extras, "_pack_distributions", lambda: [])
    monkeypatch.setattr(extras, "_in_uv_tool_env", lambda: False)


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
    assert "none detected" in extras.install_hint("viz")


def test_this_distributions_own_entry_points_are_not_reported_as_packs():
    """The demo pair registers through the same groups; it is the engine, not a pack."""
    assert settings.DISTRIBUTION_NAME.lower() not in {n.lower() for n in extras._pack_distributions()}


def test_a_checkout_hint_is_uv_sync_with_every_extra(monkeypatch):
    """`uv sync` removes whatever is not named, so the union is the command."""
    monkeypatch.setattr(settings, "running_from_checkout", lambda: True)
    monkeypatch.setattr(extras, "installed_extras", lambda: ["store", "viz"])
    hint = extras.install_hint("mcp")
    assert "uv sync --extra mcp --extra store --extra viz" in hint
    assert "uv tool" not in hint


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
    for base in (root / "tradeflow", root / "docs" / "content"):
        for path in base.rglob("*"):
            if path.suffix in {".py", ".md", ".mdx"} and needle in path.read_text():
                hits.append(str(path))
    assert needle not in (root / "README.md").read_text()
    assert hits == []


# --- the surface a user reaches ------------------------------------------------
def _every_module_missing(monkeypatch):
    for module in ("sklearn", "ortools", "matplotlib", "anthropic", "openai", "mcp"):
        monkeypatch.setitem(sys.modules, module, None)


def _assert_installed_reader_message(text: str):
    assert "uv sync" not in text, text
    assert settings.DISTRIBUTION_NAME in text, text
    assert "uv tool install --force" in text, text


def test_no_missing_dependency_message_sends_an_installed_reader_to_a_checkout(monkeypatch, capsys):
    """Drive every guard that reports a missing extra, as an installed copy with all of
    them absent. A guard that still prints its own `uv sync` fails here, whichever one."""
    from tradeflow.analytics import charts, htmlreport
    from tradeflow.optimization.optimizer import ParameterOptimizer
    from tradeflow.portfolio.allocator import PortfolioAllocator

    _as_installed_copy(monkeypatch)
    _every_module_missing(monkeypatch)

    for guard in (ParameterOptimizer._make_gp, PortfolioAllocator._make_solver, charts._mpl):
        with pytest.raises(RuntimeError) as caught:
            guard()
        _assert_installed_reader_message(str(caught.value))

    _assert_installed_reader_message(htmlreport._chart(None, "equity curve"))

    for provider in ("anthropic", "openai"):
        with pytest.raises(SystemExit) as caught:
            cli.cmd_research(cli.parse_cli(["research", "--provider", provider]))
        _assert_installed_reader_message(str(caught.value.code))

    import tradeflow.mcp.server as server

    def no_sdk():
        raise ImportError("No module named 'mcp'")

    monkeypatch.setattr(server, "serve", no_sdk)
    with pytest.raises(SystemExit) as caught:
        cli.cmd_mcp(cli.parse_cli(["mcp"]))
    _assert_installed_reader_message(str(caught.value.code))


def test_the_html_placeholder_escapes_the_hint(monkeypatch):
    from tradeflow.analytics import htmlreport

    monkeypatch.setattr(extras, "install_hint", lambda *a: "pip install 'x[viz]' <b>&")
    rendered = htmlreport._chart(None, "equity curve")
    assert "<b>" not in rendered and "&lt;b&gt;" in rendered
    assert "<pre" in rendered
