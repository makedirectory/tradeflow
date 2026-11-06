"""The executable's name has one definition, and nothing printed spells it out.

A reader is told what to type in help text, error messages, hints and the MCP
client configuration. Each of those that hard-codes ``tradeflow <verb>`` is a place a
rename leaves behind, and it gives no signal: the old name keeps working, so every
test stays green while the instructions go stale. This scans the package's string
literals rather than a list somebody remembered.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from tradeflow import cli, settings
from tradeflow.services import setup

PACKAGE = Path(settings.PROJECT_ROOT) / "tradeflow"

# Literals allowed to spell the name, as (path relative to the package, enclosing
# function) -> why. The import package, entry-point groups, ``TRADEFLOW_HOME`` and
# ``~/.tradeflow`` are not the executable and never match; what is listed here is the
# few places where the *word* is legitimately a string.
ALLOWED = {
    ("research/sandbox.py", "_guarded_import"): (
        "compares an import's root module against the import package name"
    ),
    ("services/analysis.py", "code_version"): (
        "stamps `tradeflow <version>` as the engine's code identity - the import "
        "package, which a rename of the executable does not touch - so provenance "
        "written before and after compares as the same code"
    ),
}

# Exact-match literals that name the executable or the server and so must come from
# the constants. The constants' own definitions are the only allowed spellings.
DEFINITIONS = {("settings.py", "<module>")}


def _verbs() -> List[str]:
    parser = cli.build_parser()
    return sorted(parser._subparsers._group_actions[0].choices)


def _docstring_nodes(tree: ast.AST) -> set:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def _strings(path: Path) -> Iterator[Tuple[int, str, str]]:
    """Every non-docstring string literal as (line, text, enclosing function).

    An f-string becomes one text with each interpolation replaced by ``\\0``, so
    ``f"tradeflow {verb}"`` is seen as an instruction rather than as two harmless
    halves.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_nodes(tree)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def enclosing(node: ast.AST) -> str:
        while id(node) in parents:
            node = parents[id(node)]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return "<module>"

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(part.value if isinstance(part, ast.Constant) else "\0" for part in node.values)
            yield node.lineno, text, enclosing(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip or isinstance(parents.get(id(node)), ast.JoinedStr):
                continue
            yield node.lineno, node.value, enclosing(node)


def hard_coded_names(root: Path = PACKAGE) -> List[str]:
    verbs = "|".join(re.escape(v) for v in _verbs())
    # `tradeflow <verb>`, `tradeflow --flag`, `tradeflow \0` (an interpolated verb),
    # `tradeflow: error` (argparse's own prefix) - but not `tradeflow.module`,
    # `tradeflow/dir`, `~/.tradeflow` or `.tradeflow/`, where a separator precedes or follows.
    instruction = re.compile(rf"(?<![\w./~-])tradeflow(?: (?:(?:{verbs}|--)\b|\0)|: error)")
    problems = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for line, text, func in _strings(path):
            if (rel, func) in ALLOWED:
                continue
            if instruction.search(text) or (text == "tradeflow" and (rel, func) not in DEFINITIONS):
                problems.append(f"{rel}:{line}: {text.strip()[:70]!r}")
    return problems


def test_no_user_facing_string_hard_codes_the_executable():
    problems = hard_coded_names()
    assert not problems, (
        "these spell out the command instead of reading settings.CLI_NAME "
        "(or settings.invocation / settings.MCP_SERVER_NAME):\n  " + "\n  ".join(problems)
    )


def test_the_scan_sees_a_hard_coded_instruction(tmp_path):
    """The guard is only worth having if it can fail; a mutation here would pass silently."""
    (tmp_path / "mod.py").write_text(
        'def hint(verb):\n    return f"run tradeflow {verb} now"\n\n'
        'def fixed():\n    print("tradeflow demo")\n\n'
        'def prog():\n    return "tradeflow"\n\n'
        'def ok():\n    """Run ``tradeflow demo``."""\n    return "import tradeflow.cli; ~/.tradeflow"\n'
    )
    found = hard_coded_names(tmp_path)
    assert sorted(int(p.split(":")[1]) for p in found) == [2, 5, 8]


def test_prog_and_version_use_the_cli_name(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CLI_NAME", "renamed")
    assert cli.build_parser().prog == "renamed"
    assert cli._version_banner().startswith("renamed ")
    with pytest.raises(SystemExit):
        cli.parse_cli(["research", "info"])
    assert capsys.readouterr().err.startswith("renamed: error:")


def test_instructions_follow_the_constant(monkeypatch):
    monkeypatch.setattr(settings, "CLI_NAME", "renamed")
    monkeypatch.setattr(settings, "running_from_checkout", lambda: False)
    assert setup.invocation("trials show x") == "renamed trials show x"
    assert cli._invocation("demo") == "renamed demo"
    assert "renamed init" in settings._missing_credentials_message(["APCA_API_KEY_ID"])


def test_the_mcp_launch_command_finds_the_script_by_its_current_then_legacy_name(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text("")
    monkeypatch.setattr("sys.executable", str(python))
    monkeypatch.setattr(settings, "_looks_like_checkout", lambda path: False)
    monkeypatch.setattr(settings, "CLI_NAME", "renamed")
    monkeypatch.setattr(settings, "LEGACY_CLI_NAMES", ("older",))

    assert setup.mcp_launch_command() == [str(python), "-m", "tradeflow.cli", "mcp"]
    (bin_dir / "older").write_text("")
    assert setup.mcp_launch_command() == [str(bin_dir / "older"), "mcp"]
    (bin_dir / "renamed").write_text("")
    assert setup.mcp_launch_command() == [str(bin_dir / "renamed"), "mcp"]


def test_the_mcp_server_name_is_one_definition():
    from tradeflow.mcp import server

    assert server.SERVER_NAME == settings.MCP_SERVER_NAME
    assert list(setup.mcp_client_config()["mcpServers"]) == [settings.MCP_SERVER_NAME]
