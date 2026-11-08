"""The top-level help is a map a newcomer can read, not a list in definition order."""

import argparse

import pytest

from quantred import cli


def _subcommands():
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    return set(action.choices)


def test_every_command_appears_in_exactly_one_help_group():
    """The grouped listing replaces argparse's own, so a command missing from the groups
    would vanish from `--help` entirely, and one in two groups would be listed twice."""
    grouped = [name for _, names in cli.COMMAND_GROUPS for name in names]
    assert sorted(grouped) == sorted(set(grouped)), "a command is in two groups"
    assert set(grouped) == _subcommands()


def test_the_help_starts_with_what_a_newcomer_needs(capsys):
    """Live trading used to be listed before `init` and `demo`, which sat at the very end
    of one long list, under a brace repeating all 26 names."""
    with pytest.raises(SystemExit):
        cli.parse_cli(["--help"])
    text = capsys.readouterr().out
    assert text.splitlines()[0].endswith("<command> [options]")
    assert "{backtest," not in text
    assert text.index("start here:") < text.index("  init ") < text.index("  live ")
    assert "small-real places real orders" in text


def test_the_bare_command_prints_the_map_rather_than_an_error(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.parse_cli([])
    assert exited.value.code == 0
    assert "start here:" in capsys.readouterr().out


def test_the_setup_check_keeps_its_columns_and_names_the_state_root_once(capsys, monkeypatch, tmp_path):
    """A long row name pushed its own detail out of the column, and every path under the
    state root repeated the full root, burying the part that differs."""
    from quantred import settings
    from quantred.services.setup import Check

    monkeypatch.setattr(settings, "state_root", lambda: tmp_path)
    checks = [
        Check("version", "ready", "x"),
        Check("state outside version control", "ready", "not inside a git working tree"),
        Check("research journal", "ready", f"{tmp_path}/logs/research_journal.jsonl"),
    ]
    cli._print_doctor(checks, task="demo")
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  [ok]")]
    columns = {row.index(detail) for row, detail in zip(rows, ["x", "not inside", "<state root>"])}
    assert len(columns) == 1
    assert "<state root>/logs/research_journal.jsonl" in rows[2]


def test_the_state_root_is_abbreviated_only_where_it_starts_a_path():
    """A plain substring replace rewrote `/mnt/data/x` under a root of `/data`."""
    assert cli._under_state_root("/data/logs/x is writable", "/data") == "<state root>/logs/x is writable"
    assert cli._under_state_root("/mnt/data/x", "/data") == "/mnt/data/x"
    assert cli._under_state_root("/data2/x", "/data") == "/data2/x"


def _subparser(*path):
    parser = cli.build_parser()
    for name in path:
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        parser = action.choices[name]
    return parser


def test_paging_and_sort_options_state_their_defaults_in_help():
    """`trials list --help` showed --sort and --limit with no default and --limit with no
    help at all, so the reader could not tell what a bare `trials list` would do."""
    for path in (("trials", "list"), ("trials", "best")):
        parser = _subparser(*path)
        for action in parser._actions:
            if action.dest in ("sort", "limit", "offset") and action.help != argparse.SUPPRESS:
                text = parser._get_formatter()._expand_help(action)
                assert text, f"{path} {action.option_strings} has no help"
                assert f"default: {action.default}" in text, (path, action.dest, text)


def test_scan_and_alphas_describe_their_strategy_and_symbols_flags():
    """Flags with no help line left `--symbols` and `--strategy` as bare names in --help."""
    for command in ("scan", "alphas"):
        parser = _subparser(command)
        for action in parser._actions:
            if action.dest in ("symbols", "strategy"):
                assert action.help, f"{command} {action.option_strings} has no help"
