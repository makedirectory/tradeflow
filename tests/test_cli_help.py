"""The top-level help is a map a newcomer can read, not a list in definition order."""

import pytest

from tradeflow import cli


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
