"""The CLI's help text and the one mistake it now explains instead of rejecting."""

from __future__ import annotations

import argparse

import pytest

from tradeflow import cli
from tradeflow.settings import DOCS_URL
from tradeflow.utils.extras import EXTRAS


def test_research_info_suggests_the_sibling_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cli(["research", "info", "--symbols", "A"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "info --help" in err
    assert "unrecognized arguments" not in err


@pytest.mark.parametrize("sibling", ["verdict", "backtest", "walkforward"])
def test_the_hint_is_not_special_to_info(sibling, capsys):
    with pytest.raises(SystemExit):
        cli.parse_cli(["research", sibling])
    assert f"{sibling} --help" in capsys.readouterr().err


def test_a_real_research_flag_is_not_reinterpreted():
    args = cli.parse_cli(["research", "--goal", "info"])
    assert args.command == "research"
    assert args.goal == "info"


def test_the_hint_finds_the_sibling_word_after_other_options(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_cli(["research", "--goal", "x", "info", "--symbols", "A"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "info --help" in err and "unrecognized arguments" not in err


def test_a_flag_value_that_names_a_command_is_still_a_value():
    args = cli.parse_cli(["research", "--goal", "verdict", "--provider", "ollama"])
    assert args.goal == "verdict"


def test_an_unknown_word_after_research_keeps_argparses_own_error(capsys):
    with pytest.raises(SystemExit):
        cli.parse_cli(["research", "notacommand"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_info_still_rejects_research_only_flags():
    with pytest.raises(SystemExit) as exc:
        cli.parse_cli(["info", "--provider", "ollama", "--symbols", "A"])
    assert exc.value.code == 2


def test_top_level_help_names_the_extras_guide(capsys):
    with pytest.raises(SystemExit):
        cli.parse_cli(["--help"])
    out = capsys.readouterr().out
    assert f"{DOCS_URL}/usage/extras" in out
    for name in EXTRAS:
        assert name in out


def test_only_the_top_level_parser_keeps_raw_formatting():
    """A raw formatter on a subcommand would stop its help from wrapping."""
    parser = cli.build_parser()
    sub = parser._subparsers._group_actions[0].choices["backtest"]
    assert sub.formatter_class is argparse.HelpFormatter
    assert parser.formatter_class is argparse.RawDescriptionHelpFormatter


def test_init_help_mentions_the_model_and_readiness_options(capsys):
    with pytest.raises(SystemExit):
        cli.parse_cli(["init", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    for token in ("--for", "--provider", "--probe-model", "/usage/extras"):
        assert token in out
