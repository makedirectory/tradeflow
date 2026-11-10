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


# --- the flags every research run turns on ---------------------------------
# `--start`, `--end`, `--capital` and `--commission-bps` appear on most commands, and a
# default is part of a run's identity, yet `--help` showed them with no text at all - so
# a reader could not tell that `backtest` reads 30 days and `verdict` a year, or what
# unit a commission is in.

#: Flags whose help must exist and state the default.
CORE = {"--start", "--end", "--capital", "--commission-bps"}
WALKFORWARD = {
    "--holdout-days",
    "--method",
    "--objective",
    "--mode",
    "--seed",
    "--train-days",
    "--test-days",
    "--folds",
}


def _commands():
    parser = cli.build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def _action(sub, flag):
    return next((a for a in sub._actions if flag in a.option_strings), None)


def _cases():
    commands = _commands()
    for name, sub in commands.items():
        for flag in sorted(CORE):
            action = _action(sub, flag)
            # A None default ("from the config", "the cached extent") is explained in
            # prose rather than as a value, and is covered by the commands' own tests.
            if action is not None and action.default is not None:
                yield name, flag
    for flag in sorted(WALKFORWARD):
        yield "walkforward", flag
    yield "demo", "--seed"
    yield "optimize", "--output"
    yield "optimize", "--method"
    yield "optimize", "--objective"


@pytest.mark.parametrize("command, flag", list(_cases()))
def test_a_core_flag_says_what_it_defaults_to(command, flag):
    action = _action(_commands()[command], flag)
    assert action is not None, f"{command} has no {flag}"
    assert action.help, f"{command} {flag} has no help text"
    assert "default" in action.help, f"{command} {flag} does not state its default"


def test_each_command_states_its_own_date_window():
    """Two windows exist (30 days on the search commands, a year on verdict/info/horizon),
    so a help string copied from one to the other would state the wrong one."""
    seen = set()
    for name, sub in _commands().items():
        start, end = _action(sub, "--start"), _action(sub, "--end")
        if start is None or end is None or start.default is None or end.default is None:
            continue
        days = round((end.default - start.default).total_seconds() / 86400)
        assert f"{days} days before --end" in start.help, f"{name} --start: {start.help!r}"
        seen.add(days)
    assert seen == {30, 365}


def test_cache_dir_help_names_the_real_default():
    """It said `default: cache/bars`, a relative path, while the cache lives under the
    state root - so a reader looked for it in whatever directory they were in."""
    for name, sub in _commands().items():
        action = _action(sub, "--cache-dir")
        if action is not None and action.default is None:
            assert "state root" in action.help, f"{name} --cache-dir: {action.help!r}"


@pytest.mark.parametrize("group", ["trials", "cache"])
def test_a_bare_command_group_prints_its_subcommands_not_an_internal_name(group, capsys):
    """`quantred trials` failed with "the following arguments are required:
    trials_command" - argparse's internal destination, which names nothing a reader
    can type. Bare, a group now prints its own help, as the bare program does."""
    with pytest.raises(SystemExit) as exited:
        cli.parse_cli([group])
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert f"{group}_command" not in out
    assert "status" in out  # a subcommand both groups have


@pytest.mark.parametrize("group", ["trials", "cache"])
def test_a_group_error_names_the_subcommand_in_words(group, capsys):
    """With an option but no subcommand it is still an error, phrased for a reader."""
    with pytest.raises(SystemExit) as exited:
        cli.parse_cli([group, "--help-me"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert f"{group}_command" not in err
