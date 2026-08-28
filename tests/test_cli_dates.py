"""Window bounds parsed off the command line.

`--as-of` accepts a time and a zone; `--start`/`--end` default to a naive
`datetime.now()`. Anything that returns both kinds hands the rest of the program a
pair it cannot compare, so the coercion belongs here, once, where the value is parsed.
"""

from datetime import datetime, timedelta

import pytest

from tradeflow.cli import _date, build_parser, parse_cli
from tradeflow.utils.timeutils import NEW_YORK


def test_a_plain_date_still_parses():
    assert _date("2024-06-01") == datetime(2024, 6, 1)


def test_an_aware_value_becomes_the_same_instant_on_the_exchange_clock():
    """A zone is a shift, not a label. Dropping it without converting would move the
    value by the offset, which for a session boundary is a different trading day."""
    parsed = _date("2024-06-01T16:00:00Z")

    assert parsed.tzinfo is None
    assert NEW_YORK.localize(parsed) == datetime.fromisoformat("2024-06-01T16:00:00+00:00")


def test_the_same_instant_written_two_ways_parses_identically():
    assert _date("2024-06-01T16:00:00Z") == _date("2024-06-01T12:00:00-04:00")


@pytest.mark.parametrize("value", ["2024-06-01", "2024-06-01T16:00:00Z", "2024-06-01T12:00:00-04:00"])
def test_every_accepted_form_is_naive(value):
    """The property the rest of the program depends on. One aware value anywhere in
    the set reintroduces the mixed comparison."""
    assert _date(value).tzinfo is None


def test_an_aware_end_can_be_compared_against_a_defaulted_start():
    """The regression.

    The ISO fallback accepted a zone that nothing downstream could use: argparse took
    the value and the first comparison raised `can't subtract offset-naive and
    offset-aware datetimes` inside a window calculation. Before the fallback existed
    argparse rejected the same value cleanly at the command line, so widening what was
    accepted without normalizing it traded a good error for a bad one.
    """
    args = build_parser().parse_args(
        ["backtest", "--strategy", "volume_spike", "--end", "2024-06-01T16:00:00Z"]
    )

    assert isinstance(args.end - args.start, timedelta)  # used to raise TypeError
    assert args.end == datetime(2024, 6, 1, 12)  # 16:00Z is noon in New York in June


def test_every_command_that_scans_a_historical_window_can_pin_its_scan_clock():
    """`cache warm` resolved the scanner at `--end` with no way to override it, while
    every other historical command took `--scan-as-of`. A cache warmed for a
    deliberately different selection clock was simply not expressible."""
    parser = build_parser()
    subparsers = parser._subparsers._group_actions[0].choices

    for command in ("backtest", "optimize", "walkforward", "research"):
        flags = {opt for action in subparsers[command]._actions for opt in action.option_strings}
        assert "--scan-as-of" in flags, f"{command} cannot pin its scan clock"

    warm = subparsers["cache"]._subparsers._group_actions[0].choices["warm"]
    assert "--scan-as-of" in {opt for action in warm._actions for opt in action.option_strings}


def test_live_deliberately_has_no_scan_clock_to_pin():
    """A live book is selected from the universe as it stands, not as it stood at the
    end of a window — so `live` resolving at wall-clock now is the intended behaviour,
    and the absence of the flag is the design rather than an omission."""
    subparsers = build_parser()._subparsers._group_actions[0].choices
    flags = {opt for action in subparsers["live"]._actions for opt in action.option_strings}

    assert "--scanner" in flags
    assert "--scan-as-of" not in flags


# --- the CLI's own output ----------------------------------------------------
def test_no_help_string_contains_an_unescaped_percent():
    """argparse %-formats help text against the action's own attributes.

    `--param-sensitivity` read "Perturb chosen params +-10% and re-test", so `% a`
    became a format spec and `--help` printed the action's entire `__dict__` in the
    middle of the sentence. A literal percent has to be doubled, and the rule is
    asserted over every command rather than the one that happened to break.
    """
    import re

    subparsers = build_parser()._subparsers._group_actions[0].choices
    offenders = [
        (command, action.option_strings or [action.dest], action.help)
        for command, sub in subparsers.items()
        for action in sub._actions
        if action.help and re.search(r"(?<!%)%(?!%)", action.help)
    ]
    assert not offenders, f"unescaped % in help text: {offenders}"


def test_every_command_renders_its_own_help_without_leaking_internals():
    """The defect was only visible once help was *formatted*, not when it was declared,
    so the check has to render every command rather than inspect the strings."""
    subparsers = build_parser()._subparsers._group_actions[0].choices
    for command, sub in subparsers.items():
        rendered = sub.format_help()
        assert "option_strings" not in rendered, f"{command} --help leaked argparse internals"
        assert "_ArgumentGroup" not in rendered, f"{command} --help leaked argparse internals"


# --- reading a warm cache from every command that reads bars -----------------
def test_every_read_only_bar_command_can_use_the_cache():
    """`scan`, `alphas`, `risk`, `horizon`, `allocate` and `info` had neither flag.

    They built a live client regardless, so a warm cache was unusable from them and a
    DNS failure degraded them into empty or insufficient-data results rather than
    reading the bars already on disk.

    This list is what makes an omission *fail*: the wiring test below is parametrized
    over whichever commands declare the flags, so a command declaring none is simply
    invisible to it. `info` was missed exactly that way.
    """
    subparsers = build_parser()._subparsers._group_actions[0].choices
    for command in (
        "scan",
        "alphas",
        "risk",
        "horizon",
        "allocate",
        "info",
        "backtest",
        "optimize",
        "walkforward",
        "verdict",
    ):
        flags = {opt for action in subparsers[command]._actions for opt in action.option_strings}
        assert {"--cache", "--offline", "--cache-dir"} <= flags, f"{command} cannot read the cache"


def test_live_deliberately_cannot_be_run_offline():
    """The trade clock reads the market as it is. A cached live run is not a thing."""
    subparsers = build_parser()._subparsers._group_actions[0].choices
    flags = {opt for action in subparsers["live"]._actions for opt in action.option_strings}
    assert "--offline" not in flags and "--cache" not in flags


def _commands_declaring_cache_flags():
    subparsers = build_parser()._subparsers._group_actions[0].choices
    return sorted(
        name
        for name, sub in subparsers.items()
        if "--cache" in {opt for action in sub._actions for opt in action.option_strings}
    )


@pytest.mark.parametrize("command", _commands_declaring_cache_flags())
def test_the_cache_flags_reach_the_data_client_of_every_command_that_offers_them(command, monkeypatch):
    """A flag that parses and is never read is worse than no flag: it advertises a
    capability the command does not have.

    This was first written against `scan` alone, and a parser-shaped check for the
    rest — which certified `allocate` as fixed while `_allocate_utility`, the branch
    `--objective utility` takes, still built a live client and ignored `--offline`.
    Every command that offers the flags is exercised now, because the hole was in the
    coverage rather than in the rule.
    """
    from tradeflow import cli

    seen = {}

    def _spy(cache=False, offline=False, cache_dir=None):
        seen.update(cache=cache, offline=offline)
        raise RuntimeError("stop once the client would have been built")

    monkeypatch.setattr(cli, "build_data_and_broker", _spy)
    args = parse_cli([command, "--offline"])
    with pytest.raises(RuntimeError):
        args.func(args)

    assert seen.get("offline") is True, f"{command} ignored --offline"


def test_the_allocate_utility_book_reads_the_cache_too(monkeypatch):
    """`--objective utility` is a different function, and it was the one that ignored
    the flags its own command declared."""
    from tradeflow import cli

    seen = {}

    def _spy(cache=False, offline=False, cache_dir=None):
        seen.update(cache=cache, offline=offline)
        raise RuntimeError("stop")

    monkeypatch.setattr(cli, "build_data_and_broker", _spy)
    args = parse_cli(["allocate", "--objective", "utility", "--offline"])
    with pytest.raises(RuntimeError):
        args.func(args)

    assert seen.get("offline") is True


def test_an_offline_scan_says_its_universe_is_only_as_current_as_the_cache(monkeypatch, capsys):
    """Nothing errors when coverage ends before the scan clock — the newest cached bar
    just becomes "the latest", so a universe picked from stale bars looks exactly like
    one picked from fresh bars. That case has to announce itself."""
    from tests.fakes import DictMarketData, make_ohlcv
    from tradeflow import cli
    from tradeflow.marketdata.client import MarketDataClient

    client = MarketDataClient(DictMarketData({"AAA": make_ohlcv(n=60, seed=0, freq="1D")}))
    monkeypatch.setattr(cli, "build_data_and_broker", lambda **kw: (None, client))

    cli.cmd_scan(build_parser().parse_args(["scan", "--offline", "--symbols", "AAA"]))
    offline_output = capsys.readouterr().out
    assert "OFFLINE" in offline_output
    assert "as current as the cache" in offline_output

    # And stays quiet when the run can actually reach the network, or the notice
    # becomes noise that means nothing.
    cli.cmd_scan(build_parser().parse_args(["scan", "--symbols", "AAA"]))
    assert "OFFLINE" not in capsys.readouterr().out
