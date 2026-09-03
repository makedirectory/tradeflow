"""Window bounds parsed off the command line.

`--as-of` accepts a time and a zone; `--start`/`--end` default to a naive
`datetime.now()`. Anything that returns both kinds hands the rest of the program a
pair it cannot compare, so the coercion belongs here, once, where the value is parsed.
"""

from datetime import datetime, timedelta

import pytest

from tradeflow.cli import _date, build_parser, parse_cli
from tradeflow.utils.timeutils import NEW_YORK


def test_a_bare_date_means_that_market_date_in_the_exchange_zone():
    """One date contract. `2026-08-22` is that session, not UTC midnight.

    Two readings of one string is what broke an offline scan: `cache warm --end DATE`
    recorded coverage through 00:00Z because the store reads a naive datetime as UTC,
    while `scan --as-of DATE` asked for 04:00Z because the scanner reads one as New
    York - a four-hour hole in a cache that held exactly the right daily bar.
    """
    parsed = _date("2026-08-22")

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == NEW_YORK.localize(datetime(2026, 8, 22)).utcoffset()
    assert (parsed.year, parsed.month, parsed.day, parsed.hour) == (2026, 8, 22, 0)


def test_the_cache_and_the_scanner_now_read_a_date_the_same_way():
    """The defect itself, pinned end to end rather than through either component."""
    from tradeflow.scanners.symbol_scanner import resolve_scan_clock
    from tradeflow.store.bars import _to_utc

    parsed = _date("2026-08-22")

    assert _to_utc(parsed) == _to_utc(resolve_scan_clock(parsed))


def test_an_explicit_zone_is_converted_not_discarded():
    """A zone is a shift, not a label: dropping it would move a session boundary onto
    a different trading day."""
    assert _date("2026-08-22T16:00:00Z") == _date("2026-08-22T12:00:00-04:00")
    assert _date("2026-08-22T16:00:00Z").hour == 12  # noon in New York in August


@pytest.mark.parametrize("value", ["2026-08-22", "2026-08-22T16:00:00Z", "2026-08-22T12:00:00-04:00"])
def test_every_accepted_form_carries_the_zone(value):
    """The property the contract rests on: a naive value has no zone to disagree
    about, which is exactly how the two readings diverged in the first place."""
    assert _date(value).tzinfo is not None


def test_the_window_defaults_share_the_contract():
    """Defaults that stayed naive would reintroduce the mixed-awareness comparison the
    contract exists to remove — which is how the previous round of this broke."""
    args = build_parser().parse_args(["backtest", "--strategy", "demo_trend"])

    assert args.start.tzinfo is not None
    assert args.end.tzinfo is not None
    assert isinstance(args.end - args.start, timedelta)


def test_a_typed_end_still_compares_against_a_defaulted_start():
    args = build_parser().parse_args(
        ["backtest", "--strategy", "demo_trend", "--end", "2026-08-22T16:00:00Z"]
    )

    assert isinstance(args.end - args.start, timedelta)


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


# --- surface drift across commands ---------------------------------------------
#: Options that are layered, not per-command: they mean the same thing everywhere they
#: appear, so a command offering one and not reading it - or a second definition
#: drifting from the first - is a defect rather than a design choice.
_LAYERED_OPTIONS = {
    "--config": ("backtest", "live", "verdict", "info", "alphas", "horizon", "allocate", "risk"),
    "--re-resolve-universe": ("backtest", "live", "verdict", "info", "alphas", "horizon", "allocate", "risk"),
    "--cache": (
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
    ),
    "--offline": (
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
    ),
}


@pytest.mark.parametrize("option,expected", sorted(_LAYERED_OPTIONS.items()))
def test_a_layered_option_reaches_every_command_that_should_have_it(option, expected):
    """Four separate instances of the same drift in two months.

    `--cache`/`--offline` reached three commands and not five read-only ones;
    `--config` reached two and not six; `--benchmark` reached `backtest` and not
    `walkforward`; and `--re-resolve-universe` reached six and missed the two the
    defect was reported against. Each was found by a user crossing two commands that
    should have agreed.
    """
    subparsers = build_parser()._subparsers._group_actions[0].choices
    missing = [
        command
        for command in expected
        if option not in {opt for action in subparsers[command]._actions for opt in action.option_strings}
    ]

    assert not missing, f"{option} missing from {missing}"


def test_a_layered_option_has_exactly_one_definition():
    """The fourth instance was not a missing call - it was a *second definition*.

    `--config` was declared twice and the two drifted, so flags added to the newer one
    reached six commands and silently missed the two that used the older. Coverage
    alone cannot catch that: both definitions produce a flag, so every command looks
    served right up until the two disagree about what it does.

    Identical help text across commands is the observable proxy for one definition.
    """
    subparsers = build_parser()._subparsers._group_actions[0].choices
    for option, expected in _LAYERED_OPTIONS.items():
        helps = set()
        for command in expected:
            for action in subparsers[command]._actions:
                if option in action.option_strings:
                    helps.add(action.help)
        assert len(helps) == 1, f"{option} has {len(helps)} different definitions across commands"


def test_live_is_deliberately_excluded_from_the_cache_options():
    """The list is a decision, not an inventory: the trade clock reads the market as it
    is, so `live` must *not* acquire these however many other commands do."""
    subparsers = build_parser()._subparsers._group_actions[0].choices
    flags = {opt for action in subparsers["live"]._actions for opt in action.option_strings}

    assert "--cache" not in flags and "--offline" not in flags
    assert "--config" in flags  # but it does take a saved run configuration


def test_the_cost_curve_defaults_on_where_a_promotion_is_decided():
    """Opt-in on `backtest`, where the loop is interactive and each point is a full
    re-run; on by default on `walkforward`, which is already expensive and is where the
    promotion decision actually happens - cost sensitivity belongs in that story rather
    than in an optional follow-up."""
    parse = build_parser().parse_args

    assert parse(["walkforward", "--strategy", "demo_trend"]).cost_stress == "all"
    assert parse(["backtest", "--strategy", "demo_trend"]).cost_stress is None


def test_the_default_can_be_turned_off_without_argument_juggling():
    """A default that costs time needs a plain way out, not `--cost-stress none`."""
    parse = build_parser().parse_args

    assert parse(["walkforward", "--strategy", "demo_trend", "--no-cost-stress"]).cost_stress is None
    assert (
        parse(["walkforward", "--strategy", "demo_trend", "--cost-stress", "borrow"]).cost_stress == "borrow"
    )


def test_an_interrupted_command_exits_cleanly_and_says_what_did_not_happen():
    """Ctrl-C during a walk-forward printed a Python traceback ending in
    `KeyboardInterrupt`, which reads as a crash rather than a choice.

    The message matters more than the tidiness: a research command interrupted part-way
    has written no config and journaled no trial, and a half-finished validation that
    silently recorded one would corrupt the campaign's own trial count - the number
    every deflated Sharpe deflates against.
    """
    from unittest import mock

    from tradeflow import cli

    def interrupted(args):
        raise KeyboardInterrupt

    namespace = mock.Mock()
    namespace.func = interrupted

    with mock.patch.object(cli, "parse_cli", return_value=namespace), mock.patch.object(cli, "setup_logging"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main()

    message = str(exit_info.value)
    assert "Interrupted" in message
    assert "no config saved" in message and "no trial recorded" in message
