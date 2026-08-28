"""Window bounds parsed off the command line.

`--as-of` accepts a time and a zone; `--start`/`--end` default to a naive
`datetime.now()`. Anything that returns both kinds hands the rest of the program a
pair it cannot compare, so the coercion belongs here, once, where the value is parsed.
"""

from datetime import datetime, timedelta

import pytest

from tradeflow.cli import _date, build_parser
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
