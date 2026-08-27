"""Startup guardrails for `live --portfolio`.

The allocator's cardinality cap and the strategy's own `position_limits.max_positions`
are two bounds on the same thing. When they disagree the live book silently becomes
the smaller of the two, filled by whichever signals arrive first — so the deployment
trades a book nobody allocated and nobody validated. Startup refuses that rather than
reporting it.
"""

import pytest

from tradeflow.cli import _refuse_contradictory_portfolio_cardinality, build_parser, cmd_live
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy


def _strategy(max_positions):
    strategy = VolumeSpikeStrategy.create_with_defaults()
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": max_positions}
    return strategy


def test_an_allocator_that_funds_more_names_than_the_book_holds_is_refused():
    """The bug: this warned and then traded anyway.

    A warning on a process that goes on to place orders for hours under a book a
    fifth of the intended size is not a guardrail. Both numbers are known before the
    broker is reached, so there is nothing to discover at runtime.
    """
    with pytest.raises(SystemExit) as exit_info:
        _refuse_contradictory_portfolio_cardinality(_strategy(1), 5)

    message = str(exit_info.value)
    # Both numbers and both remedies, or the operator cannot act on it.
    assert "up to 5 names" in message and "at most 1" in message
    assert "position_limits.max_positions to 5" in message
    assert "--max-positions 1" in message


def test_caps_that_agree_exactly_are_allowed():
    """The boundary case. A guard that also rejects this rejects everything."""
    _refuse_contradictory_portfolio_cardinality(_strategy(5), 5)


def test_a_book_larger_than_the_allocator_funds_is_allowed():
    """Holding room to spare is not a contradiction — only the reverse is."""
    _refuse_contradictory_portfolio_cardinality(_strategy(10), 3)


def test_an_unset_limit_is_not_treated_as_zero():
    """Absent is not zero: no declared limit means nothing to contradict."""
    strategy = VolumeSpikeStrategy.create_with_defaults()
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": None}
    _refuse_contradictory_portfolio_cardinality(strategy, 5)


def test_the_default_live_portfolio_invocation_is_refused_before_the_broker_is_reached(monkeypatch):
    """Every shipped strategy declares max_positions=1 and --max-positions defaults to 5.

    So the default `live --portfolio` has always funded five names and held one. The
    refusal must land before `build_data_and_broker`, or the operator pays a
    credential check, a market-data fetch and a solve to be told the configuration
    could never have worked.
    """
    from tradeflow import cli

    def _fail(*args, **kwargs):
        raise AssertionError("the broker was reached before the cardinality check")

    monkeypatch.setattr(cli, "build_data_and_broker", _fail)
    args = build_parser().parse_args(["live", "--strategy", "volume_spike", "--portfolio"])

    with pytest.raises(SystemExit) as exit_info:
        cmd_live(args)
    assert "--portfolio would fund up to 5 names" in str(exit_info.value)
