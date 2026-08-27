"""CLI guardrails for live portfolio sizing."""

import logging

from tradeflow.cli import _warn_if_portfolio_sizer_exceeds_strategy_limit
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy


def test_live_portfolio_warns_when_allocator_exceeds_strategy_position_limit(caplog):
    strategy = VolumeSpikeStrategy.create_with_defaults()

    caplog.set_level(logging.WARNING)
    _warn_if_portfolio_sizer_exceeds_strategy_limit(strategy, ["AAA", "BBB"])

    assert "Portfolio allocator funded 2 names" in caplog.text
    assert "position_limits.max_positions=1" in caplog.text


def test_live_portfolio_allocation_within_strategy_position_limit_is_quiet(caplog):
    strategy = VolumeSpikeStrategy.create_with_defaults()
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": 2}

    caplog.set_level(logging.WARNING)
    _warn_if_portfolio_sizer_exceeds_strategy_limit(strategy, ["AAA", "BBB"])

    assert caplog.text == ""
