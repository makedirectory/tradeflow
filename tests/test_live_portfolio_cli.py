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


# --- preflight and broker mode --------------------------------------------------
def _preflight_output(capsys, monkeypatch, *, paper=True, capital=8_000.0, ledger_path="/tmp/l.jsonl"):
    from unittest import mock

    from tests.fakes import FakeBroker
    from tradeflow import cli
    from tradeflow.services.registry import STRATEGIES

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: paper)
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": 8}
    args = build_parser().parse_args(["live", "--preflight", "--scanner", "none"])
    ledger = mock.Mock()
    ledger.path = ledger_path

    cli._print_live_preflight(
        args, strategy, FakeBroker(buying_power=100_000.0), [f"S{i}" for i in range(61)], capital, ledger
    )
    return capsys.readouterr().out


def test_preflight_states_the_contract_the_run_will_trade_under(capsys, monkeypatch):
    """Printed on every live run, not only under --preflight: a check you have to
    remember to ask for is one that gets skipped exactly when it matters."""
    printed = _preflight_output(capsys, monkeypatch)

    assert "broker mode" in printed and "PAPER" in printed
    assert "capital this run" in printed and "8,000" in printed
    assert "61 symbols" in printed
    assert "max positions" in printed


def test_preflight_shows_capital_beside_the_account_it_differs_from(capsys, monkeypatch):
    """The discrepancy that would invalidate the telemetry: $8,000 of intent against a
    $100,000 paper balance. Both numbers, on adjacent lines."""
    printed = _preflight_output(capsys, monkeypatch)

    assert "100,000" in printed  # what the venue handed out
    assert "8,000" in printed  # what this run may deploy


def test_preflight_names_where_the_telemetry_lands(capsys, monkeypatch):
    """Telemetry nobody can find is telemetry nobody checks, and the drift gates are
    worthless without it."""
    printed = _preflight_output(capsys, monkeypatch, ledger_path="/tmp/findme/ledger.jsonl")

    assert "/tmp/findme/ledger.jsonl" in printed
    assert "journal" in printed and "halt state" in printed


def test_a_disabled_ledger_says_so_rather_than_printing_nothing(capsys, monkeypatch):

    from tests.fakes import FakeBroker
    from tradeflow import cli
    from tradeflow.services.registry import STRATEGIES

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: True)
    args = build_parser().parse_args(["live", "--no-ledger", "--scanner", "none"])
    cli._print_live_preflight(
        args, STRATEGIES["ma_crossover"].create_with_defaults(), FakeBroker(), ["AAA"], None, None
    )

    assert "DISABLED" in capsys.readouterr().out


def test_real_money_is_refused_on_an_environment_variable_alone(monkeypatch):
    """PAPER_TRADE defaults to true, which is right and is also why this exists: a
    default nobody set is indistinguishable from a decision somebody made, until it is
    wrong. A live run must be asserted twice, in two places."""
    from tradeflow import cli

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: False)
    args = build_parser().parse_args(["live"])

    with pytest.raises(SystemExit) as exit_info:
        cli._refuse_ambiguous_broker_mode(args)

    assert "real money" in str(exit_info.value)
    assert "--live-money" in str(exit_info.value)


def test_real_money_proceeds_when_said_on_the_command_line_too(monkeypatch, capsys):
    from tradeflow import cli

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: False)
    cli._refuse_ambiguous_broker_mode(build_parser().parse_args(["live", "--live-money"]))

    assert "LIVE MONEY confirmed" in capsys.readouterr().out


def test_paper_mode_never_asks_for_an_acknowledgement(monkeypatch):
    """The guard exists for real money. Making paper runs confirm anything would train
    people to pass the flag reflexively."""
    from tradeflow import cli

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: True)
    cli._refuse_ambiguous_broker_mode(build_parser().parse_args(["live"]))  # must not raise
