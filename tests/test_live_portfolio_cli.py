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


# --- risk limits typed on the command line ---------------------------------------
def _book(argv, declared=None):
    """Resolve the book limits a `live` invocation would actually trade under."""
    from tradeflow.cli import _apply_limit_overrides, parse_cli
    from tradeflow.services.registry import STRATEGIES

    args = parse_cli(argv)
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    if declared:
        strategy.config["position_limits"] = {**strategy.position_limits(), **declared}
    _apply_limit_overrides(args, strategy)
    return strategy.position_limits()


def test_a_typed_position_limit_reaches_the_book():
    """The bug: `--max-positions 8` only ever sized the --portfolio allocator.

    Without --portfolio it was parsed and discarded, so a run asked to hold 8 held
    whatever the strategy declared, and the preflight truthfully printed that number
    while the operator read the one they had typed.
    """
    assert _book(["live", "--max-positions", "8"], declared={"max_positions": 10})["max_positions"] == 8


def test_a_dollar_ceiling_and_a_gross_cap_reach_the_book():
    """There was no way at all to state either from the command line."""
    limits = _book(["live", "--max-position-size", "1200", "--max-gross-exposure", "0.9"])

    assert limits["max_position_size"] == 1200.0
    assert limits["max_gross_exposure"] == 0.9


def test_an_untyped_default_does_not_overrule_a_frozen_config():
    """The other direction, and the reason only typed flags may apply: --max-positions
    defaults to 5, so applying it unconditionally would silently shrink the book a
    frozen config pinned at 10 — the precise failure the capital freeze exists to stop.
    """
    assert _book(["live"], declared={"max_positions": 10})["max_positions"] == 10


def test_an_untyped_flag_leaves_an_undeclared_limit_undeclared():
    """Absent is not zero: no typed flag and nothing declared stays None, not 0."""
    assert _book(["live"])["max_gross_exposure"] is None


def test_max_weight_without_portfolio_is_refused_rather_than_ignored():
    """The bug: `--max-weight 0.15` sizes the allocator, so without --portfolio it was
    accepted and discarded. A flag that cannot reach anything stops the run.
    """
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    with pytest.raises(SystemExit) as exit_info:
        _refuse_inert_flags(parse_cli(["live", "--capital", "8000", "--max-weight", "0.15"]))

    message = str(exit_info.value)
    assert "--portfolio" in message  # the remedy that keeps the flag
    # The book equivalent, so the refusal answers the question it raises.
    assert "--max-position-size 1200" in message


def test_the_refusal_offers_a_value_that_pastes_back_into_a_shell():
    """A thousands separator would make the suggested command fail to parse."""
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    with pytest.raises(SystemExit) as exit_info:
        _refuse_inert_flags(parse_cli(["live", "--capital", "80000", "--max-weight", "0.15"]))

    assert "--max-position-size 12000" in str(exit_info.value)


def test_max_weight_is_allowed_when_the_allocator_it_sizes_is_running():
    """Both directions: a guard that also rejects the working case rejects everything."""
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    _refuse_inert_flags(parse_cli(["live", "--portfolio", "--max-weight", "0.15"]))


def test_a_benchmark_nothing_measures_against_is_refused():
    """`--benchmark` only selects the symbol beta sizing uses."""
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    with pytest.raises(SystemExit) as exit_info:
        _refuse_inert_flags(parse_cli(["live", "--benchmark", "SPY"]))

    assert "--beta-sizing" in str(exit_info.value)


def test_a_benchmark_is_allowed_when_beta_sizing_uses_it():
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    _refuse_inert_flags(parse_cli(["live", "--beta-sizing", "--benchmark", "QQQ"]))


def test_an_untyped_benchmark_default_is_not_treated_as_typed():
    """--benchmark carries a default of SPY; not typing it must not refuse the run."""
    from tradeflow.cli import _refuse_inert_flags, parse_cli

    _refuse_inert_flags(parse_cli(["live", "--capital", "8000"]))


def test_preflight_shows_gross_exposure_in_dollars_not_only_as_a_fraction(capsys, monkeypatch):
    """A bare 0.9 is the one limit an operator cannot sanity-check against the book."""
    from unittest import mock

    from tests.fakes import FakeBroker
    from tradeflow import cli
    from tradeflow.cli import _apply_limit_overrides, parse_cli
    from tradeflow.services.registry import STRATEGIES

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: True)
    args = parse_cli(["live", "--scanner", "none", "--capital", "8000", "--max-gross-exposure", "0.9"])
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    _apply_limit_overrides(args, strategy)
    ledger = mock.Mock()
    ledger.path = "/tmp/l.jsonl"

    cli._print_live_preflight(args, strategy, FakeBroker(buying_power=100_000.0), ["AAA"], 8_000.0, ledger)

    assert "$7,200.00" in capsys.readouterr().out


def test_preflight_says_when_a_position_ceiling_cannot_bind(capsys, monkeypatch):
    """A $100k per-position cap on an $8k book reads as a limit somebody chose."""
    printed = _preflight_output(capsys, monkeypatch, capital=8_000.0)

    assert "not a binding limit" in printed


def test_typing_the_cardinality_makes_the_two_caps_agree_by_construction():
    """--max-positions now sets the book limit as well as the allocator's, so the pair
    the refusal exists to catch cannot be produced by typing the flag. The refusal
    still stands for an untyped default meeting a smaller declared limit."""
    from tradeflow.cli import _apply_limit_overrides, _refuse_contradictory_portfolio_cardinality, parse_cli
    from tradeflow.services.registry import STRATEGIES

    args = parse_cli(["live", "--portfolio", "--max-positions", "8"])
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": 1}
    _apply_limit_overrides(args, strategy)

    _refuse_contradictory_portfolio_cardinality(strategy, args.max_positions)  # must not raise


def test_preflight_states_the_feed_including_that_the_defaults_disagree(capsys, monkeypatch):
    """The mismatch that produced 0-of-61 warm-up bars is invisible until it bites, so
    the preflight says which feed each half will use before anything connects."""
    from unittest import mock

    from tests.fakes import FakeBroker
    from tradeflow import cli
    from tradeflow.cli import parse_cli
    from tradeflow.services.registry import STRATEGIES

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: True)
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    ledger = mock.Mock()
    ledger.path = "/tmp/l.jsonl"

    def show(argv):
        cli._print_live_preflight(
            parse_cli(argv),
            STRATEGIES["ma_crossover"].create_with_defaults(),
            FakeBroker(buying_power=100_000.0),
            ["AAA"],
            8_000.0,
            ledger,
        )
        return capsys.readouterr().out

    assert "iex" in show(["live", "--scanner", "none", "--feed", "iex"])
    # Unpinned must say so rather than printing nothing — the default is the risky case.
    unpinned = show(["live", "--scanner", "none"])
    assert "SDK default" in unpinned and "IEX for the stream" in unpinned


def test_preflight_shows_the_two_limits_that_were_enforced_but_never_printed(capsys, monkeypatch):
    """max_total_risk and min_notional gate every entry and had no preflight line, so a
    run could be throttled by a limit the operator had no way to see."""
    from unittest import mock

    from tests.fakes import FakeBroker
    from tradeflow import cli
    from tradeflow.cli import _apply_limit_overrides, parse_cli
    from tradeflow.services.registry import STRATEGIES

    monkeypatch.setattr("tradeflow.settings.paper_trade_mode", lambda: True)
    args = parse_cli(["live", "--scanner", "none", "--max-total-risk", "0.05", "--min-notional", "50"])
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    _apply_limit_overrides(args, strategy)
    ledger = mock.Mock()
    ledger.path = "/tmp/l.jsonl"

    cli._print_live_preflight(args, strategy, FakeBroker(), ["AAA"], 8_000.0, ledger)

    printed = capsys.readouterr().out
    assert "max total risk" in printed and "$400.00" in printed  # 5% of $8,000
    assert "min notional" in printed and "$50.00" in printed
