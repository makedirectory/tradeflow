"""Execution-layer tests using an in-memory FakeBroker."""

from datetime import datetime

import pytest

from tests.fakes import FakeBroker
from tradeflow.brokers.base import AccountSnapshot, OrderSide, Position
from tradeflow.execution import decision as decisions
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.execution.order_id import client_order_id
from tradeflow.execution.sizing import BetaSizer, PortfolioWeightSizer, RiskBasedSizer
from tradeflow.strategies import signals
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy


def _trader(broker, sizer=None):
    return LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults(), sizer=sizer)


def _open_position(symbol="AAA"):
    return Position(
        symbol=symbol,
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pl=0.0,
    )


# --- position sizers --------------------------------------------------------
def _account(equity=100_000.0):
    return AccountSnapshot(cash=equity, equity=equity, buying_power=equity, portfolio_value=equity)


def test_portfolio_weight_sizer():
    sizer = PortfolioWeightSizer({"AAA": 0.25})
    assert sizer.size("AAA", 100.0, _account()) == 250.0  # 0.25 * 100k / 100
    assert sizer.size("BBB", 100.0, _account()) == 0.0  # unfunded symbol


def test_risk_based_sizer_delegates_to_strategy():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    sizer = RiskBasedSizer(strategy)
    assert sizer.size("AAA", 100.0, _account()) == strategy.calculate_position_size(100_000.0, 100.0)


def test_beta_sizer_scales_inversely_with_beta():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    # beta 1.0 is the neutral baseline; beta 2.0 should roughly halve the size.
    base = BetaSizer(strategy, {"AAA": 1.0}).size("AAA", 100.0, account)
    high_beta = BetaSizer(strategy, {"AAA": 2.0}).size("AAA", 100.0, account)
    assert high_beta == pytest.approx(base / 2.0)


def test_beta_sizer_uses_default_for_unknown_symbol():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    unknown = BetaSizer(strategy, {}, default_beta=1.0).size("ZZZ", 100.0, account)
    neutral = BetaSizer(strategy, {"ZZZ": 1.0}).size("ZZZ", 100.0, account)
    assert unknown == neutral


def test_beta_sizer_clamps_extreme_beta():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    # Beyond max_abs_beta, sizes should match the clamp (not keep shrinking).
    at_cap = BetaSizer(strategy, {"AAA": 4.0}, max_abs_beta=4.0).size("AAA", 100.0, account)
    beyond = BetaSizer(strategy, {"AAA": 99.0}, max_abs_beta=4.0).size("AAA", 100.0, account)
    assert beyond == pytest.approx(at_cap)


def test_live_trader_sizes_from_portfolio_weights():
    broker = FakeBroker(buying_power=100_000)  # equity == buying_power in the fake
    _trader(broker, sizer=PortfolioWeightSizer({"AAA": 0.5})).handle_signal("AAA", signals.BUY, 100.0)
    assert broker.orders[0]["qty"] == 500  # floor(0.5 * 100k / 100)


def test_entry_submits_bracket_order():
    broker = FakeBroker(buying_power=100_000)
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)

    assert len(broker.orders) == 1
    order = broker.orders[0]
    assert order["type"] == "bracket" and order["symbol"] == "AAA"
    assert order["stop_loss"] < 100.0 < order["take_profit"]
    assert order["qty"] > 0


def test_entry_skipped_when_position_exists():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_exit_closes_matching_position():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.CLOSE_BUY, price=100.0)
    assert broker.closed == ["AAA"]


def test_exit_ignored_when_side_mismatches():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.CLOSE_SELL, price=100.0)  # closing a short
    assert broker.closed == []


def test_hold_is_noop():
    broker = FakeBroker()
    _trader(broker).handle_signal("AAA", signals.HOLD, price=100.0)
    assert broker.orders == [] and broker.closed == []


def test_insufficient_buying_power_blocks_entry():
    broker = FakeBroker(buying_power=10.0)  # can't afford one $100 share
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_closed_market_blocks_orders():
    broker = FakeBroker(buying_power=100_000, market_open=False)
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_market_hours_can_be_disabled():
    broker = FakeBroker(buying_power=100_000, market_open=False)
    LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults(), respect_market_hours=False).handle_signal(
        "AAA", signals.BUY, price=100.0
    )
    assert len(broker.orders) == 1


# --- order management -------------------------------------------------------
def test_pending_order_blocks_duplicate_entry():
    broker = FakeBroker(buying_power=100_000)
    trader = _trader(broker)
    trader.handle_signal("AAA", signals.BUY, 100.0)  # places a bracket order
    trader.handle_signal("AAA", signals.BUY, 100.0)  # order pending -> must not double-submit
    assert len(broker.orders) == 1


def test_exit_cancels_resting_orders_before_closing():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    # Simulate a resting bracket leg for the symbol.
    broker.submit_bracket_order("AAA", 5, OrderSide.SELL, 95, 110)
    assert broker.list_open_orders("AAA")  # precondition

    _trader(broker).handle_signal("AAA", signals.CLOSE_BUY, 100.0)
    assert broker.closed == ["AAA"]
    assert broker.list_open_orders("AAA") == []  # resting legs canceled


# --- order identity ---------------------------------------------------------
def _ts(minute=0):
    return datetime(2024, 1, 2, 10, minute)


def test_the_same_decision_always_yields_the_same_order_id():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    first = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    again = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    assert first == again


@pytest.mark.parametrize(
    "symbol,signal,ts",
    [("BBB", signals.BUY, _ts(1)), ("AAA", signals.SELL, _ts(1)), ("AAA", signals.BUY, _ts(2))],
)
def test_a_different_decision_yields_a_different_order_id(symbol, signal, ts):
    """Cover every axis: the same symbol on a later bar is a new order, not a replay."""
    strategy = VolumeSpikeStrategy.create_with_defaults()
    baseline = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    assert client_order_id(strategy, symbol, signal, ts) != baseline


def test_reconfiguring_a_strategy_changes_its_order_ids():
    """Two parameterizations can legitimately disagree about the same bar; one must
    not be deduplicated against the other."""
    base = VolumeSpikeStrategy.create_with_defaults()
    other = VolumeSpikeStrategy.create_with_defaults()
    other.config["risk_per_trade"] = base.config["risk_per_trade"] / 2
    assert client_order_id(base, "AAA", signals.BUY, _ts(1)) != client_order_id(
        other, "AAA", signals.BUY, _ts(1)
    )


def test_an_entry_carries_its_client_order_id_to_the_broker():
    broker = FakeBroker()
    _trader(broker).handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=_ts(1))
    assert broker.orders[0]["client_order_id"]


def test_a_replayed_bar_cannot_place_a_second_order():
    """The failure this exists for: a reconnect redelivers a bar, or the process
    restarts, and the check-then-act guard against open orders no longer remembers
    anything. The broker refuses the duplicate id instead."""
    broker = FakeBroker()
    strategy = VolumeSpikeStrategy.create_with_defaults()
    trader = LiveTrader(broker, strategy)

    first = trader.handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=_ts(1))
    # Wipe the local shortcut and the book, exactly as a restart would.
    broker.open_orders_list.clear()
    strategy.positions.clear()
    second = trader.handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=_ts(1))

    assert first.allowed
    assert not second.allowed
    assert "already placed" in second.reason
    assert len(broker.orders) == 1
    assert broker.rejected_duplicates  # refused by identity, not by looking first


def test_a_genuinely_new_bar_still_places_an_order_after_a_restart():
    """The other direction: identity must not become a reason to never trade again."""
    broker = FakeBroker()
    strategy = VolumeSpikeStrategy.create_with_defaults()
    trader = LiveTrader(broker, strategy)

    trader.handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=_ts(1))
    broker.open_orders_list.clear()
    strategy.positions.clear()
    later = trader.handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=_ts(2))

    assert later.allowed
    assert len(broker.orders) == 2


# --- decisions --------------------------------------------------------------
def test_a_decision_names_the_guard_that_stopped_it():
    """ "Nothing happened" used to be one answer for a dozen causes."""
    broker = FakeBroker(positions=[_open_position()])
    decision = _trader(broker).handle_signal("AAA", signals.BUY, 100.0)

    assert not decision.allowed
    assert decision.reason == "position already open"
    assert decisions.EXISTING_POSITION in decision.guards_consulted


def test_a_decision_lists_the_guards_it_actually_consulted_not_just_the_one_that_fired():
    """A veto list naming only the guard that tripped cannot distinguish a guard that
    passed from one that never ran — which is how a check silently stops applying."""
    broker = FakeBroker(buying_power=1.0)  # too poor to size anything
    decision = _trader(broker).handle_signal("AAA", signals.BUY, 100.0)

    assert not decision.allowed
    # Everything before sizing was consulted and passed.
    assert decisions.EXISTING_POSITION in decision.guards_consulted
    assert decisions.PENDING_ORDER in decision.guards_consulted
    assert decisions.ACCOUNT in decision.guards_consulted
    assert decisions.SIZING in decision.guards_consulted
    # The broker was never reached, so it must not appear.
    assert decisions.BROKER not in decision.guards_consulted


def test_an_allowed_decision_carries_the_order():
    broker = FakeBroker()
    decision = _trader(broker).handle_signal("AAA", signals.BUY, 100.0)

    assert decision.allowed
    assert decision.order is not None
    assert bool(decision) is True


def test_a_market_hours_veto_is_recorded_as_such():
    broker = FakeBroker(market_open=False)
    decision = LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults()).handle_signal(
        "AAA", signals.BUY, 100.0
    )

    assert not decision.allowed
    assert decision.reason == "market closed"
    assert decisions.MARKET_HOURS in decision.guards_consulted


def test_a_skipped_market_hours_check_does_not_claim_to_have_run():
    """`respect_market_hours=False` means the guard did not run, which is not the
    same as running and passing."""
    trader = LiveTrader(FakeBroker(), VolumeSpikeStrategy.create_with_defaults(), respect_market_hours=False)
    decision = trader.handle_signal("AAA", signals.BUY, 100.0)
    assert decisions.MARKET_HOURS not in decision.guards_consulted


def test_a_declined_decision_is_written_to_the_ledger(tmp_path):
    """The case that leaves no other trace is exactly the one worth recording."""
    import json

    from tradeflow.execution.ledger import PositionLedger

    ledger = PositionLedger(tmp_path / "ledger.jsonl")
    broker = FakeBroker(positions=[_open_position()])
    decision = _trader(broker).handle_signal("AAA", signals.BUY, 100.0)
    ledger.record_decision(decision)

    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "decision"
    assert records[0]["allowed"] is False
    assert records[0]["guards_consulted"]


def test_a_recorded_decision_carries_no_position_meaning(tmp_path):
    """It is an explanation, not a fill — it must not move the expected book."""
    from tradeflow.execution.ledger import PositionLedger

    ledger = PositionLedger(tmp_path / "ledger.jsonl")
    ledger.record_fill("AAA", "buy", 10.0)
    before = ledger.expected_positions()

    ledger.record_decision(_trader(FakeBroker()).handle_signal("BBB", signals.BUY, 100.0))

    assert ledger.expected_positions() == before
