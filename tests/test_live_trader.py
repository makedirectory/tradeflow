"""Execution-layer tests using an in-memory FakeBroker."""

from datetime import datetime

import pytest

from tests.fakes import FakeBroker
from tradeflow.brokers.base import AccountSnapshot, OrderSide, Position
from tradeflow.demo.strategies import DemoTrendStrategy
from tradeflow.execution import decision as decisions
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.execution.order_id import client_order_id
from tradeflow.execution.sizing import BetaSizer, PortfolioWeightSizer, RiskBasedSizer
from tradeflow.strategies import signals


def _trader(broker, sizer=None):
    return LiveTrader(broker, DemoTrendStrategy.create_with_defaults(), sizer=sizer)


def _open_position(symbol="AAA", qty=10.0):
    return Position(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=qty * 100.0,
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
    strategy = DemoTrendStrategy.create_with_defaults()
    sizer = RiskBasedSizer(strategy)
    assert sizer.size("AAA", 100.0, _account()) == strategy.calculate_position_size(100_000.0, 100.0)


def test_beta_sizer_scales_inversely_with_beta():
    strategy = DemoTrendStrategy.create_with_defaults()
    account = _account()
    # beta 1.0 is the neutral baseline; beta 2.0 should roughly halve the size.
    base = BetaSizer(strategy, {"AAA": 1.0}).size("AAA", 100.0, account)
    high_beta = BetaSizer(strategy, {"AAA": 2.0}).size("AAA", 100.0, account)
    assert high_beta == pytest.approx(base / 2.0)


def test_beta_sizer_uses_default_for_unknown_symbol():
    strategy = DemoTrendStrategy.create_with_defaults()
    account = _account()
    unknown = BetaSizer(strategy, {}, default_beta=1.0).size("ZZZ", 100.0, account)
    neutral = BetaSizer(strategy, {"ZZZ": 1.0}).size("ZZZ", 100.0, account)
    assert unknown == neutral


def test_beta_sizer_clamps_extreme_beta():
    strategy = DemoTrendStrategy.create_with_defaults()
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
    LiveTrader(broker, DemoTrendStrategy.create_with_defaults(), respect_market_hours=False).handle_signal(
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
    strategy = DemoTrendStrategy.create_with_defaults()
    first = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    again = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    assert first == again


@pytest.mark.parametrize(
    "symbol,signal,ts",
    [("BBB", signals.BUY, _ts(1)), ("AAA", signals.SELL, _ts(1)), ("AAA", signals.BUY, _ts(2))],
)
def test_a_different_decision_yields_a_different_order_id(symbol, signal, ts):
    """Cover every axis: the same symbol on a later bar is a new order, not a replay."""
    strategy = DemoTrendStrategy.create_with_defaults()
    baseline = client_order_id(strategy, "AAA", signals.BUY, _ts(1))
    assert client_order_id(strategy, symbol, signal, ts) != baseline


def test_reconfiguring_a_strategy_changes_its_order_ids():
    """Two parameterizations can legitimately disagree about the same bar; one must
    not be deduplicated against the other."""
    base = DemoTrendStrategy.create_with_defaults()
    other = DemoTrendStrategy.create_with_defaults()
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
    strategy = DemoTrendStrategy.create_with_defaults()
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
    strategy = DemoTrendStrategy.create_with_defaults()
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
    decision = LiveTrader(broker, DemoTrendStrategy.create_with_defaults()).handle_signal(
        "AAA", signals.BUY, 100.0
    )

    assert not decision.allowed
    assert decision.reason == "market closed"
    assert decisions.MARKET_HOURS in decision.guards_consulted


def test_a_skipped_market_hours_check_does_not_claim_to_have_run():
    """`respect_market_hours=False` means the guard did not run, which is not the
    same as running and passing."""
    trader = LiveTrader(FakeBroker(), DemoTrendStrategy.create_with_defaults(), respect_market_hours=False)
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


# --- portfolio limits on the trade clock ------------------------------------
#: $20k of notional per position at the default 1% stop, so one position risks $200
#: and deploys $20k - numbers big enough for a book-level limit to bite.
def _trader_with_limits(broker, **limits):
    strategy = DemoTrendStrategy.create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_position_size": 20_000.0,
        **limits,
    }
    trader = LiveTrader(broker, strategy)
    trader.sync_strategy_book()  # the book the limits are counted from
    return trader


class _CountingBroker(FakeBroker):
    """Counts book reads, so the loop fence can be asserted rather than assumed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.list_calls = 0

    def list_positions(self):
        self.list_calls += 1
        return super().list_positions()


def _held(*symbols, qty=200.0):
    return [_open_position(sym, qty=qty) for sym in symbols]


def test_max_positions_is_enforced_against_the_whole_book():
    """It is a portfolio limit live too, not merely a per-symbol one."""
    broker = FakeBroker(positions=_held("BBB"))
    trader = _trader(broker)  # the strategy's own default is one position
    trader.sync_strategy_book()
    decision = trader.handle_signal("AAA", signals.BUY, 100.0)

    assert not decision.allowed
    assert "book is full" in decision.reason
    assert decisions.POSITION_LIMITS in decision.guards_consulted
    assert broker.orders == []


def test_the_risk_budget_is_shared_across_the_book_not_granted_per_position():
    """The defect this guard exists for.

    Sizing clamps one position against the *whole* max_total_risk budget, so every
    entry can be individually within budget while the book is far past it. Five open
    positions here have spent the entire $1,000 budget; the sixth is still a legal
    size on its own, and must still be refused.
    """
    broker = FakeBroker(positions=_held("BBB", "CCC", "DDD", "EEE", "FFF"))
    trader = _trader_with_limits(broker, max_positions=10, max_total_risk=0.01)

    # 200 shares is what sizing asks for, and it is inside the per-position clamp:
    # the budget alone would allow 1,000 shares.
    assert trader._strategy.calculate_position_size(100_000.0, 100.0) == 200.0

    decision = trader.handle_signal("AAA", signals.BUY, 100.0)
    assert not decision.allowed
    assert "risk budget exhausted" in decision.reason
    assert broker.orders == []


def test_gross_exposure_is_capped_live():
    """$40k already deployed, a $20k entry, a $45k cap."""
    broker = FakeBroker(positions=_held("BBB", "CCC"))
    decision = _trader_with_limits(
        broker, max_positions=10, max_total_risk=1.0, max_gross_exposure=0.45
    ).handle_signal("AAA", signals.BUY, 100.0)

    assert not decision.allowed
    assert "gross exposure capped" in decision.reason
    assert broker.orders == []


def test_an_entry_that_fits_the_limits_still_goes_through():
    """The guard must reject the breach, not the trade - it is consulted and passes."""
    broker = FakeBroker(positions=_held("BBB", "CCC"))
    decision = _trader_with_limits(
        broker, max_positions=10, max_total_risk=1.0, max_gross_exposure=0.70
    ).handle_signal("AAA", signals.BUY, 100.0)

    assert decision.allowed
    assert decisions.POSITION_LIMITS in decision.guards_consulted
    assert len(broker.orders) == 1


def test_an_entry_this_trader_opened_counts_against_the_next_one():
    """Within a bar, positions compete - the same thing the engine does when it
    admits candidates in conviction order against one shrinking book."""
    broker = FakeBroker(positions=_held("BBB"))
    trader = _trader_with_limits(broker, max_positions=3, max_total_risk=1.0, max_gross_exposure=0.55)

    first = trader.handle_signal("AAA", signals.BUY, 100.0)  # $20k on top of $20k
    second = trader.handle_signal("CCC", signals.BUY, 100.0)  # would make $60k

    assert first.allowed
    assert not second.allowed
    assert "gross exposure capped" in second.reason


def test_checking_the_limits_adds_no_broker_call_to_the_bar_loop():
    """Several symbols can signal an entry on one bar; a book read per entry would
    put a broker round trip on each of them."""
    broker = _CountingBroker(positions=_held("BBB"))
    trader = _trader_with_limits(broker, max_positions=10, max_total_risk=1.0)
    assert broker.list_calls == 1  # the start-up sync, and only that

    for symbol in ("AAA", "CCC", "DDD"):
        trader.handle_signal(symbol, signals.BUY, 100.0)
    assert broker.list_calls == 1


# --- capital: what this run may deploy -----------------------------------------
def _sized_at(capital, account_balance=100_000.0):
    broker = FakeBroker(buying_power=account_balance)
    strategy = DemoTrendStrategy.create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_positions": 5,
        # High enough that the notional cap never binds: this test is about which
        # capital figure the risk calculation reads, and a cap clipping either side
        # would hide exactly the difference it exists to detect.
        "max_position_size": 1_000_000.0,
    }
    LiveTrader(broker, strategy, capital=capital).handle_signal("AAA", signals.BUY, price=100.0)
    return broker.orders[0]["qty"] if broker.orders else 0


def test_sizing_uses_the_configured_capital_not_the_account_balance():
    """A paper account arrives with whatever equity the venue handed out.

    Sizing against that trades a different book from the one validated - which does not
    merely flatter the result, it invalidates the execution telemetry the run exists to
    gather, because fills, slippage and rounding are all properties of a book at a size.
    """
    # Risk 2% of the configured capital over a 3% stop, at $100: 15,000 * 0.02 / 3.
    assert _sized_at(15_000.0) == 100
    # The same arithmetic against the account balance instead - 100,000 * 0.02 / 3.
    assert _sized_at(None) == 666


def test_capital_caps_and_never_inflates():
    """A $8,000 config on a $3,000 account may deploy $3,000, never the number in the
    file. It is a ceiling on what may be used, not a claim about what exists."""
    assert _sized_at(8_000.0, account_balance=3_000.0) * 100 <= 3_000.0


def test_no_capital_leaves_an_unconfigured_run_exactly_as_it_was():
    assert _sized_at(None) == _sized_at(0)  # falsy capital is "the whole account"


def test_portfolio_limits_are_fractions_of_the_capital_not_the_account():
    """`max_total_risk` and `max_gross_exposure` are fractions *of equity*, and 5% of a
    paper account's balance is not 5% of the capital a config was validated at."""
    broker = FakeBroker(buying_power=100_000.0)
    strategy = DemoTrendStrategy.create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_positions": 5,
        "max_position_size": 50_000.0,
        "max_gross_exposure": 0.5,  # half of *what*?
    }
    trader = LiveTrader(broker, strategy, capital=8_000.0)

    trader.handle_signal("AAA", signals.BUY, price=100.0)
    filled = sum(order["qty"] * 100 for order in broker.orders)

    assert filled <= 8_000.0 * 0.5 + 1e-6  # half the capital, not half the account
