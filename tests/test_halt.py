"""The kill switch.

Two properties carry the weight. A halt must **block entries and never exits** — a
switch that trapped the book would be one nobody dares pull, and it would deadlock a
flatten against its own gate. And **absent is not halted**: a missing or corrupt state
file means no halt was recorded, because the reverse default turns an unrelated disk
problem into a silent trading freeze.
"""

import json

import pytest

from tests.fakes import FailingBroker, FakeBroker
from tradeflow.brokers.base import Position
from tradeflow.brokers.errors import BrokerUnavailableError
from tradeflow.demo.strategies import DemoTrendStrategy
from tradeflow.execution.flatten import flatten
from tradeflow.execution.halt import ALL, HaltState
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.strategies import signals


@pytest.fixture
def halts(tmp_path):
    return HaltState(tmp_path / "halts.json")


def _position(symbol="AAA"):
    return Position(
        symbol=symbol,
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pl=0.0,
    )


# --- state ------------------------------------------------------------------
def test_a_halt_is_not_in_force_until_it_is_set(halts):
    assert halts.is_halted() is False
    assert halts.active() is None


def test_a_halt_survives_a_new_reader(tmp_path):
    """Durability is the whole point: a restarted engine must still see it."""
    HaltState(tmp_path / "halts.json").set("bad data", actor="cli")
    assert HaltState(tmp_path / "halts.json").is_halted() is True


def test_a_halt_records_who_and_why(halts):
    halt = halts.set("feed looked wrong", actor="andy")
    assert halt.reason == "feed looked wrong"
    assert halt.actor == "andy"
    assert halt.set_at


def test_a_global_halt_covers_every_scope(halts):
    halts.set("everything off", actor="cli", scope=ALL)
    assert halts.is_halted("DemoTrendStrategy") is True


def test_a_scoped_halt_leaves_other_strategies_alone(halts):
    halts.set("just this one", actor="cli", scope="DemoTrendStrategy")
    assert halts.is_halted("DemoTrendStrategy") is True
    # Any other scope. A literal rather than a second shipped class: the engine now
    # ships exactly one strategy, and the scoping rule is about the name, not the class.
    assert halts.is_halted("SomeOtherStrategy") is False


def test_lifting_a_halt_reports_whether_one_was_in_force(halts):
    assert halts.clear() is False  # nothing to lift
    halts.set("stop", actor="cli")
    assert halts.clear() is True
    assert halts.is_halted() is False


def test_a_corrupt_state_file_means_no_halt_not_a_permanent_freeze(tmp_path):
    """The reverse default would turn an unrelated disk problem into a trading
    freeze nobody chose and nobody can explain."""
    path = tmp_path / "halts.json"
    path.write_text("{not json at all")
    assert HaltState(path).is_halted() is False


def test_a_malformed_record_is_skipped_without_losing_the_others(tmp_path):
    path = tmp_path / "halts.json"
    path.write_text(json.dumps({"all": {"no_reason_field": True}, "Strat": {"reason": "kept"}}))
    state = HaltState(path)
    assert state.is_halted("Strat") is True
    assert state.active("Strat").reason == "kept"


# --- what the trader does with it -------------------------------------------
def _trader(broker, halts):
    return LiveTrader(
        broker,
        DemoTrendStrategy.create_with_defaults(),
        respect_market_hours=False,
        halt_state=halts,
    )


def test_a_halt_refuses_a_new_entry(halts):
    broker = FakeBroker()
    halts.set("stop", actor="cli")

    _trader(broker, halts).handle_signal("AAA", signals.BUY, 100.0)

    assert broker.orders == []


def test_a_halt_never_refuses_an_exit(halts):
    """A switch that trapped the book is one nobody dares pull — and it would
    deadlock a flatten against its own gate."""
    broker = FakeBroker(positions=[_position()])
    halts.set("stop", actor="cli")
    trader = _trader(broker, halts)
    trader.sync_strategy_book()

    trader.handle_signal("AAA", signals.CLOSE_BUY, 100.0)

    assert broker.closed == ["AAA"]


def test_lifting_the_halt_allows_entries_again(halts):
    """The other direction: a halt must be reversible, or it is just an outage."""
    broker = FakeBroker()
    halts.set("stop", actor="cli")
    trader = _trader(broker, halts)
    trader.handle_signal("AAA", signals.BUY, 100.0)
    assert broker.orders == []

    halts.clear()
    trader.handle_signal("AAA", signals.BUY, 100.0)
    assert len(broker.orders) == 1


def test_a_strategy_scoped_halt_stops_only_that_strategy(halts):
    broker = FakeBroker()
    halts.set("this one misbehaves", actor="cli", scope="DemoTrendStrategy")

    _trader(broker, halts).handle_signal("AAA", signals.BUY, 100.0)

    assert broker.orders == []


# --- flatten ----------------------------------------------------------------
def test_flatten_halts_cancels_and_closes(halts):
    broker = FakeBroker(positions=[_position()])

    report = flatten(broker, reason="drill", actor="cli", halt_state=halts)

    assert report.complete
    assert halts.is_halted() is True
    assert broker.positions == {}


def test_flatten_halts_before_it_closes_anything(halts):
    """A running engine re-enters on the next bar; cancelling and closing while it
    still believes it may trade is a race the engine can win."""
    observed = {}

    class Watching(FakeBroker):
        def close_all_positions(self, cancel_orders=True):
            observed["halted_first"] = halts.is_halted()
            return super().close_all_positions(cancel_orders)

    flatten(Watching(positions=[_position()]), reason="drill", halt_state=halts)

    assert observed["halted_first"] is True


def test_flatten_still_closes_positions_when_cancelling_orders_fails(halts):
    """A partial flatten is bad; stopping halfway and leaving positions open is
    worse.

    Note what `complete` now means here. The cancel *call* failed, but the confirming
    read saw no positions and no resting orders — so the account is verifiably in the
    terminal state and this reports it as such, with the failure still listed. That is
    the point of observing rather than trusting: the old logic called this incomplete
    on the strength of a call that failed over an order book which turned out to be
    empty anyway.
    """
    broker = FailingBroker(positions=[_position()])
    broker.failures["cancel_all_orders"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is True
    assert report.orders_cancelled is False
    assert report.failures  # the failed call is still reported
    assert report.open_orders == 0  # and the book was verified clear regardless
    assert report.complete


def test_an_incomplete_flatten_says_so_rather_than_reporting_success(halts):
    broker = FailingBroker(positions=[_position()])
    broker.failures["close_all_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert not report.complete
    assert "NOT FLAT" in report.summary()
    assert "was not accepted" in report.summary()
    assert halts.is_halted() is True  # the halt still stands


# --- submitted is not observed -----------------------------------------------------
class QueueingBroker(FakeBroker):
    """A broker that accepts the close and keeps the positions.

    Exactly what a real venue does outside market hours: `close_all_positions` returns
    successfully, the orders queue, and the account still holds everything until the
    open. Observed in practice — a flatten reported every position closed and the
    account still held its whole book minutes later.
    """

    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        return True  # accepted, and nothing is closed


def test_a_queued_close_is_not_reported_as_a_closed_position(halts):
    """The defect this exists to stop: `positions_closed` was set when the *request*
    returned, so a flatten whose orders merely queued printed a complete report and a
    reassuring next step, while the book was untouched."""
    from tradeflow.execution.flatten import PENDING

    broker = QueueingBroker(positions=[_position()])

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is True  # the request was accepted
    assert report.observed == PENDING  # and nothing was observed closed
    assert not report.complete
    assert report.remaining == [_position().symbol]
    assert report.checked_at

    text = report.summary()
    assert "NOT FLAT" in text
    assert "close orders submitted    : yes" in text
    assert "positions observed closed : pending" in text
    # It must not claim the terminal state anywhere.
    assert "Flat, and confirmed" not in text


def test_a_confirmed_flatten_says_flat_and_names_when_it_looked(halts):
    """Both directions: the guard must still recognise the case it exists to permit,
    and the confirmation has to carry the instant it was taken — a flat report with no
    timestamp is a claim with no evidence behind it."""
    from tradeflow.execution.flatten import CLOSED

    report = flatten(FakeBroker(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.observed == CLOSED
    assert report.complete
    assert report.remaining == []
    assert report.checked_at
    text = report.summary()
    assert "positions observed closed : yes" in text
    assert "Flat, and confirmed by a broker read" in text


def test_an_unreadable_account_is_never_reported_as_flat(halts):
    """Absent is not zero, in the place it would hurt most. A confirming read that
    failed must not render as a clean book."""
    from tradeflow.execution.flatten import UNKNOWN

    broker = FailingBroker(positions=[_position()])
    broker.failures["list_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.observed == UNKNOWN
    assert not report.complete
    assert "UNCONFIRMED" in report.summary()
    assert "positions observed closed : unknown" in report.summary()


def test_a_failed_close_is_distinguished_from_a_queued_one(halts):
    """`no` and `pending` are different operator situations: one means the venue never
    took the request, the other means it took it and has not filled it."""
    from tradeflow.execution.flatten import NOT_CLOSED

    broker = FailingBroker(positions=[_position()])
    broker.failures["close_all_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is False
    assert report.observed == NOT_CLOSED
    assert not report.complete


def test_a_flatten_that_could_not_record_its_halt_is_not_complete(halts, monkeypatch):
    """A mutation dropping `halted` from `complete` survived the whole suite: the
    halt-write failure path had no test at all. Without the halt a running engine
    re-enters on the next bar and the account refills behind you, so a flat book is not
    a finished flatten."""

    def _refuse(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(halts, "set", _refuse)

    report = flatten(FakeBroker(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.halted is False
    assert report.observed == "yes"  # the book really is empty
    assert not report.complete  # and it is still not finished
    assert "POSITIONS FLAT, BUT NOT FINISHED" in report.summary()
    assert "engine may re-enter" in report.summary()


def test_a_broker_that_returns_no_position_list_is_not_read_as_flat(halts):
    """`or []` turned "I don't know" into "nothing is open" — in the one function whose
    whole purpose is refusing to call an unobserved book flat."""
    from tradeflow.execution.flatten import UNKNOWN

    class Silent(FakeBroker):
        def list_positions(self):
            return None

    report = flatten(Silent(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.observed == UNKNOWN
    assert not report.complete


def test_any_failure_of_the_confirming_read_still_returns_a_report(halts):
    """The regression that mattered most. `_confirm` runs *after* the halt, the cancel
    and the closes, so a reporting step that raises destroys the record of the order
    path — the operator loses not just the confirmation but the fact that the halt was
    set and the closes were sent. Reproduced with `TimeoutError`, which is not a
    `BrokerError` and escaped the original narrow catch."""
    from tradeflow.execution.flatten import UNKNOWN

    class Rude(FakeBroker):
        def list_positions(self):
            raise TimeoutError("socket timeout")

    report = flatten(Rude(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.observed == UNKNOWN
    assert report.halted is True  # the facts that survive are still reported
    assert report.close_submitted is True
    assert "UNCONFIRMED" in report.summary()


def test_a_broker_refusing_the_close_is_not_recorded_as_having_accepted_it(halts):
    """`close_all_positions()` returns a bool and it was discarded, so a broker saying
    "I did not take this" was recorded as submitted — the same lie this change exists to
    remove, one line above the fix."""
    from tradeflow.execution.flatten import NOT_CLOSED

    class Refusing(FakeBroker):
        def close_all_positions(self, cancel_orders: bool = True) -> bool:
            return False

    report = flatten(Refusing(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.close_submitted is False
    assert report.observed == NOT_CLOSED
    assert not report.complete
    assert any("did not accept" in f for f in report.failures)


def test_the_confirmation_timestamp_is_a_real_instant(halts):
    """A constant timestamp passed the earlier truthiness check. The claim this report
    makes is "flat *as of* this moment", so the moment has to be one."""
    from datetime import datetime, timezone

    before = datetime.now(timezone.utc)
    report = flatten(FakeBroker(positions=[_position()]), reason="drill", halt_state=halts)
    after = datetime.now(timezone.utc)

    assert before <= datetime.fromisoformat(report.checked_at) <= after


def test_resting_orders_are_observed_not_assumed(halts):
    """The cancel is a submitted fact too. A resting order the cancel missed can refill
    the book after the instant the read was taken, so a verifiably empty account with
    orders still working is not a terminal state."""
    from tradeflow.brokers.base import OrderResult

    class StillResting(FakeBroker):
        def list_open_orders(self, symbol=None):
            return [OrderResult(id="o1", symbol="AAA", side="buy", qty=1, status="new")]

    report = flatten(StillResting(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.observed == "yes"  # positions genuinely gone
    assert report.open_orders == 1
    assert not report.complete  # but the book can still refill
    assert "still resting" in report.summary()


# --- the surface, which had no test at all ------------------------------------------
def _flatten_args(**overrides):
    from tradeflow.cli import build_parser

    args = build_parser().parse_args(["flatten", "--confirm", "--reason", "drill"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def isolated_halt_state(tmp_path, monkeypatch):
    """Point `cmd_flatten`'s *default* halt state at this test's own directory.

    The command builds `HaltState()` itself, so it writes to the session-wide
    `TRADEFLOW_HOME` that conftest sets — not to the `halts` fixture. Without this the
    halt these tests set stands for the rest of the session and every later test that
    tries to enter a position is refused, which is exactly what happened: a cascade of
    unrelated failures in `test_live_trader` and `test_min_notional_parity`, none of
    them reproducible in isolation. Twenty-seven when it first happened; twenty-four
    when the leak was staged again on 2026-09-11, because the suite has changed since.
    The count is the symptom, not the fact — the fact is that none of them was about
    the code that failed.

    Still needed with the leak guard in place, and the two are complementary: this
    fixture is the prevention, the guard is the detection. Without it these tests error
    at teardown instead of passing.
    """
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    return tmp_path


def test_the_command_exits_non_zero_when_the_book_is_not_observed_flat(
    isolated_halt_state, monkeypatch, capsys
):
    """`cmd_flatten` had no test at all: the re-check block, the `--json` branch and the
    non-zero exit could each be deleted without failing anything. This is the contract a
    script actually keys on."""
    from tradeflow import cli

    class Queueing(FakeBroker):
        def close_all_positions(self, cancel_orders: bool = True) -> bool:
            return True  # accepted, nothing closed

    monkeypatch.setattr(
        cli, "build_data_and_broker", lambda *a, **k: (Queueing(positions=[_position()]), None)
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_flatten(_flatten_args())

    assert exc.value.code == 1
    printed = capsys.readouterr().out
    assert "NOT FLAT" in printed
    # And it must point somewhere that re-reads the broker, not somewhere that compares
    # the ledger — `reconcile` answers "does my ledger match", not "am I flat", and in
    # this exact state it reports no divergence while the whole book is still open.
    assert "flatten --confirm" in printed
    assert "reconcile" not in printed


def test_the_json_report_carries_the_observed_fields(isolated_halt_state, monkeypatch, capsys):
    """A machine reader has to be able to tell submitted from observed too."""
    import json

    from tradeflow import cli

    monkeypatch.setattr(
        cli, "build_data_and_broker", lambda *a, **k: (FakeBroker(positions=[_position()]), None)
    )

    cli.cmd_flatten(_flatten_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["observed"] == "yes"
    assert payload["close_submitted"] is True
    assert payload["open_orders"] == 0
    assert payload["checked_at"]
    assert payload["complete"] is True


def test_a_halt_file_that_is_valid_json_but_not_an_object_reads_as_no_halt(tmp_path, caplog):
    """`absent is not halted` has to hold for every shape a hand edit can produce, not
    only for a file that fails to parse. `null`, a list and a bare number are all valid
    JSON, and each one used to raise AttributeError out of `_read` — reaching the trade
    clock through the per-signal entry check, so a mistyped file during an incident
    crashed the decision path instead of answering it."""
    from tradeflow.execution.halt import HaltState

    for document in ("null", "5", "[]", '[{"scope": "all"}]', '"halted"'):
        path = tmp_path / "halts.json"
        path.write_text(document)
        state = HaltState(path)

        assert state.is_halted() is False, document
        assert state.active("anything") is None, document
        assert state.list() == [], document

    # Loud, because the switch is not working: a silent default here would be the
    # unreadable-file case all over again.
    assert "treating as NO halt" in caplog.text
