"""The one mode that can place real orders.

Dry run is safe because trading is a capability its broker does not have, and none of
that transfers here — this reaches a broker that really can trade, because broker fills,
slippage and fees cannot be observed any other way. Its safety is operational, so these
tests are about the operations: what the parser refuses to express, what the command
refuses to start, what the telemetry records, and what the research journal never sees.

The load-bearing one is `test_no_flag_here_can_move_a_cap`. The mode exists because
someone shrank caps by hand until fills happened; the guarantee is that this surface
cannot spell that, and the guarantee is enforced by the flags not existing rather than
by a check somebody has to remember.
"""

import json

import pytest

from tests.fakes import DictMarketData, FakeBroker
from tradeflow import cli
from tradeflow.marketdata.client import MarketDataClient

VALIDATED_CAPITAL = 200_000.0
VALIDATED_BOOK = {
    "max_positions": 8,
    "max_position_size": 10_000.0,
    "max_total_risk": 0.05,
    "max_gross_exposure": 0.80,
    "max_net_exposure": 0.30,
    "min_notional": 50.0,
}


def _config(tmp_path, capital=VALIDATED_CAPITAL, limits=VALIDATED_BOOK, name="validated.json"):
    path = tmp_path / name
    payload = {
        "strategy": "demo_trend",
        "params": {
            "fast_ema_period": 10,
            "slow_ema_period": 30,
            "risk_per_trade": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.06,
        },
        "scanner": "none",
        "symbols": ["AAA"],
    }
    if capital is not None:
        payload["capital"] = capital
    if limits is not None:
        payload["position_limits"] = limits
    path.write_text(json.dumps(payload))
    return path


def _frame(periods=60):
    import pandas as pd

    from tradeflow.utils.timeutils import NEW_YORK

    index = pd.date_range("2024-01-02", periods=periods, freq="D", tz=NEW_YORK)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
        index=index,
    )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """The command with a fake broker and the real data client over a fake provider."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PAPER_TRADE", "true")
    broker = FakeBroker(buying_power=100_000.0)
    client = MarketDataClient(DictMarketData({"AAA": _frame()}))
    monkeypatch.setattr(cli, "build_data_and_broker", lambda **kw: (broker, client))
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])
    return broker


def _run(argv):
    args = cli.build_parser().parse_args(argv)
    args.flags_given = set()
    cli.cmd_small_real(args)
    return args


# --- the semantics cannot be composed from ordinary flags ---------------------------
def test_no_flag_here_can_move_a_cap():
    """The whole reason this is a command and not `live --small-real`.

    The mode exists because caps were shrunk by hand until fills happened, biasing the
    sample toward low-priced names. Enforced by the flags not existing — argparse
    refuses them — rather than by a check inside the command, which could be forgotten
    on one branch or inverted in a refactor.

    Enumerated from the parser rather than from a list somebody remembered, so a cap
    flag added later fails here instead of quietly becoming available.
    """
    parser = cli.build_parser()
    small = parser._subparsers._group_actions[0].choices["small-real"]
    flags = {option for action in small._actions for option in action.option_strings}

    forbidden = {
        "--max-positions",
        "--max-position-size",
        "--max-gross-exposure",
        "--max-net-exposure",
        "--max-total-risk",
        "--min-notional",
        "--max-weight",
    }
    assert not (flags & forbidden), f"small-real can move a cap: {sorted(flags & forbidden)}"

    # And the same flags really do exist on `live`, so the assertion above is about this
    # parser rather than about the names having been renamed everywhere.
    live = parser._subparsers._group_actions[0].choices["live"]
    live_flags = {option for action in live._actions for option in action.option_strings}
    assert forbidden <= live_flags


def test_the_telemetry_cannot_be_switched_off():
    """A run whose whole purpose is to record what execution did has nothing left if it
    does not record, so there is no `--no-ledger` here even though `live` has one."""
    parser = cli.build_parser()
    small = parser._subparsers._group_actions[0].choices["small-real"]
    flags = {option for action in small._actions for option in action.option_strings}

    assert "--no-ledger" not in flags
    live = parser._subparsers._group_actions[0].choices["live"]
    assert "--no-ledger" in {o for a in live._actions for o in a.option_strings}


def test_a_config_is_required_by_the_parser():
    """Without the file there is no validated contract to scale, and scaling class
    defaults preserves proportions nobody validated."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["small-real", "--scale", "0.05"])


# --- what it refuses to start ------------------------------------------------------
def test_the_size_of_the_run_must_be_stated(wired, tmp_path):
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="needs the size of the run stated"):
        _run(["small-real", "--config", str(config)])


def test_a_config_capital_does_not_silently_become_the_typed_one(wired, tmp_path, capsys):
    """Found by running it. The config loader writes the file's capital into
    `args.capital`, so reading the namespace after the config is layered on made a
    `--scale` run indistinguishable from one that passed both — and this command refuses
    both. It failed as "two sources for one number" with only `--scale` given.
    """
    config = _config(tmp_path)

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    printed = capsys.readouterr().out
    assert "$10,000.00" in printed and "scale 0.05" in printed


def test_a_config_that_records_no_book_is_refused(wired, tmp_path):
    config = _config(tmp_path, limits=None)

    with pytest.raises(SystemExit, match="no position_limits"):
        _run(["small-real", "--config", str(config), "--capital", "10000"])


def test_a_config_with_no_capital_refuses_a_scale_and_names_the_way_out(wired, tmp_path):
    config = _config(tmp_path, capital=None)

    with pytest.raises(SystemExit, match="--capital with the amount"):
        _run(["small-real", "--config", str(config), "--scale", "0.05"])


def test_a_dollar_ceiling_that_cannot_be_restated_refuses_rather_than_going_inert(wired, tmp_path):
    """With no validated capital there is no ratio, so a $10,000 ceiling would sit above
    a $10,000 run and bind nothing — the contract silently missing a limit it was
    validated with, in the direction that lets positions get too big."""
    config = _config(tmp_path, capital=None)

    with pytest.raises(SystemExit, match="no ratio to restate"):
        _run(["small-real", "--config", str(config), "--capital", "10000"])


def test_a_book_with_no_dollar_ceiling_runs_without_a_ratio(wired, tmp_path, capsys):
    """Both directions: the refusal above must not reject the case it has no quarrel
    with. Fractions and counts do not need a ratio, so this contract is fully expressible."""
    config = _config(tmp_path, capital=None, limits={**VALIDATED_BOOK, "max_position_size": None})

    _run(["small-real", "--config", str(config), "--capital", "10000", "--preflight"])

    printed = capsys.readouterr().out
    assert "scale unknown: no validated capital" in printed
    assert "max_position_size" in printed and "unset" in printed


def test_an_account_too_small_for_the_scaled_contract_refuses(monkeypatch, tmp_path):
    """Sizing caps at whatever the account holds rather than failing, so this would
    otherwise trade a smaller book than the one whose proportions it claims to preserve
    — and the telemetry would look exactly like telemetry from the right contract."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PAPER_TRADE", "true")
    monkeypatch.setattr(
        cli,
        "build_data_and_broker",
        lambda **kw: (
            FakeBroker(buying_power=5_000.0),
            MarketDataClient(DictMarketData({"AAA": _frame()})),
        ),
    )
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="less than the"):
        _run(["small-real", "--config", str(config), "--capital", "10000", "--preflight"])


# --- the paper/live posture --------------------------------------------------------
def test_live_money_on_a_paper_environment_is_refused_not_ignored(wired, tmp_path, monkeypatch):
    """`live` ignores the flag here. For the one mode that can lose money, "you asked
    for real capital and quietly got paper" is not a state to enter."""
    monkeypatch.setenv("PAPER_TRADE", "true")
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="would go to the paper account"):
        _run(["small-real", "--config", str(config), "--scale", "0.05", "--live-money"])


def test_real_money_without_saying_so_on_the_command_line_is_refused(wired, tmp_path, monkeypatch):
    """A default nobody set is indistinguishable from a decision somebody made, right up
    until it is wrong."""
    monkeypatch.setenv("PAPER_TRADE", "false")
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="environment variable alone"):
        _run(["small-real", "--config", str(config), "--scale", "0.05"])


def test_the_posture_refusals_happen_before_a_broker_is_built(tmp_path, monkeypatch):
    """Both are the command line and the environment disagreeing, which is answerable
    without reaching a venue — and reaching one to say so would need credentials the
    operator has no reason to have working yet."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PAPER_TRADE", "false")
    monkeypatch.setattr(
        cli, "build_data_and_broker", lambda **kw: pytest.fail("a broker was built before the refusal")
    )
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="environment variable alone"):
        _run(["small-real", "--config", str(config), "--scale", "0.05"])


def test_real_money_needs_confirming_after_the_contract_has_been_shown(wired, tmp_path, monkeypatch, capsys):
    """The confirmation follows the preflight on purpose: agreeing to a contract you
    have not been shown is a formality, not a check."""
    monkeypatch.setenv("PAPER_TRADE", "false")
    config = _config(tmp_path)

    with pytest.raises(SystemExit, match="--confirm"):
        _run(["small-real", "--config", str(config), "--scale", "0.05", "--live-money"])

    printed = capsys.readouterr().out
    assert "SMALL-REAL PREFLIGHT" in printed
    assert "LIVE - REAL MONEY" in printed


def test_paper_needs_no_confirmation(wired, tmp_path, capsys):
    """Both directions. A gate on the run that cannot lose anything is friction that
    teaches the reflex, and the reflex is what makes the real gate stop working."""
    config = _config(tmp_path)

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    assert "PAPER" in capsys.readouterr().out


# --- the preflight a reader actually sees ------------------------------------------
def test_the_preflight_shows_each_limit_beside_the_validated_one(wired, tmp_path, capsys):
    """Asserted against the rendered text, not the payload. The claim this mode makes is
    that the proportions survived, and a column of scaled figures cannot be checked
    against a claim nobody printed — a report whose formatter dropped the whole new
    vocabulary has shipped here before, under a green suite that asserted only JSON.
    """
    config = _config(tmp_path)

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    printed = capsys.readouterr().out
    assert "SMALL-REAL PREFLIGHT — this run can place orders" in printed
    assert "Fractions scale. Counts stay counts." in printed
    # The validated value and the scaled one, on one line, in their own units.
    assert "$10,000.00       $500.00   dollar ceiling — scaled" in printed
    assert "0.8           0.8   fraction of capital — unchanged" in printed
    assert "8             8   count — unchanged" in printed
    assert "$50.00        $50.00   venue floor — not scaled" in printed
    assert "This can place orders. Nothing below is a rehearsal." in printed


def test_the_preflight_states_the_bias_the_scale_creates(wired, tmp_path, capsys):
    """Share granularity is the bias no amount of care avoids: a book with a few hundred
    dollars a position cannot buy one share of a four-figure stock. A reader must learn
    that here rather than conclude from a clean-looking sample that it did not happen."""
    config = _config(tmp_path)

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    printed = capsys.readouterr().out
    assert "a position gets about   $500.00" in printed
    assert "cannot be traded here" in printed
    assert "counted, not silent" in printed


def test_the_max_loss_envelope_says_what_it_assumes(wired, tmp_path, capsys):
    """A stop-weighted budget bounds the loss only if every stop fills at its price, and
    a gap fills below it. Printing the number without the condition would present a
    floor on the loss as a ceiling on it."""
    config = _config(tmp_path)

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    printed = capsys.readouterr().out
    assert "max loss envelope       $500.00" in printed
    assert "at its stop price" in printed
    assert "floor on the" in printed and "not a ceiling" in printed


def test_an_undeclared_risk_budget_reports_no_envelope_rather_than_zero(wired, tmp_path, capsys):
    """Rendering an unbounded book's envelope as zero would print the most reassuring
    possible number for the least bounded possible book."""
    config = _config(tmp_path, limits={**VALIDATED_BOOK, "max_total_risk": None})

    _run(["small-real", "--config", str(config), "--scale", "0.05", "--preflight"])

    assert "not computable (no max_total_risk declared)" in capsys.readouterr().out


def test_the_preflight_is_printed_even_when_the_run_will_proceed(wired, tmp_path, capsys):
    """Mandatory, with no flag to turn it off: a check you have to remember to ask for
    is one that gets skipped exactly when it matters."""
    parser = cli.build_parser()
    small = parser._subparsers._group_actions[0].choices["small-real"]
    flags = {option for action in small._actions for option in action.option_strings}

    assert "--no-preflight" not in flags and "--quiet" not in flags


# --- what it records, and what it must never touch ---------------------------------
def _start_and_let_the_stream_fail(config, extra=()):
    """Drive the command through to the engine, which cannot stream from a fixture.

    The failure is the point of the fixture, not an accident: it lands *after* the
    session header is written, which is exactly the ordering worth pinning. A run that
    dies on its first bar must still have recorded what contract it was about to trade,
    or its telemetry is a set of fills with no denominator.
    """
    with pytest.raises(RuntimeError, match="does not support streaming"):
        _run(["small-real", "--config", str(config), "--scale", "0.05", *extra])


def test_the_session_header_records_the_contract_the_fills_will_belong_to(wired, tmp_path):
    from tradeflow.execution.ledger import PositionLedger, small_real_ledger_path

    _start_and_let_the_stream_fail(_config(tmp_path))

    (session,) = PositionLedger(small_real_ledger_path()).sessions()

    assert session["capital"] == 10_000.0
    assert session["validated_capital"] == VALIDATED_CAPITAL
    assert session["scale"] == 0.05
    assert session["capital_source"] == "--scale 0.05"
    assert session["broker_mode"] == "paper"
    assert session["strategy"] == "demo_trend"
    # The scaled book, not the validated one: these records describe what was traded.
    assert session["book"]["max_position_size"] == 500.0
    assert session["validated_book"]["max_position_size"] == 10_000.0
    assert session["universe"] == ["AAA"]


def test_nothing_reaches_the_research_journal(wired, tmp_path):
    """It must not count toward the multiple-testing total or the deflated Sharpe: a run
    that measures its own execution has searched nothing. A journaled trial here would
    permanently raise the bar for every future candidate in the family, for evidence
    that is not evidence of a strategy at all."""
    from tradeflow.settings import trial_journal_path

    before = trial_journal_path().read_text() if trial_journal_path().exists() else None

    _start_and_let_the_stream_fail(_config(tmp_path))

    after = trial_journal_path().read_text() if trial_journal_path().exists() else None
    assert after == before


def test_the_telemetry_goes_to_its_own_ledger_not_the_live_one(wired, tmp_path):
    """Every roll-up over a ledger is an average, and averaging a full-size book's fills
    with the same book's fills at a twentieth of it describes neither."""
    from tradeflow.execution.ledger import default_ledger_path, small_real_ledger_path

    _start_and_let_the_stream_fail(_config(tmp_path))

    assert small_real_ledger_path().exists()
    assert not default_ledger_path().exists()


def test_the_run_says_how_to_read_its_own_telemetry(wired, tmp_path, capsys):
    """Telemetry nobody can find is telemetry nobody checks, and the command that reads
    it differs between an installed copy and a checkout."""
    from tradeflow.execution.ledger import small_real_ledger_path

    _start_and_let_the_stream_fail(_config(tmp_path))

    printed = capsys.readouterr().out
    assert "execution-report --ledger" in printed
    assert str(small_real_ledger_path()) in printed


def test_the_scaled_book_is_what_the_engine_is_actually_handed(wired, tmp_path, monkeypatch):
    """The mode's whole claim, and the first version of this test did not check it.

    Asserting the *recorded* contract proves the arithmetic and nothing else: a book
    computed correctly and never applied to the strategy is the same as no book, and a
    mutation deleting the assignment passed every test here. So this captures the
    strategy the engine is constructed with and asks it to size a position — the number
    the trade clock would really use.

    At $10,000 of capital and $100 a share the risk target alone wants ~66 shares; the
    scaled $500 ceiling allows 5. The validated $10,000 ceiling would allow 100, so the
    two books give visibly different answers and the assertion cannot pass by accident.
    """
    captured = {}

    class _Spy:
        def __init__(self, strategy, *args, **kwargs):
            captured["strategy"] = strategy
            raise RuntimeError("does not support streaming")

    monkeypatch.setattr("tradeflow.engine.live.LiveEngine", _Spy)
    _start_and_let_the_stream_fail(_config(tmp_path))

    strategy = captured["strategy"]
    assert strategy.position_limits()["max_position_size"] == 500.0
    assert strategy.calculate_position_size(10_000.0, 100.0) == pytest.approx(5.0)
    # And the shape of the book is untouched, which is the other half of the rule.
    assert strategy.position_limits()["max_positions"] == 8
    assert strategy.position_limits()["max_gross_exposure"] == 0.80
    assert strategy.position_limits()["min_notional"] == 50.0


def test_the_recorded_contract_matches_the_book_that_was_applied(wired, tmp_path):
    """The telemetry has to describe the run it belongs to. Two statements of one book —
    the strategy's and the session header's — and the fills only mean something against
    the right one."""
    from tradeflow.execution.ledger import PositionLedger, small_real_ledger_path

    _start_and_let_the_stream_fail(_config(tmp_path))

    (session,) = PositionLedger(small_real_ledger_path()).sessions()
    assert session["book"] == {
        "max_positions": 8,
        "max_position_size": 500.0,
        "max_total_risk": 0.05,
        "max_gross_exposure": 0.80,
        "max_net_exposure": 0.30,
        "min_notional": 50.0,
    }


def test_the_preflight_warns_about_a_book_this_run_did_not_open(monkeypatch, tmp_path, capsys):
    """Found by running the preflight against a real paper account holding twelve
    positions. The engine adopts them, their exits land in this session's telemetry at
    the size they were opened at, and — twelve against a scaled book of eight — no entry
    could be admitted at all. The operator would have seen a run that traded nothing."""
    from tradeflow.brokers.base import Position

    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PAPER_TRADE", "true")
    held = [Position(f"S{i}", 10, "long", 100.0, 100.0, 1_000.0, 0.0) for i in range(12)]
    monkeypatch.setattr(
        cli,
        "build_data_and_broker",
        lambda **kw: (
            FakeBroker(buying_power=100_000.0, positions=held),
            MarketDataClient(DictMarketData({"AAA": _frame()})),
        ),
    )
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA"])

    _run(["small-real", "--config", str(_config(tmp_path)), "--scale", "0.05", "--preflight"])

    printed = capsys.readouterr().out
    assert "ADOPTED BOOK" in printed
    assert "12 of 8" in printed
    assert "measure no entry execution at all" in printed


def test_the_session_header_records_what_the_run_inherited(wired, tmp_path, monkeypatch):
    """So a reader of the telemetry can tell this session's own fills from the exits of
    positions it inherited at another size."""
    from tradeflow.brokers.base import Position
    from tradeflow.execution.ledger import PositionLedger, small_real_ledger_path

    held = [Position("OLD", 10, "long", 100.0, 100.0, 1_000.0, 0.0)]
    monkeypatch.setattr(
        cli,
        "build_data_and_broker",
        lambda **kw: (
            FakeBroker(buying_power=100_000.0, positions=held),
            MarketDataClient(DictMarketData({"AAA": _frame()})),
        ),
    )

    _start_and_let_the_stream_fail(_config(tmp_path))

    (session,) = PositionLedger(small_real_ledger_path()).sessions()
    assert session["adopted_positions"] == 1
    assert session["adopted_symbols"] == ["OLD"]
