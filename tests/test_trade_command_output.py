"""What the order-capable commands tell an operator about the run they are about to affect.

Each test drives the real command in-process against a fake broker and reads the text
(or JSON) it prints, because the defects here were all in what was rendered, not in
what was computed.
"""

import argparse
import json

import pytest

from tests.fakes import FakeBroker
from tradeflow import cli
from tradeflow.brokers.base import Position


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


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Halt state and ledgers under this test's own root, never the session-wide one."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path / "state"))
    return tmp_path


def _broker(monkeypatch, positions=()):
    monkeypatch.setattr(
        cli, "build_data_and_broker", lambda *a, **k: (FakeBroker(positions=list(positions)), None)
    )


def _ledger_args(**overrides):
    defaults = {"ledger": None, "small_real": False, "json": False, "orders": False}
    return argparse.Namespace(**{**defaults, **overrides})


def _flatten_args(**overrides):
    defaults = {"confirm": True, "json": False, "reason": "r", "ledger": None, "small_real": False}
    return argparse.Namespace(**{**defaults, **overrides})


@pytest.mark.parametrize("paper, label", [(True, "PAPER"), (False, "LIVE - REAL MONEY")])
def test_flatten_and_reconcile_state_which_account_they_act_on(state, monkeypatch, capsys, paper, label):
    """`flatten` and `reconcile` printed a ledger label and never said whether the
    account was paper or real, while both preflights do. The refusal without --confirm
    did not name the account either."""
    monkeypatch.setattr(cli, "_paper_mode", lambda: paper)
    _broker(monkeypatch, [_position()])

    with pytest.raises(SystemExit) as refused:
        cli.cmd_flatten(_flatten_args(confirm=False))
    assert label in str(refused.value)

    cli.cmd_flatten(_flatten_args())
    assert f"broker mode: {label}" in capsys.readouterr().out

    _broker(monkeypatch)  # flat: a divergence would exit nonzero and this is not about that
    cli.cmd_reconcile(_ledger_args(ledger=str(state / "l.jsonl")))
    assert f"broker mode: {label}" in capsys.readouterr().out

    cli.cmd_reconcile(_ledger_args(ledger=str(state / "l.jsonl"), json=True))
    assert json.loads(capsys.readouterr().out)["broker_mode"] == label


def test_the_default_ledger_label_cannot_be_read_as_the_live_account(state):
    """`ledger: ... (live)` meant the live-session ledger, and read as the live account."""
    label = cli._describe_ledger(None)
    assert "(live-session ledger)" in label
    assert not label.endswith("(live)")


def _write_config(tmp_path, capital=8000.0, limits=None, name="cfg.json"):
    """A config in the shape `save_config` writes, built by hand so the strategy params
    are exactly the ones the demo strategy accepts."""
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
        "symbols": ["AAA", "BBB"],
        "position_limits": limits
        or {"max_positions": 4, "max_position_size": 900.0, "min_notional": 50.0},
    }
    if capital is not None:
        payload["capital"] = capital
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_the_provenance_line_names_the_limits_a_flag_overrode(tmp_path, capsys):
    """`Config ... position_limits=<config>` was printed while the table below it showed
    the flag-overridden caps; only capital switched to `<--capital given>`, so the file
    looked like the source of numbers it did not supply."""
    path = _write_config(tmp_path)

    cli.apply_run_config(
        cli.parse_cli(
            ["live", "--config", str(path), "--max-positions", "2", "--max-position-size", "500"]
        )
    )
    line = capsys.readouterr().out.splitlines()[0]
    assert "position_limits=<config + --max-positions, --max-position-size given>" in line
    assert "position_limits=<config>" not in line


def test_the_provenance_line_stays_config_when_no_limit_flag_was_typed(tmp_path, capsys):
    """The other direction: untyped defaults must not be reported as overrides."""
    path = _write_config(tmp_path)

    cli.apply_run_config(cli.parse_cli(["live", "--config", str(path)]))
    assert "position_limits=<config>" in capsys.readouterr().out


# --- the book a run checks against ----------------------------------------------
def _wired_dry_run(monkeypatch):
    """The CLI wired to the real data client over a fake provider, with no broker."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient

    monkeypatch.setattr(
        "tradeflow.services.data.build_data_client",
        lambda **kw: MarketDataClient(FakeMarketData(["AAA", "BBB"], n=200, freq="1D")),
    )
    monkeypatch.setattr(cli, "build_data_and_broker", lambda *a, **k: pytest.fail("broker factory reached"))
    monkeypatch.setattr(cli, "resolve_universe", lambda *a, **k: ["AAA", "BBB"])


def test_a_dry_run_prints_the_book_it_checks_against(tmp_path, monkeypatch, capsys):
    """`WOULD BIND: 0` was unreadable: the dry run printed capital and buckets but not the
    position limits, gross, net or floor it was deciding against. The flat starting book
    was also stated as fact although a dry run cannot see the account."""
    _wired_dry_run(monkeypatch)
    path = _write_config(tmp_path)

    cli.cmd_live(
        cli.parse_cli(["live", "--config", str(path), "--dry-run", "--max-positions", "2"])
    )
    out = capsys.readouterr().out
    assert "max positions         2" in out
    assert "max position size" in out and "min notional" in out and "max gross exposure" in out
    assert "assumed flat" in out
    assert "starting book         flat" not in out


def test_a_dry_run_json_carries_the_book(tmp_path, monkeypatch, capsys):
    """The JSON reader has the same question as the text reader."""
    _wired_dry_run(monkeypatch)
    path = _write_config(tmp_path)

    cli.cmd_live(cli.parse_cli(["live", "--config", str(path), "--dry-run", "--json", "--max-positions", "2"]))
    out = capsys.readouterr().out
    # `apply_run_config` prints its provenance lines ahead of the document; that is a
    # separate matter from what the document holds.
    report = json.loads(out[out.index("{"):])
    assert report["position_limits"]["max_positions"] == 2


def test_preflight_together_with_dry_run_is_refused_not_dropped(tmp_path, monkeypatch):
    """`--preflight --dry-run` ran the dry run and discarded the preflight without a
    word, so the operator believed the account had been checked."""
    _wired_dry_run(monkeypatch)
    path = _write_config(tmp_path)

    with pytest.raises(SystemExit) as refused:
        cli.cmd_live(cli.parse_cli(["live", "--config", str(path), "--dry-run", "--preflight"]))
    assert "--preflight and --dry-run" in str(refused.value)


def test_a_book_whose_ceiling_is_below_the_floor_is_flagged_not_repaired(tmp_path, monkeypatch, capsys):
    """`--max-position-size 500 --min-notional 2000` refuses every entry and the report
    showed three ordinary-looking numbers. The warning is printed; the limits are left
    exactly as given."""
    _wired_dry_run(monkeypatch)
    path = _write_config(tmp_path)

    cli.cmd_live(
        cli.parse_cli(
            ["live", "--config", str(path), "--dry-run", "--capital", "3000",
             "--max-position-size", "500", "--min-notional", "2000"]
        )
    )
    out = capsys.readouterr().out
    assert "max position size $500.00 is below min notional $2,000.00" in out
    assert "$500.00" in out and "$2,000.00" in out  # not repaired


def test_a_consistent_book_gets_no_warning():
    """Boundary: a ceiling equal to the floor can place an order."""
    assert cli._book_warnings({"max_position_size": 2000.0, "min_notional": 2000.0}, 8000.0) == []
    assert cli._book_warnings({"max_position_size": 1999.0, "min_notional": 2000.0}, 8000.0)


def test_a_gross_cap_and_a_capital_below_the_floor_are_flagged():
    warned = cli._book_warnings({"max_gross_exposure": 0.1, "min_notional": 2000.0}, 8000.0)
    assert any("max gross exposure" in w for w in warned)
    warned = cli._book_warnings({"min_notional": 2000.0}, 1500.0)
    assert any("capital $1,500.00" in w for w in warned)


def test_the_live_preflight_says_a_default_ceiling_is_not_the_account(monkeypatch, capsys):
    """Without a config, `max position size $100,000.00` sat beside `capital this run the
    whole account` and read as though the account were the cap. It is the strategy
    class's own default."""
    from tradeflow.services.registry import STRATEGIES

    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    args = cli.parse_cli(["live", "--symbols", "AAA,BBB", "--scanner", "none", "--preflight"])
    for line in cli._book_limit_lines(args, strategy.position_limits(), None):
        print(line)
    assert "(strategy default, not set for this run)" in capsys.readouterr().out

    typed = cli.parse_cli(["live", "--symbols", "AAA", "--scanner", "none", "--max-position-size", "500"])
    lines = cli._book_limit_lines(typed, {"max_position_size": 500.0}, None)
    assert not any("strategy default" in line for line in lines)
