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
