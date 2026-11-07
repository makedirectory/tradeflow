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
