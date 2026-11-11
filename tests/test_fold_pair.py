"""``train_days`` and ``test_days`` are one setting, not two.

A lone ``--train-days 120`` used to be parsed and discarded: the fold builder derived
both lengths from the fold count and the run validated at a geometry nobody asked for,
printing nothing. Giving exactly one is now refused at the fold builder, which every
caller reaches, and earlier on the surfaces so the refusal precedes any data read.
"""

from datetime import datetime

import pytest

from quantred import cli
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator, require_fold_pair
from quantred.services import analysis
from tests.fakes import FakeMarketData
from tests.test_walk_forward import END, START, PeriodicStrategy, _validator


@pytest.mark.parametrize(
    ("given", "kwargs"),
    [("train_days", {"train_days": 120}), ("test_days", {"test_days": 30})],
)
def test_the_fold_builder_refuses_one_of_the_pair(given, kwargs):
    with pytest.raises(ValueError) as info:
        _validator().build_folds(START, END, embargo_days=2, **kwargs)
    message = str(info.value)
    other = "test_days" if given == "train_days" else "train_days"
    assert given in message and "pair" in message and other in message
    assert "n_folds" in message


def test_the_fold_builder_still_accepts_both_or_neither():
    both, _ = _validator().build_folds(START, END, embargo_days=2, train_days=120, test_days=30)
    neither, _ = _validator().build_folds(START, END, embargo_days=2)
    assert both and neither
    assert (both[0].is_end - both[0].is_start).days == 120
    assert (both[0].oos_end - both[0].oos_start).days >= 29


def test_every_path_to_the_fold_builder_goes_through_the_check():
    """`run` and `evaluate_config` both build folds, so a lone value fails in each."""
    validator = _validator()
    with pytest.raises(ValueError, match="pair"):
        validator.run(["AAA"], START, END, train_days=120)
    with pytest.raises(ValueError, match="pair"):
        validator.evaluate_config(["AAA"], START, END, {}, test_days=30)


def test_the_service_refuses_before_reading_anything():
    client = MarketDataClient(FakeMarketData(["AAA", "SPY"], n=600, freq="1D"))
    with pytest.raises(analysis.RunRefused, match="train_days was given without test_days"):
        analysis.run_walk_forward(client, "demo_trend", ["AAA"], START, END, train_days=120)
    with pytest.raises(analysis.RunRefused, match="test_days was given without train_days"):
        analysis.run_draft_walk_forward(client, "code", ["AAA"], START, END, test_days=30)


@pytest.mark.parametrize("flag", ["--train-days", "--test-days"])
def test_the_cli_refuses_a_lone_flag_in_one_line_before_any_work(monkeypatch, capsys, flag):
    monkeypatch.setattr(
        cli,
        "resolve_universe",
        lambda *a, **k: pytest.fail("the universe was resolved before the refusal"),
    )
    monkeypatch.setattr("sys.argv", ["quantred", "walkforward", "--strategy", "demo_trend", flag, "40"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code not in (0, None)
    message = str(exit_info.value.code)
    assert flag in message and "pair" in message and "--folds" in message
    assert "\n" not in message.strip()
    assert "Traceback" not in capsys.readouterr().err


def test_require_fold_pair_accepts_the_defaults():
    require_fold_pair(None, None)
    require_fold_pair(10, 5)


def test_the_helper_is_what_the_validator_class_exports():
    assert WalkForwardValidator.build_folds.__module__ == require_fold_pair.__module__
    assert isinstance(START, datetime) and PeriodicStrategy
