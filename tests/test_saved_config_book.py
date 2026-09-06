"""A saved config trades the book it was validated at.

`position_limits` is not a tunable parameter, so anything writing a config from a
validated result has to carry it deliberately. Omitting it does not leave the book
unspecified — it resolves at load to the strategy class's default of one position. A
config whose own provenance records that eight were validated then silently trades one,
and the file disagrees with itself in the direction that costs money.

Every surface that can write a runnable config is checked here, and the structural test
below enumerates them from the source rather than from a list somebody remembered.
"""

import ast
import json
from datetime import datetime

import pandas as pd
import pytest

from tradeflow.cli import build_parser
from tradeflow.optimization.config_store import DEFAULT_CONFIG_DIR, load_config
from tradeflow.services.analysis import recorded_book, walk_forward_recipe
from tradeflow.services.audit import journal_trial
from tradeflow.store.trials import db_path_for_journal

BOOK = {"max_positions": 8, "max_total_risk": 0.2}


#: A complete parameter set, the way a real walk-forward journals its chosen config.
#: Read from the strategy rather than written out here, so this fixture cannot drift
#: into being a set of params the strategy could not actually be built from.
def _chosen_params():
    from tradeflow.services.registry import STRATEGIES

    return dict(STRATEGIES["demo_trend"].create_with_defaults().config.get("params") or {}) or {
        name: spec["default"] for name, spec in STRATEGIES["demo_trend"].PARAM_RANGES.items()
    }


def _journal_walkforward(journal, limits=None):
    """A walk-forward trial, whose book lives in its recipe rather than its params."""
    return journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA", "BBB", "CCC"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 29),
        params=_chosen_params(),
        metrics={"sharpe_ratio": 1.1},
        extra={"promotable": True},
        returns=pd.Series([0.001] * 120, index=pd.date_range("2024-01-02", periods=120)),
        dedup_params=walk_forward_recipe(
            mode="anchored",
            n_folds=None,
            train_days=252,
            test_days=63,
            embargo_days=5,
            holdout_days=60,
            method="grid",
            objective="sharpe_ratio",
            max_evals=50,
            seed=42,
            cost_key={},
            limits=BOOK if limits is None else limits,
        ),
        path=journal,
    )


def _promote(journal, trial_id, name, monkeypatch):
    monkeypatch.setattr("tradeflow.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)
    args = build_parser().parse_args(
        [
            "trials",
            "promote",
            trial_id,
            "--save-config",
            name,
            "--db",
            str(db_path_for_journal(journal)),
            "--journal",
            str(journal),
        ]
    )
    args.func(args)
    return json.loads((DEFAULT_CONFIG_DIR / name).read_text())


# --- the defect ---------------------------------------------------------------------
def test_a_promoted_config_trades_the_book_its_provenance_says_was_validated(tmp_path, monkeypatch):
    """The bug. A promoted config recorded `_limits: {max_positions: 8}` inside its
    provenance and carried no `position_limits` of its own, so a run from it inherited
    the class default of one position — one file, two answers, and the one a live run
    obeys is the one nothing ever validated."""
    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    config = _promote(journal, trial_id, str(tmp_path / "promoted.json"), monkeypatch)

    validated = config["provenance"]["campaign"]["recipe"]["folded_into_identity"]["_limits"]
    assert config["position_limits"] == validated
    assert config["position_limits"]["max_positions"] == 8


def test_the_promoted_book_survives_a_reload_and_reaches_a_constructed_strategy(tmp_path, monkeypatch):
    """End to end, through the path a run actually takes: promote, reload the file, and
    build the strategy the way a backtest or a live session would. The number that comes
    out the far end is the one that was validated."""
    from tradeflow.services.registry import STRATEGIES
    from tradeflow.strategies.base import build_with_limits

    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    _promote(journal, trial_id, str(tmp_path / "roundtrip.json"), monkeypatch)

    payload = load_config(tmp_path / "roundtrip.json")
    strategy = build_with_limits(
        STRATEGIES[payload["strategy"]], payload["params"], payload.get("position_limits")
    )

    assert strategy.position_limits()["max_positions"] == 8
    assert strategy.position_limits()["max_total_risk"] == pytest.approx(0.2)


def test_a_backtest_trial_promotes_its_book_too(tmp_path, monkeypatch):
    """A backtest journals its dedup params *as* its params, so its book is in the
    params rather than in a recipe. A resolver that looked in one place would work for
    one kind and silently return nothing for the other."""
    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={**_chosen_params(), "_limits": {"max_positions": 6}, "_cost": {}},
        metrics={"sharpe_ratio": 1.0},
        extra={"promotable": True},
        path=journal,
    )

    config = _promote(journal, trial_id, str(tmp_path / "bt.json"), monkeypatch)

    assert config["position_limits"] == {"max_positions": 6}
    # And the reserved keys still do not leak into the runnable params.
    assert not any(k.startswith("_") for k in config["params"])


def test_a_trial_that_recorded_no_book_says_nothing_rather_than_inventing_one(tmp_path, monkeypatch):
    """Both directions, and the absent-is-not-zero half. A trial predating limits in the
    dedup identity keyed exactly as it did before; writing a book it never had would be
    manufacturing the evidence this whole path exists to preserve."""
    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params=_chosen_params(),
        metrics={"sharpe_ratio": 1.0},
        extra={"promotable": True},
        path=journal,
    )

    config = _promote(journal, trial_id, str(tmp_path / "old.json"), monkeypatch)

    assert "position_limits" not in config


# --- the resolver ------------------------------------------------------------------
def test_the_book_is_resolved_from_the_first_source_that_recorded_one():
    assert recorded_book({"_limits": BOOK}) == BOOK
    assert recorded_book(None, {"_limits": BOOK}) == BOOK
    assert recorded_book({}, {}) is None
    # An empty declaration is not a book. `limits_key` omits unset limits entirely, so
    # `_limits: {}` never occurs — and if it did, it says nothing was declared.
    assert recorded_book({"_limits": {}}) is None
    # The first source wins: a walk-forward's recipe is authoritative over its params.
    assert recorded_book({"_limits": {"max_positions": 8}}, {"_limits": {"max_positions": 1}}) == {
        "max_positions": 8
    }


# --- the structural guard ------------------------------------------------------------
def test_every_call_site_that_writes_a_runnable_config_states_its_book():
    """Enumerated from the source rather than from a list somebody remembered.

    Four places write a config a run can be frozen from, and each one lost the book in
    its own way: `trials promote` discarded it with the reserved keys, `walkforward
    --save-config` wrote the *class* defaults instead of the run's book (so
    round-tripping a config shrank it), the research agent wrote none, and the MCP
    service had no parameter to put one in.
    """
    import tradeflow.cli
    import tradeflow.research.agent
    import tradeflow.services.configs

    missing = []
    for module in (tradeflow.cli, tradeflow.research.agent, tradeflow.services.configs):
        source = ast.parse(open(module.__file__).read())
        for node in ast.walk(source):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name != "save_config":
                continue
            keywords = {kw.arg for kw in node.keywords}
            # A call with no `strategy` is the definition's own delegation, not a
            # writer choosing what to record.
            if "strategy" not in keywords:
                continue
            if "position_limits" not in keywords:
                missing.append(f"{module.__name__}:{node.lineno}")

    assert missing == [], f"save_config call sites that do not state a book: {missing}"


def test_the_walk_forward_saver_writes_the_book_it_validated_not_the_class_default():
    """The other half of the same defect, and the one that made it recur: the saver read
    `create_with_defaults().position_limits()`, so a walk-forward run against a config
    asking for eight positions validated eight and saved one."""
    import inspect

    from tradeflow import cli

    source = inspect.getsource(cli.cmd_walkforward)
    saver = source[source.index("path = save_config(") : source.index("provenance=provenance")]

    assert "create_with_defaults" not in saver
    assert "config_position_limits" in saver
