"""A promoted config trades under the cost model it was validated at.

`trials promote` read `_cost` out of the trial's params. A walk-forward or verdict keeps
it in its recipe instead, so the config was written with no `cost` and a run from it
priced trades at the defaults - cheaper than the validation - with nothing on the load
line to say so.
"""

import json
from datetime import datetime

import pandas as pd

from quantred.cli import apply_run_config, build_cost_model, build_parser
from quantred.services.analysis import recorded_cost, walk_forward_recipe
from quantred.services.audit import journal_trial
from quantred.services.registry import STRATEGIES
from tests.test_saved_config_book import _chosen_params, _promote

COST = {"gross": False, "commission_bps": 7.0, "impact_eta": 0.5, "borrow_bps": 80.0}


def _walkforward(journal):
    return journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA", "BBB"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 29),
        params=_chosen_params(),
        metrics={"sharpe_ratio": 1.1},
        extra={"promotable": True},
        returns=pd.Series([0.001] * 120, index=pd.date_range("2024-01-02", periods=120)),
        context={"capital": 25_000.0},
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
            cost_key={**COST, "_vintage": "abc123"},
            strategy_class=STRATEGIES["demo_trend"],
            capital=25_000.0,
            benchmark=None,
        ),
        path=journal,
    )


def test_a_promoted_walkforward_prices_trades_at_the_validated_cost(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("quantred.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    config = _promote(journal, _walkforward(journal), str(tmp_path / "p.json"), monkeypatch)

    # The vintage stamp fingerprints the data, and a reusable config must not pin one.
    assert config["cost"] == COST

    # Through the path a run takes: the cost model the backtest would actually build.
    args = build_parser().parse_args(["backtest", "--config", str(tmp_path / "p.json")])
    apply_run_config(args)
    model = build_cost_model(args)
    assert model.commission_rate == 7.0 / 1e4
    assert model.impact_eta == 0.5
    assert model.annual_borrow_rate == 80.0 / 1e4
    assert "commission_bps=7.0" in capsys.readouterr().out


def test_a_config_with_no_recorded_cost_says_so_on_the_load_line(tmp_path, capsys):
    path = tmp_path / "nocost.json"
    path.write_text(json.dumps({"strategy": "demo_trend", "scanner": "none", "params": _chosen_params()}))
    args = build_parser().parse_args(["backtest", "--config", str(path)])
    apply_run_config(args)
    assert "cost=<none recorded" in capsys.readouterr().out


def test_a_promoted_config_names_the_scanner_the_way_save_config_does(tmp_path, monkeypatch):
    monkeypatch.setattr("quantred.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    config = _promote(journal, _walkforward(journal), str(tmp_path / "s.json"), monkeypatch)
    assert config["scanner"] == "none"


def test_the_mcp_save_config_path_carries_the_campaigns_cost_too(tmp_path, monkeypatch):
    """The other surface that writes a runnable config: with campaign material in its
    provenance it takes the validated cost the way it takes the validated book."""
    from quantred.services import configs

    monkeypatch.setattr("quantred.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    campaign = {"recipe": {"folded_into_identity": {"_cost": {**COST, "_vintage": "v"}}}}
    result = configs.save_config(
        "viamcp", strategy="demo_trend", params=_chosen_params(), provenance={"campaign": campaign}
    )
    assert json.loads(open(result["path"]).read())["cost"] == COST


def test_the_resolver_skips_the_vintage_and_returns_none_when_unrecorded():
    assert recorded_cost({"_cost": {"gross": True, "_vintage": "x"}}) == {"gross": True}
    assert recorded_cost(None, {"_cost": COST}) == COST
    assert recorded_cost({}, {"_cost": {}}) is None


def test_a_backtest_campaign_block_names_its_cost_so_mcp_save_config_writes_it(tmp_path, monkeypatch):
    """Found in review: a backtest's campaign block has no recipe (its identity is its
    params), so a config written from it over MCP carried no cost where `trials promote`
    did - the two surfaces disagreeing about one trial."""
    from quantred.marketdata.client import MarketDataClient
    from quantred.optimization import config_store
    from quantred.services import analysis, audit, campaign, configs
    from quantred.utils.timeutils import parse_market_date as day
    from tests.fakes import FakeMarketData

    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config_store, "DEFAULT_CONFIG_DIR", tmp_path / "configs")
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="1D"))
    fresh = analysis.run_backtest(
        client, "demo_trend", ["AAA", "BBB"], day("2023-01-02"), day("2024-06-01"), commission_bps=7
    )
    with audit.open_trial_store() as store:
        material = campaign.campaign_material(store, fresh["trial_id"])
    assert material["recipe"]["folded_into_identity"]["_cost"]["commission_bps"] == 7
    saved = configs.save_config("bt", strategy="demo_trend", params={}, provenance={"campaign": material})
    assert json.loads(open(saved["path"]).read())["cost"]["commission_bps"] == 7
