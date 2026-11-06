"""Labels that must say what a number is: sentinels, undeflated values, vacuous passes."""

import pytest

from tests.fakes import make_ohlcv


# --- capacity: a search bound is not a capacity ---------------------------------------
def _book():
    syms = [f"S{i}" for i in range(6)]
    weights = {s: 1.0 / len(syms) for s in syms}
    bars = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(syms)}
    return weights, bars


@pytest.mark.parametrize(
    "alpha, status",
    [(0.0, "no_edge"), (-0.01, "no_edge"), (1e-12, "below_search_floor"), (1e6, "above_search_ceiling")],
)
def test_capacity_names_the_bound_instead_of_returning_it(alpha, status):
    from tradeflow.services.analysis import _capacity

    weights, bars = _book()
    capital, got = _capacity(weights, bars, gross_alpha=alpha, holding_period_years=1 / 12)
    assert got == status
    assert capital is None


def test_capacity_text_says_the_bound_in_words_and_never_prints_a_sentinel():
    from tradeflow.analytics.reporting import capacity_text

    assert capacity_text({}) is None
    measured = capacity_text({"capacity_status": "measured", "capacity_capital": 2_500_000.0})
    assert "$2,500,000" in measured
    for status in ("no_edge", "below_search_floor", "above_search_ceiling"):
        text = capacity_text({"capacity_status": status, "capacity_capital": None})
        assert text.startswith("not ")
        assert "≈" not in text


def test_html_portfolio_does_not_render_a_capacity_number_for_a_bound():
    from tradeflow.analytics.htmlreport import _portfolio_card

    d = {"capacity_status": "no_edge", "capacity_capital": None, "expected_active_return": -0.01}
    card = _portfolio_card({"feasible": True, "weights": {"A": 1.0}, "diagnostics": d})
    assert "Capacity" in card and "not applicable" in card


# --- attribution: say why periods were skipped, not that history is short -------------
def _attribution_client(n_names):
    from tests.fakes import DictMarketData
    from tradeflow.marketdata.client import MarketDataClient

    syms = [f"S{i}" for i in range(n_names)]
    data = {s: make_ohlcv(n=600, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    return MarketDataClient(DictMarketData(data)), syms


def test_attribution_failure_names_the_dominant_reason_not_history():
    from datetime import datetime

    from tradeflow.services import analysis

    client, syms = _attribution_client(2)  # too few names to build a covariance
    r = analysis.compute_attribution(
        client, "demo_trend", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=16
    )
    assert r["periods"] == 0
    assert r["skipped_periods"], "skip reasons must be counted"
    assert sum(r["skipped_periods"].values()) == r["rebalances_sampled"]
    assert "fewer than 3 names" in r["note"]
    assert "Insufficient overlapping history" not in r["note"]


def test_info_attribution_prints_the_ic_report_and_the_attribution_failure(monkeypatch, capsys):
    import argparse
    from datetime import datetime

    from tradeflow import cli

    client, syms = _attribution_client(6)
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: client)
    monkeypatch.setattr(cli, "apply_run_config", lambda a: None)
    monkeypatch.setattr(cli, "_print_attribution_report", lambda *a: print("ATTRIBUTION-SECTION"))
    args = argparse.Namespace(
        strategy="demo_trend",
        symbols=syms,
        start=datetime(2023, 1, 1),
        end=datetime(2024, 12, 31),
        source="strategy",
        scanner="demo_volume",
        benchmark="SPY",
        neutralize_factors=(),
        horizon=5,
        n_trials=1,
        cache=False,
        offline=True,
        cache_dir=None,
        attribution=True,
        html=None,
    )
    cli.cmd_info(args)
    out = capsys.readouterr().out
    assert "Information report" in out
    assert out.index("Information report") < out.index("ATTRIBUTION-SECTION")


# --- deflated Sharpe: do not call it deflated when no deflation was applied ----------
def test_html_metrics_label_an_undeflated_sharpe_and_carry_units():
    from tradeflow.analytics.htmlreport import _backtest_sections

    metrics = {
        "deflated_sharpe_ratio": 0.537,
        "deflation_applied": False,
        "total_return": 0.2274,
        "total_trades": 12,
        "max_drawdown": 3.5,
    }
    html_out = "".join(_backtest_sections({"metrics": metrics}, {}))
    assert "PSR, undeflated" in html_out
    assert "0.227%" in html_out or "0.23%" in html_out
    assert "3.50%" in html_out
    metrics["deflation_applied"] = True
    html_out = "".join(_backtest_sections({"metrics": metrics}, {}))
    assert "PSR, undeflated" not in html_out


def test_every_percent_metric_the_terminal_prints_has_a_declared_unit():
    from tradeflow.analytics import reporting
    from tradeflow.optimization.walk_forward import UNIT_DECLARED

    pct_keys = [k for k, _label, fmt in reporting._ROWS if fmt.endswith("%")]
    assert pct_keys
    assert set(pct_keys) <= UNIT_DECLARED


def test_optimization_payload_says_its_deflated_sharpe_is_not_deflated():
    from datetime import datetime

    from tests.fakes import DictMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis

    syms = ["S0", "S1", "S2"]
    data = {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    r = analysis.run_optimization(
        MarketDataClient(DictMarketData(data)),
        "demo_trend",
        syms,
        datetime(2023, 1, 1),
        datetime(2024, 12, 31),
        method="random",
        max_evals=3,
        seed=1,
    )
    assert "NOT deflated" in r["deflation_note"]


# --- a gate must not pass only because nothing traded ---------------------------------
def _result(oos_trades):
    from datetime import datetime

    from tradeflow.optimization.walk_forward import Fold, FoldResult, WalkForwardResult

    fold = FoldResult(
        fold=Fold(0, datetime(2024, 1, 1), datetime(2024, 3, 1), datetime(2024, 3, 2), datetime(2024, 5, 1)),
        is_best_params={},
        is_metrics={"max_drawdown": 5.0, "sharpe_ratio": 1.0},
        oos_metrics={"max_drawdown": 2.0 if oos_trades else 0.0, "sharpe_ratio": 0.0},
        oos_trades=oos_trades,
        n_trials=1,
    )
    return WalkForwardResult(
        folds=[fold],
        oos_aggregate={},
        holdout=None,
        holdout_params=None,
        efficiency=0.0,
        degradation={},
        n_trials_total=1,
        objective="sharpe_ratio",
    )


def test_drawdown_gate_does_not_pass_vacuously_with_no_oos_trades():
    quiet = _result(0).gate_report()["checks"]["oos_drawdown_vs_is"]
    assert quiet["passed"] is False
    assert "no OOS trades" in quiet["note"]
    traded = _result(40).gate_report()["checks"]["oos_drawdown_vs_is"]
    assert traded["passed"] is True  # the boundary case still passes


def test_walk_forward_payload_says_when_nothing_traded_out_of_sample():
    from datetime import datetime

    from tradeflow.services.analysis import NO_OOS_TRADES_NOTE, walk_forward_payload

    def build(n):
        return walk_forward_payload(
            _result(n),
            run_id="r",
            strategy="s",
            symbols=["A"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 1),
            mode="anchored",
            objective="sharpe_ratio",
            method="grid",
            gross=False,
            seed=1,
            gates=None,
        )

    assert build(0)["no_oos_trades_note"] == NO_OOS_TRADES_NOTE
    assert build(5)["no_oos_trades_note"] is None
