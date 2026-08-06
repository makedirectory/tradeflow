"""Service-layer wiring for the conditional risk model: compute_risk /
construct_portfolio / compute_attribution's ``conditional`` flag, the MZ/QLIKE
evidence-gate service function, and the net-of-cost A/B harness. Offline and
deterministic — built on the same ``DictMarketData``/``make_ohlcv`` fakes as the
rest of the suite.
"""

from src.marketdata.client import MarketDataClient
from src.services import analysis
from tests.fakes import DictMarketData, make_ohlcv


def _client(symbols, n=500, benchmark="SPY"):
    data = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate([*symbols, benchmark])}
    return MarketDataClient(DictMarketData(data)), data


# --- compute_risk --------------------------------------------------------------
def test_compute_risk_conditional_off_is_unchanged():
    symbols = [f"S{i}" for i in range(6)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    baseline = analysis.compute_risk(client, symbols, as_of)
    off = analysis.compute_risk(client, symbols, as_of, conditional=None)
    assert baseline["top_risk_contributors"] == off["top_risk_contributors"]
    assert "sigma_regime" not in off


def test_compute_risk_conditional_reports_sigma_regime():
    symbols = [f"S{i}" for i in range(6)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    r = analysis.compute_risk(client, symbols, as_of, conditional="ewma")
    assert r["conditional"] == "ewma"
    assert "sigma_regime" in r
    assert r["positive_definite"]
    assert set(r["sigma_regime"]) >= {"method", "lambda", "sigma_regime", "mean_sigma_regime"}


def test_compute_risk_conditional_factor_model_also_conditions():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    r = analysis.compute_risk(client, symbols, as_of, model="factor", conditional="ewma")
    assert r["positive_definite"]
    assert "sigma_regime" in r
    assert "factor_risk_share" in r


# --- construct_portfolio --------------------------------------------------------
def test_construct_portfolio_conditional_off_is_unchanged():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    baseline = analysis.construct_portfolio(client, "volume_spike", symbols, as_of)
    off = analysis.construct_portfolio(client, "volume_spike", symbols, as_of, conditional=None)
    assert baseline["weights"] == off["weights"]
    assert "sigma_regime" not in off


def test_construct_portfolio_conditional_stays_feasible_and_reports_regime():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    r = analysis.construct_portfolio(client, "volume_spike", symbols, as_of, conditional="ewma")
    assert r["feasible"]
    assert r["conditional"] == "ewma"
    assert "sigma_regime" in r
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-6


# --- compute_attribution: te_by_regime ------------------------------------------
def test_compute_attribution_conditional_adds_te_by_regime():
    symbols = [f"S{i}" for i in range(10)]
    client, data = _client(symbols, n=900)
    start = data["S0"].index[100].to_pydatetime()
    end = data["S0"].index[800].to_pydatetime()

    r = analysis.compute_attribution(
        client, "volume_spike", symbols, start, end, conditional="ewma", n_points=30
    )
    assert r["periods"] >= 5
    assert r["conditional"] == "ewma"
    te_by_regime = r["te_by_regime"]
    assert set(te_by_regime) == {"low", "mid", "high"}
    total_n = sum(te_by_regime[k].get("n", 0) for k in te_by_regime)
    assert total_n == r["periods"]
    for label in ("low", "mid", "high"):
        row = te_by_regime[label]
        if row.get("n", 0) > 1:
            assert row["predicted_te"] >= 0.0
            assert row["realized_te"] >= 0.0


def test_compute_attribution_without_conditional_has_empty_te_by_regime_when_no_vol_variation():
    # Without --conditional (default None), _build_covariance still runs, and
    # te_by_regime is computed from whatever bench-vol variation exists — this just
    # confirms the field is always present (possibly {}), never a KeyError upstream.
    symbols = [f"S{i}" for i in range(10)]
    client, data = _client(symbols, n=900)
    start = data["S0"].index[100].to_pydatetime()
    end = data["S0"].index[800].to_pydatetime()

    r = analysis.compute_attribution(client, "volume_spike", symbols, start, end, n_points=30)
    assert "te_by_regime" in r
    assert r["conditional"] is None


# --- evaluate_conditional_risk (the evidence gate) ------------------------------
def test_evaluate_conditional_risk_reports_per_name_and_pooled():
    symbols = [f"S{i}" for i in range(6)]
    client, data = _client(symbols, n=900)
    start = data["S0"].index[100].to_pydatetime()
    end = data["S0"].index[800].to_pydatetime()

    r = analysis.evaluate_conditional_risk(client, symbols, start, end, n_points=40)
    assert r["n_names"] == 6
    assert set(r["pooled"]) == {"ewma", "har", "unconditional"}
    assert set(r["per_name"]) == set(symbols)
    assert r["best_method_pooled_qlike"] in {"ewma", "har", "unconditional"}
    assert isinstance(r["gate_passed"], bool)


def test_evaluate_conditional_risk_insufficient_history_reports_note():
    symbols = ["S0"]
    data = {"S0": make_ohlcv(n=20, seed=0, freq="1D")}
    client = MarketDataClient(DictMarketData(data))
    start = data["S0"].index[0].to_pydatetime()
    end = data["S0"].index[-1].to_pydatetime()

    r = analysis.evaluate_conditional_risk(client, symbols, start, end, min_obs=60)
    assert r["n_names"] == 0
    assert "note" in r


# --- run_conditional_risk_ab (net-of-cost A/B) ----------------------------------
def test_run_conditional_risk_ab_produces_both_variants():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols, n=900)
    start = data["S0"].index[100].to_pydatetime()
    end = data["S0"].index[800].to_pydatetime()

    r = analysis.run_conditional_risk_ab(client, "volume_spike", symbols, start, end, n_points=10, horizon=21)
    assert r["periods"] >= 2
    assert set(r["summaries"]) == {"unconditional", "conditional"}
    for name, s in r["summaries"].items():
        assert s["periods"] >= 2
        assert "net_ir" in s
        assert "mean_turnover" in s
    assert r["winner_net_ir"] in {"unconditional", "conditional"}


def test_run_conditional_risk_ab_insufficient_window_reports_note():
    symbols = ["S0", "S1"]
    data = {s: make_ohlcv(n=30, seed=i, freq="1D") for i, s in enumerate(["S0", "S1", "SPY"])}
    client = MarketDataClient(DictMarketData(data))
    start = data["S0"].index[0].to_pydatetime()
    end = data["S0"].index[-1].to_pydatetime()

    r = analysis.run_conditional_risk_ab(client, "volume_spike", symbols, start, end, horizon=21)
    assert r["periods"] == 0
    assert "note" in r
