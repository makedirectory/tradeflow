"""Tests for the v2 alpha-forecast refinements.

Covers the three refinements the base refine/combine pipeline skipped:
  1. the per-signal Case test + Case-2 scaling (``IC·c_g·z`` vs ``ω·IC·z``),
  2. the IC-uncertainty **level** shrink with an honest effective T, and
  3. the equal-risk-contribution bucket diagnostic.

Offline and deterministic. The closed-form math is tested directly; the service
flow is exercised for structure and the Case-1 equivalence guard.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.alphas import refine
from src.alphas.base import AlphaContext, refine_alpha
from src.alphas.horizon import effective_sample_size
from src.analytics.information import risk_bucket_diagnostic
from src.data.panel import FeaturePanel
from src.marketdata.client import MarketDataClient
from src.services import analysis
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)


# --------------------------------------------------------------------------- #
# 1. The Case test
# --------------------------------------------------------------------------- #
def _signal_history(std_ts_by_name, n_obs=48, seed=0):
    """A time×name frame whose per-name time-series std matches ``std_ts_by_name``."""
    rng = np.random.default_rng(seed)
    cols = {name: rng.normal(0.0, s, n_obs) for name, s in std_ts_by_name.items()}
    return pd.DataFrame(cols)


def test_case1_signal_detected_when_vol_constant():
    # Per-name signal vol ~constant across names ⇒ Std_TS ⊥ ω ⇒ Case 1.
    names = [f"S{i}" for i in range(40)]
    omega = pd.Series({n: 0.10 + 0.005 * i for i, n in enumerate(names)})
    # Long history + a wide cross-section so the per-name sample-std noise (which is
    # uncorrelated with ω, but spuriously fits ~1/(K−1) of the variance) stays well
    # below the R² floor; the true relation is flat ⇒ Case 1.
    history = _signal_history({n: 0.30 for n in names}, n_obs=400)  # constant TS vol
    res = refine.case_test(history, omega, price_derived=True)
    assert res["case"] == 1
    assert res["engaged"] and not res["ambiguous"]
    assert res["r_squared"] <= 0.05


def test_case2_signal_detected_when_vol_proportional_to_omega():
    # Per-name signal vol ∝ ω ⇒ Std_TS = a + b·ω with high R² ⇒ Case 2.
    names = [f"S{i}" for i in range(20)]
    omega = pd.Series({n: 0.10 + 0.02 * i for i, n in enumerate(names)})
    history = _signal_history({n: 2.0 * omega[n] for n in names})  # vol ∝ ω
    res = refine.case_test(history, omega, price_derived=False)
    assert res["case"] == 2
    assert res["engaged"] and not res["ambiguous"]
    assert res["r_squared"] >= 0.25 and res["t_stat"] >= 2.0


def test_case_test_short_history_defaults_to_base_rate():
    # Below min_obs the test can't decide → the empirical base rate, flagged ambiguous.
    names = [f"S{i}" for i in range(20)]
    omega = pd.Series({n: 0.15 for n in names})
    history = _signal_history({n: 0.30 for n in names}, n_obs=10)  # too short
    price = refine.case_test(history, omega, price_derived=True)
    other = refine.case_test(history, omega, price_derived=False)
    assert price["case"] == 2 and other["case"] == 1
    assert not price["engaged"] and price["ambiguous"]


# --------------------------------------------------------------------------- #
# 2. c_g and the Case-2 scaling
# --------------------------------------------------------------------------- #
def test_case_scale_factor_is_vol_dimensioned_constant():
    # c_g = Std_CS{g}/Std_CS{g/ω}. For g ∝ ω exactly, g/ω is constant ⇒ Std→0 ⇒ c_g→∞;
    # for a realistic spread it lands between the min and max ω (a representative vol).
    g = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})
    omega = pd.Series({"A": 0.10, "B": 0.15, "C": 0.20, "D": 0.25, "E": 0.30})
    c_g = refine.case_scale_factor(g, omega, winsorize_limits=(0.0, 1.0))
    assert np.isfinite(c_g) and c_g > 0  # a finite, positive volatility-scale constant
    assert 0.05 <= c_g <= 1.0


def test_case2_replaces_per_name_vol_with_constant():
    # Case 2 uses one c_g for every name; Case 1 uses each name's own ω. So the two
    # alpha vectors are proportional only when ω is constant — here ω varies, so
    # Case 2 removes the high-vol tilt Case 1 imposes.
    raw = {f"S{i}": float(i - 6) for i in range(1, 13)}
    vols = {s: 0.10 + 0.02 * i for i, s in enumerate(raw)}
    p1 = FeaturePanel.for_universe(AS_OF, list(raw))
    p1.set("score", raw)
    p1.set("residual_vol", vols)
    p2 = FeaturePanel.for_universe(AS_OF, list(raw))
    p2.set("score", raw)
    p2.set("residual_vol", vols)

    refine_alpha(p1, AlphaContext(ic=0.05, scaling="case1"))
    refine_alpha(p2, AlphaContext(ic=0.05, scaling="case2"))
    a1, a2 = p1.get("alpha"), p2.get("alpha")
    z = p1.get("z")
    # Case 2 alpha is exactly IC·c_g·z (a constant scale on z).
    c_g = refine.case_scale_factor(pd.Series(raw), pd.Series(vols))
    for s in z.index:
        assert abs(a2[s] - 0.05 * c_g * z[s]) < 1e-9
    # And it is NOT the per-name Case-1 vector (the vol tilt is gone).
    assert not np.allclose(a1.to_numpy(), a2.to_numpy())


# --------------------------------------------------------------------------- #
# 3. The IC-uncertainty level shrink
# --------------------------------------------------------------------------- #
def test_level_shrink_reproduces_reference_anchors():
    # A good signal (IC 0.05, 5yr monthly) keeps ~13%; a great one (IC 0.10, 10yr) ~55%.
    assert abs(refine.level_shrink_factor(0.05, 60) - 0.13) < 0.005
    assert abs(refine.level_shrink_factor(0.10, 120) - 0.55) < 0.01


def test_level_shrink_limits():
    assert refine.level_shrink_factor(0.05, 1e12) > 0.999  # T→∞ ⇒ →1
    assert refine.level_shrink_factor(1e-6, 60) < 1e-6  # IC→0 ⇒ →0
    assert refine.level_shrink_factor(0.0, 60) == 0.0
    assert refine.level_shrink_factor(0.05, 0) == 0.0  # no data ⇒ →0


def test_overlap_honesty_teff_not_raw_t():
    # Same panel at daily (n=252, horizon=21) vs monthly (n=12, horizon=1) sampling has
    # the same *independent* observation count, so the same shrink factor. Raw T would
    # differ by 21×.
    t_daily = effective_sample_size(252, horizon=21, spacing=1.0)
    t_monthly = effective_sample_size(12, horizon=1, spacing=1.0)
    assert abs(t_daily - t_monthly) < 1e-9
    ic = 0.06
    assert abs(refine.level_shrink_factor(ic, t_daily) - refine.level_shrink_factor(ic, t_monthly)) < 1e-9
    # Independent rebalances (spacing ≥ horizon) are not deflated.
    assert effective_sample_size(12, horizon=5, spacing=21.0) == 12


def test_level_shrink_applied_exactly_once_in_pipeline():
    raw = {f"S{i}": float(i - 6) for i in range(1, 13)}
    panel = FeaturePanel.for_universe(AS_OF, list(raw))
    panel.set("score", raw)
    panel.set("residual_vol", {s: 0.20 for s in raw})
    refine_alpha(panel, AlphaContext(ic=0.05, level_shrink=(0.05, 60)))
    chain = panel.meta["shrink_chain"]
    ic_steps = [s for s in chain if s["step"] == "ic_uncertainty"]
    assert len(ic_steps) == 1
    assert abs(ic_steps[0]["multiplier"] - refine.level_shrink_factor(0.05, 60)) < 1e-12


def test_no_level_shrink_when_off():
    raw = {f"S{i}": float(i - 6) for i in range(1, 13)}
    panel = FeaturePanel.for_universe(AS_OF, list(raw))
    panel.set("score", raw)
    panel.set("residual_vol", {s: 0.20 for s in raw})
    refine_alpha(panel, AlphaContext(ic=0.05))  # level_shrink defaults None
    chain = panel.meta["shrink_chain"]
    assert [s for s in chain if s["step"] == "ic_uncertainty"] == []


# --------------------------------------------------------------------------- #
# 4. The equal-risk bucket diagnostic
# --------------------------------------------------------------------------- #
def _diag_book(n, vols, weights, seed=0):
    """A diagonal-Σ diagnostic on a synthetic book (independent names)."""
    symbols = [f"S{i}" for i in range(n)]
    sigma = np.diag(np.asarray(vols) ** 2)
    return risk_bucket_diagnostic(
        pd.Series(weights, index=symbols),
        sigma,
        symbols,
        pd.Series(vols, index=symbols),
    )


def test_correctly_scaled_book_passes():
    # Correct scaling: active weight ∝ z/σ, so w·σ (contribution to vol) is ~constant ⇒
    # variance contribution ~equal per name ⇒ no vol-bucket tilt.
    rng = np.random.default_rng(1)
    n = 60
    vols = np.linspace(0.15, 0.60, n)
    z = rng.normal(0, 1, n)
    weights = z / vols  # w ∝ z/σ ⇒ w·σ = z (equal risk per name)
    res = _diag_book(n, vols, weights)
    assert res["engaged"] and not res["tilt_detected"]


def test_case_misscaled_book_shows_monotone_tilt():
    # A Case-2-signal scaled as Case 1 multiplies in σ a second time: w ∝ z·σ, so the
    # per-name variance contribution ∝ σ⁴ — a strong monotone tilt into high-vol names.
    rng = np.random.default_rng(2)
    n = 60
    vols = np.linspace(0.15, 0.60, n)
    z = rng.normal(0, 1, n)
    weights = z * vols  # the double-counted-vol book
    res = _diag_book(n, vols, weights)
    assert res["engaged"] and res["tilt_detected"]
    assert res["variance_share_gradient"] > 0  # tilt toward the high-vol bucket
    assert "high-vol" in res["verdict"]


def test_bucket_diagnostic_degrades_and_suppresses_on_thin_universe():
    thin = _diag_book(10, [0.2] * 10, [0.1] * 10)  # < terciles threshold
    assert thin["engaged"] is False
    tercile = _diag_book(18, list(np.linspace(0.1, 0.5, 18)), list(np.random.default_rng(3).normal(0, 1, 18)))
    assert tercile["engaged"] and tercile["n_buckets"] == 3


# --------------------------------------------------------------------------- #
# 5. Service wiring + the Case-1 equivalence guard
# --------------------------------------------------------------------------- #
def _client(symbols, n=260):
    data = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    return MarketDataClient(DictMarketData(data)), data


def test_compute_alphas_case1_default_is_unchanged_equivalence_guard():
    # The default path must be byte-for-byte the base refinement pipeline (scaling defaults to
    # case1, no level shrink): the refactor changes nothing it shouldn't.
    symbols = [f"S{i:02d}" for i in range(12)]
    client, data = _client(symbols)
    as_of = data["S00"].index[-1].to_pydatetime()
    res = analysis.compute_alphas(client, "ma_crossover", symbols, as_of, benchmark="SPY")
    assert res["scaling"] == "case1"
    assert res["case"] is None  # no case test run on the default path
    # The scale step is Case 1, no ic_uncertainty step (assumed IC has none to shrink).
    assert res["shrink_chain"][0]["case"] == 1
    assert [s for s in res["shrink_chain"] if s["step"] == "ic_uncertainty"] == []


def test_compute_alphas_auto_scaling_reports_case_diagnostics():
    symbols = [f"S{i:02d}" for i in range(12)]
    client, data = _client(symbols)
    as_of = data["S00"].index[-1].to_pydatetime()
    res = analysis.compute_alphas(client, "ma_crossover", symbols, as_of, benchmark="SPY", scaling="auto")
    assert res["scaling"] in ("case1", "case2")
    assert res["case"] is not None
    assert "candidate_correlation" in res["case"]
    assert res["scaling"] == f"case{res['case']['case']}"


def test_compute_information_emits_shrink_chain_and_bucket_diagnostic():
    symbols = [f"S{i:02d}" for i in range(30)]
    client, data = _client(symbols, n=400)
    start = data["S00"].index[40].to_pydatetime()
    end = data["S00"].index[-1].to_pydatetime()
    res = analysis.compute_information(client, "ma_crossover", symbols, start, end, benchmark="SPY")
    assert "level_shrink_factor" in res and "effective_t" in res
    chain = res["shrink_chain"]
    assert len([s for s in chain if s["step"] == "ic_uncertainty"]) == 1
    # 0 ≤ shrink factor ≤ 1, and consistent with the reported IC and T_eff.
    assert 0.0 <= res["level_shrink_factor"] <= 1.0
    expected = refine.level_shrink_factor(res["recommended_ic"], res["effective_t"])
    assert abs(res["level_shrink_factor"] - expected) < 1e-9
    # The bucket diagnostic engages on a 30-name universe (terciles) and returns a verdict.
    assert res["risk_bucket_diagnostic"] is not None
    assert res["risk_bucket_diagnostic"]["engaged"] in (True, False)


def test_combined_path_owns_level_shrink_and_does_not_double_apply():
    symbols = [f"S{i:02d}" for i in range(10)]
    client, data = _client(symbols, n=400)
    as_of = data["S00"].index[-1].to_pydatetime()
    res = analysis.compute_combined_alphas(
        client, ["ma_crossover", "mean_reversion"], symbols, as_of, benchmark="SPY", horizon=5, n_points=10
    )
    chain = res["shrink_chain"]
    ic_steps = [s for s in chain if s["step"] == "ic_uncertainty"]
    assert len(ic_steps) == 1  # exactly one "is the IC real" step (combination owns it here)
    assert ic_steps[0]["owner"] == "combination_shrink"


def test_run_scaling_ab_compares_both_scalings():
    symbols = [f"S{i:02d}" for i in range(15)]
    client, data = _client(symbols, n=400)
    start = data["S00"].index[40].to_pydatetime()
    end = data["S00"].index[-1].to_pydatetime()
    res = analysis.run_scaling_ab(client, "ma_crossover", symbols, start, end, benchmark="SPY", n_points=12)
    assert "case1_realized_ir" in res and "case2_realized_ir" in res
    assert res["regression_pick"] in ("case1", "case2")
    assert res["ab_pick"] in ("case1", "case2")
    assert isinstance(res["agree"], bool)
