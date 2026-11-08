"""Tests for information horizon / alpha decay."""

import math
from datetime import datetime

import numpy as np

from quantred.alphas.horizon import (
    blend_weights,
    fit_decay,
    frequency_ir_curve,
    peak_return_horizon,
    recommended_cadence,
)
from quantred.marketdata.client import MarketDataClient
from quantred.services import analysis
from tests.fakes import DictMarketData, make_ohlcv


# --- decay recovery ----------------------------------------------------------
def test_decay_is_recovered():
    ic0, delta = 0.08, 0.85
    profile = {n: ic0 * delta**n for n in range(0, 10)}
    fit = fit_decay(profile)
    assert abs(fit["delta"] - delta) < 1e-6
    assert abs(fit["half_life"] - (-0.6931 / __import__("math").log(delta))) < 0.05
    assert fit["r_squared"] > 0.999  # clean exponential → near-perfect fit


def test_peak_horizon_tracks_half_life():
    # Peak of the profile (1 - e^-x)/sqrt(x), found by brute-force grid search here (x in decay
    # time constants); half-lives = x / ln 2.
    xs = [i * 1e-5 for i in range(1, 400_000)]
    x_star = max(xs, key=lambda x: (1 - math.exp(-x)) / math.sqrt(x))
    assert abs(peak_return_horizon(4.0) - 4.0 * x_star / math.log(2)) < 1e-3


def test_fit_decay_handles_degenerate_profile():
    assert fit_decay({0: 0.05})["delta"] != fit_decay({0: 0.05})["delta"]  # NaN (too few points)


# --- decay confidence interval ------------------------------------------------
def test_fit_decay_ci_brackets_the_point_estimate():
    ic0, delta = 0.08, 0.85
    profile = {n: ic0 * delta**n for n in range(0, 10)}
    fit = fit_decay(profile)
    # A clean noiseless exponential still has finite-sample SE from np.polyfit's
    # residuals (near machine epsilon here) - the CI should be a tight bracket
    # around, not equal to, the point estimate.
    assert fit["half_life_lower"] <= fit["half_life"] <= fit["half_life_upper"]


def test_fit_decay_ci_widens_with_noise():
    rng = np.random.default_rng(0)
    ic0, delta = 0.08, 0.85
    clean = {n: ic0 * delta**n for n in range(0, 10)}
    noisy = {n: max(v * float(rng.normal(1.0, 0.4)), 1e-4) for n, v in clean.items()}
    clean_fit = fit_decay(clean)
    noisy_fit = fit_decay(noisy)
    clean_width = clean_fit["half_life_upper"] - clean_fit["half_life_lower"]
    noisy_width = noisy_fit["half_life_upper"] - noisy_fit["half_life_lower"]
    assert noisy_width > clean_width


def test_fit_decay_ci_is_nan_below_three_points():
    fit = fit_decay({0: 0.08, 1: 0.07})  # 2 points: slope defined, SE isn't (n-2=0)
    assert fit["decay_slope_se"] != fit["decay_slope_se"]  # NaN
    assert fit["half_life_lower"] == fit["half_life"] == fit["half_life_upper"]


# --- blend regimes -----------------------------------------------------------
def test_blend_closed_form_and_regimes():
    # Diversify (δ > ρ) → positive lag weight; hedge (δ < ρ) → negative; equal → zero.
    w_now, w_lag = blend_weights(0.8, 0.3)
    assert abs(w_now - (1 - 0.8 * 0.3) / ((1 + 0.8) * (1 - 0.3))) < 1e-12
    assert w_lag > 0
    assert blend_weights(0.3, 0.8)[1] < 0  # hedge
    assert abs(blend_weights(0.5, 0.5)[1]) < 1e-12  # latest-only


def test_frequency_curve_has_interior_optimum():
    # IC rises with lag but √(1/Δt) falls → the IR proxy peaks in the middle.
    profile = {1: 0.02, 2: 0.035, 3: 0.045, 5: 0.05, 10: 0.055}
    curve = frequency_ir_curve(profile)
    assert recommended_cadence(profile) == max(curve, key=curve.get)
    assert recommended_cadence(profile) not in (1, 10)  # interior


# --- service -----------------------------------------------------------------
def _client():
    syms = [f"S{i}" for i in range(10)]
    data = {s: make_ohlcv(n=600, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    return MarketDataClient(DictMarketData(data)), syms


def test_compute_horizon_structure_and_leakage():
    client, syms = _client()
    r = analysis.compute_horizon(
        client, "demo_trend", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), max_lag=8
    )
    assert r["ic_by_lag"]
    assert set(r["ic_by_lag"]) <= {str(n) for n in range(1, 9)}  # JSON-stringified keys
    assert r["recommended_cadence"] >= 1
    assert -2.0 <= r["blend_weight_now"] <= 3.0
    assert r["blend_regime"] in ("diversify", "hedge", "latest-only")
    # Net-of-cost guard: the blend is priced and recommended only when it pays off.
    assert isinstance(r["blend_recommended"], bool)
    assert r["blend_annual_cost"] >= 0.0
    if r["blend_regime"] != "diversify":
        assert r["blend_recommended"] is False  # only a diversifying blend can be worth cost


def test_compute_horizon_independent_of_post_end_bars():
    syms = [f"S{i}" for i in range(8)]
    full = {s: make_ohlcv(n=700, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    cutoff = full["S0"].index[600]
    end = cutoff.to_pydatetime()
    start = full["S0"].index[120].to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    a = analysis.compute_horizon(MarketDataClient(DictMarketData(truncated)), "demo_trend", syms, start, end)
    b = analysis.compute_horizon(MarketDataClient(DictMarketData(full)), "demo_trend", syms, start, end)
    assert a["ic_by_lag"] == b["ic_by_lag"]
    assert a["half_life"] == b["half_life"] or (
        a["half_life"] != a["half_life"] and b["half_life"] != b["half_life"]
    )


def _blend_ic(w_now: float, w_lag: float, gamma: float, rho: float) -> float:
    """IC of a blend under the two-signal model, in units of the current signal's IC."""
    var = w_now**2 + w_lag**2 + 2 * rho * w_now * w_lag
    return (w_now + gamma * w_lag) / var**0.5


def test_blend_weights_maximize_ic_by_brute_force():
    """A grid search over the weight on the current signal (lag = 1 - now) agrees with
    the closed form; the earlier formula was neither the maximizer nor close to it."""
    import numpy as np

    for gamma, rho in [(0.5, 0.8), (0.8, 0.3), (0.3, 0.8), (0.9, 0.5), (0.6, 0.1)]:
        grid = np.linspace(-6.0, 8.0, 280001)
        ics = [_blend_ic(w, 1 - w, gamma, rho) for w in grid[::20]]
        best_w = grid[::20][int(np.argmax(ics))]
        w_now, w_lag = blend_weights(gamma, rho)
        assert abs(w_now - best_w) < 1e-3, (gamma, rho)
        assert abs(w_now + w_lag - 1.0) < 1e-12


def test_blend_reviewer_case():
    w_now, w_lag = blend_weights(0.5, 0.8)
    assert (round(w_now, 9), round(w_lag, 9)) == (2.0, -1.0)
    assert _blend_ic(w_now, w_lag, 0.5, 0.8) > _blend_ic(1.0 / 0.75, 1 - 1.0 / 0.75, 0.5, 0.8)
