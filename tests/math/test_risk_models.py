"""Math contract: covariance estimation, shrinkage, the factor model and the EWMA layer.

The shrinkage reference is written in the test as explicit element loops from the
definition of the constant-correlation target and its optimal intensity; the factor
reference builds a noise-free world whose covariance is known in closed form.

Related existing tests: ``tests/test_factor_intercept.py``, ``tests/test_risk.py``,
``tests/test_risk_streaming.py``.

Defects this module guards:

* the factor model had no intercept factor, so a market-wide move landed in the
  (diagonal) residuals and a long-only book's risk was understated several-fold;
* the streaming and eager covariance paths drifting apart.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantred.data.compute import streaming_covariance
from quantred.risk.base import RiskMatrix, build_risk_matrix
from quantred.risk.conditional import (
    EWMA_LAMBDA_DAILY,
    EWMA_LAMBDA_WEEKLY,
    default_lambda,
    ewma_covariance_path,
    ewma_variance_path,
    ewma_variance_series,
    mincer_zarnowitz,
    qlike_loss,
)
from quantred.risk.factor import INTERCEPT, estimate_factor_model, with_intercept
from quantred.risk.sample import LedoitWolfCovariance, SampleCovariance


def _panel(t=120, n=4, seed=0, common=0.5):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, (t, 1))
    scale = rng.uniform(0.5, 1.5, n)
    return pd.DataFrame(
        common * market + rng.normal(0, 0.01, (t, n)) * scale, columns=[f"C{i}" for i in range(n)]
    )


# --------------------------------------------------------------------------- #
# Sample covariance and its annualisation
# --------------------------------------------------------------------------- #
def test_sample_covariance_is_the_population_covariance_by_loops():
    x = _panel(50, 3, 1).to_numpy()
    t, n = x.shape
    mu = [sum(x[k, i] for k in range(t)) / t for i in range(n)]
    ref = np.array(
        [
            [sum((x[k, i] - mu[i]) * (x[k, j] - mu[j]) for k in range(t)) / t for j in range(n)]
            for i in range(n)
        ]
    )
    sigma, shrink = SampleCovariance().estimate(pd.DataFrame(x))
    assert shrink is None
    assert sigma == pytest.approx(ref, rel=1e-12, abs=1e-18)


def test_build_risk_matrix_annualises_by_periods_per_year():
    panel = _panel(100, 3, 2)
    bars = {s: pd.DataFrame({"close": 100 * np.cumprod(1 + panel[s].to_numpy())}) for s in panel}
    rets = pd.DataFrame({s: bars[s]["close"].pct_change() for s in bars}).dropna().to_numpy()
    ref = np.cov(rets, rowvar=False, bias=True) * 252
    got = build_risk_matrix(SampleCovariance(), bars, 252.0, min_obs=60)
    assert got.sigma == pytest.approx(ref, rel=1e-10)
    weekly = build_risk_matrix(SampleCovariance(), bars, 52.0, min_obs=60)
    assert weekly.sigma == pytest.approx(ref * 52 / 252, rel=1e-10)


def test_streaming_covariance_is_chunking_invariant_and_matches_the_batch():
    x = _panel(333, 5, 3).to_numpy()
    ref = np.cov(x, rowvar=False, bias=True)
    for sizes in ([333], [1, 332], [100, 100, 100, 33], [7] * 47 + [4]):
        edges = np.cumsum([0, *sizes])
        chunks = [x[a:b] for a, b in zip(edges[:-1], edges[1:])]
        cov, n = streaming_covariance(iter(chunks))
        assert n == 333
        assert cov == pytest.approx(ref, rel=1e-10, abs=1e-15)


def test_streaming_covariance_degenerate_inputs():
    cov, n = streaming_covariance(iter([np.zeros((1, 3))]))
    assert n == 1 and not np.any(cov)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf constant-correlation shrinkage
# --------------------------------------------------------------------------- #
def _lw_reference(x):
    """Constant-correlation shrinkage written out element by element."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = (x.T @ x) / t
    sd = [math.sqrt(s[i, i]) for i in range(n)]
    r_bar = sum(s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else r_bar * sd[i] * sd[j] for j in range(n)] for i in range(n)])

    def pi(i, j):
        return sum((x[k, i] * x[k, j] - s[i, j]) ** 2 for k in range(t)) / t

    def theta(a, i, j):  # E[(x_a^2 - s_aa)(x_i x_j - s_ij)]
        return sum((x[k, a] ** 2 - s[a, a]) * (x[k, i] * x[k, j] - s[i, j]) for k in range(t)) / t

    pi_hat = sum(pi(i, j) for i in range(n) for j in range(n))
    rho_hat = sum(pi(i, i) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rho_hat += (r_bar / 2) * (
                    math.sqrt(s[j, j] / s[i, i]) * theta(i, i, j)
                    + math.sqrt(s[i, i] / s[j, j]) * theta(j, i, j)
                )
    gamma_hat = float(((f - s) ** 2).sum())
    raw = (pi_hat - rho_hat) / gamma_hat / t
    delta = max(0.0, min(1.0, raw))
    return delta * f + (1 - delta) * s, delta, raw


@pytest.mark.parametrize("seed, t, n", [(0, 60, 4), (1, 40, 6), (2, 200, 3)])
def test_ledoit_wolf_matches_the_element_loop_reference(seed, t, n):
    x = _panel(t, n, seed).to_numpy()
    ref, ref_delta, _ = _lw_reference(x)
    sigma, delta = LedoitWolfCovariance().estimate(pd.DataFrame(x))
    assert delta == pytest.approx(ref_delta, rel=1e-10, abs=1e-14)
    assert sigma == pytest.approx(ref, rel=1e-10, abs=1e-16)


@pytest.mark.parametrize("seed, t, n", [(1, 500, 4), (0, 1000, 3), (1, 2000, 6), (0, 300, 5)])
def test_ledoit_wolf_intensity_matches_the_reference_when_it_is_interior(seed, t, n):
    """The cases above all clamp the intensity to 1, so a wrong intensity formula passed:
    the output was the target whatever the estimate said. Here the raw intensity sits
    strictly inside (0, 1), so the estimator's own value is what the reference checks."""
    x = _panel(t, n, seed).to_numpy()
    ref, ref_delta, raw = _lw_reference(x)
    assert 0.05 < raw < 0.95  # not clamped: the intensity itself is exercised
    sigma, delta = LedoitWolfCovariance().estimate(pd.DataFrame(x))
    assert delta == pytest.approx(raw, rel=1e-10)
    assert sigma == pytest.approx(ref, rel=1e-10, abs=1e-16)


def test_ledoit_wolf_keeps_variances_stays_psd_and_shrinks_less_with_more_data():
    deltas = []
    for t in (30, 120, 2000):
        x = _panel(t, 6, 5).to_numpy()
        sigma, delta = LedoitWolfCovariance().estimate(pd.DataFrame(x))
        sample = np.cov(x, rowvar=False, bias=True)
        assert 0.0 <= delta <= 1.0
        assert np.diag(sigma) == pytest.approx(np.diag(sample), rel=1e-12)
        assert np.linalg.eigvalsh(sigma).min() > 0
        deltas.append(delta)
    assert deltas[0] > deltas[-1]


def test_ledoit_wolf_improves_conditioning_when_observations_are_scarce():
    x = _panel(20, 12, 6).to_numpy()
    sigma, _ = LedoitWolfCovariance().estimate(pd.DataFrame(x))
    assert np.linalg.cond(sigma) < np.linalg.cond(np.cov(x, rowvar=False, bias=True))


def test_ledoit_wolf_degenerate_inputs_return_the_sample_and_zero_shrinkage():
    one = pd.DataFrame(np.random.default_rng(7).normal(0, 0.01, (30, 1)))
    sigma, delta = LedoitWolfCovariance().estimate(one)
    assert delta == 0.0 and sigma[0, 0] == pytest.approx(one[0].var(ddof=0), rel=1e-12)
    dead = pd.DataFrame({"a": np.zeros(30), "b": np.random.default_rng(8).normal(0, 0.01, 30)})
    sigma, delta = LedoitWolfCovariance().estimate(dead)
    assert delta == 0.0 and np.all(np.isfinite(sigma))
    sigma, delta = LedoitWolfCovariance().estimate(pd.DataFrame(np.zeros((1, 3))))
    assert delta == 0.0 and not np.any(sigma)


# --------------------------------------------------------------------------- #
# Factor model
# --------------------------------------------------------------------------- #
def _exposures(n, k, seed):
    rng = np.random.default_rng(seed)
    names = [f"S{i:02d}" for i in range(n)]
    x = pd.DataFrame(rng.normal(size=(n, k)), index=names, columns=[f"f{i}" for i in range(k)])
    return (x - x.mean()) / x.std(ddof=0)


def test_with_intercept_prepends_a_ones_column_once():
    x = _exposures(5, 2, 0)
    out = with_intercept(x)
    assert list(out.columns) == [INTERCEPT, "f0", "f1"] and (out[INTERCEPT] == 1.0).all()
    assert with_intercept(out) is out


def test_noise_free_factor_world_is_recovered_exactly():
    """r_t = X f_t with no residual: the model must return F = cov(f), Delta = 0 and
    Sigma = X F X', which is also the annualised sample covariance of r."""
    n, k, t = 12, 2, 400
    x = with_intercept(_exposures(n, k, 1))
    f = np.random.default_rng(2).normal(0, 0.01, (t, k + 1)) * np.array([1.0, 0.7, 0.4])
    returns = pd.DataFrame(f @ x.to_numpy().T, columns=x.index)
    model = estimate_factor_model(returns, _exposures(n, k, 1), 252.0)
    assert model.factor_names == [INTERCEPT, "f0", "f1"]
    assert model.factor_cov == pytest.approx(np.cov(f, rowvar=False) * 252, rel=1e-8)
    assert np.max(model.specific_var) < 1e-12
    assert model.sigma == pytest.approx(np.cov(returns.to_numpy(), rowvar=False) * 252, rel=1e-8, abs=1e-12)


def test_factor_and_specific_variance_add_up_to_total_variance():
    n, k, t = 15, 3, 300
    rng = np.random.default_rng(3)
    x = _exposures(n, k, 4)
    returns = pd.DataFrame(rng.normal(0, 0.01, (t, n)) + rng.normal(0, 0.01, (t, 1)), columns=x.index)
    model = estimate_factor_model(returns, x, 252.0)
    w = rng.normal(size=n)
    assert model.factor_variance(w) + model.specific_variance(w) == pytest.approx(
        model.variance(w), rel=1e-10
    )
    assert np.diag(model.sigma) == pytest.approx(
        np.diag(model.exposures.to_numpy() @ model.factor_cov @ model.exposures.to_numpy().T)
        + model.specific_var,
        rel=1e-12,
    )


def test_without_the_intercept_a_market_move_would_vanish_from_the_correlation():
    """One common factor + idiosyncratic noise, z-scored styles unrelated to returns. The
    modelled mean pairwise correlation must follow the sample's (about 0.5); with no
    constant factor it was about zero."""
    n, t = 30, 3000
    rng = np.random.default_rng(5)
    x = _exposures(n, 2, 6)
    returns = pd.DataFrame(rng.normal(0, 0.01, (t, 1)) + rng.normal(0, 0.01, (t, n)), columns=x.index)

    def mean_corr(sigma):
        d = np.sqrt(np.diag(sigma))
        c = sigma / np.outer(d, d)
        return (c.sum() - len(c)) / (len(c) * (len(c) - 1))

    modelled = mean_corr(estimate_factor_model(returns, x, 252.0).sigma)
    sample = mean_corr(np.cov(returns.to_numpy(), rowvar=False))
    assert modelled == pytest.approx(sample, abs=0.03)
    assert modelled > 0.4  # a model with the common move in the residuals reads ~0


def test_factor_model_needs_two_names_and_two_periods():
    x = _exposures(3, 1, 7)
    assert estimate_factor_model(pd.DataFrame(np.zeros((1, 3)), columns=x.index), x, 252.0) is None
    assert estimate_factor_model(pd.DataFrame(np.zeros((5, 1)), columns=[x.index[0]]), x, 252.0) is None


# --------------------------------------------------------------------------- #
# RiskMatrix algebra
# --------------------------------------------------------------------------- #
def _risk():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(4, 4))
    return RiskMatrix(list("ABCD"), a @ a.T / 4 + np.eye(4) * 0.05)


def test_risk_matrix_variance_te_and_euler_decomposition():
    risk = _risk()
    w = np.array([0.4, 0.3, 0.2, 0.1])
    wb = np.array([0.25] * 4)
    var = sum(w[i] * risk.sigma[i, j] * w[j] for i in range(4) for j in range(4))
    assert risk.variance(w) == pytest.approx(var, rel=1e-12)
    assert risk.volatility(w) == pytest.approx(math.sqrt(var), rel=1e-12)
    active = w - wb
    te = math.sqrt(sum(active[i] * risk.sigma[i, j] * active[j] for i in range(4) for j in range(4)))
    assert risk.tracking_error(w, wb) == pytest.approx(te, rel=1e-12)
    # Euler: sum_i w_a,i * MCR_i == psi
    mcr = risk.marginal_contribution_to_risk(w, wb)
    assert float(active @ mcr.to_numpy()) == pytest.approx(te, rel=1e-12)


def test_implied_beta_of_the_benchmark_against_itself_is_one():
    risk = _risk()
    wb = np.array([0.1, 0.2, 0.3, 0.4])
    beta = risk.implied_beta(wb)
    assert float(wb @ beta.to_numpy()) == pytest.approx(1.0, rel=1e-12)
    assert not risk.implied_beta(np.zeros(4)).any()  # no benchmark risk: zeros, not NaN


def test_correlation_has_a_unit_diagonal_and_recovers_the_scaling():
    risk = _risk()
    corr = risk.correlation().to_numpy()
    assert np.diag(corr) == pytest.approx(1.0, abs=1e-14)
    vol = risk.volatilities().to_numpy()
    assert corr * np.outer(vol, vol) == pytest.approx(risk.sigma, rel=1e-12)


# --------------------------------------------------------------------------- #
# EWMA layer and forecast evaluation
# --------------------------------------------------------------------------- #
def test_default_lambda_follows_the_bar_cadence():
    assert default_lambda(252) == EWMA_LAMBDA_DAILY == 0.94
    assert default_lambda(52) == EWMA_LAMBDA_WEEKLY == 0.97
    assert default_lambda(19656) == 0.94


def test_ewma_variance_matches_the_recursion_by_loops():
    x = np.random.default_rng(9).normal(0, 0.01, (150, 2))
    lam, seed_n = 0.94, 40
    for col in range(2):
        v = sum(x[k, col] ** 2 for k in range(seed_n)) / seed_n
        for k in range(seed_n, 150):
            v = lam * v + (1 - lam) * x[k, col] ** 2
        got = ewma_variance_path(pd.DataFrame(x), lam, seed_n)
        assert got.iloc[col] == pytest.approx(v, rel=1e-12)


def test_ewma_is_as_of_causal_and_lambda_one_freezes_the_seed():
    x = np.random.default_rng(10).normal(0, 0.01, (120, 3))
    full = ewma_variance_series(x, 0.94, 30)
    truncated = ewma_variance_series(x[:80], 0.94, 30)
    assert np.isnan(full[:29]).all()
    assert full[:80] == pytest.approx(truncated, nan_ok=True)  # later rows cannot change earlier ones
    frozen = ewma_variance_series(x, 1.0, 30)
    assert frozen[-1] == pytest.approx(np.mean(x[:30] ** 2, axis=0), rel=1e-12)


def test_ewma_covariance_diagonal_agrees_with_the_variance_recursion():
    x = pd.DataFrame(np.random.default_rng(11).normal(0, 0.01, (100, 3)))
    cov = ewma_covariance_path(x, 0.94, 30)
    assert np.diag(cov) == pytest.approx(ewma_variance_path(x, 0.94, 30).to_numpy(), rel=1e-12)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() >= -1e-15
    assert ewma_covariance_path(x.iloc[:10], 0.94, 30) is None
    assert ewma_variance_path(x.iloc[:10], 0.94, 30) is None


def test_mincer_zarnowitz_recovers_an_exact_line_and_flags_short_samples():
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = mincer_zarnowitz(2.0 + 3.0 * f, f)
    assert (out["a"], out["b"], out["r2"]) == pytest.approx((2.0, 3.0, 1.0), abs=1e-12)
    short = mincer_zarnowitz(f[:2], f[:2])
    assert short["n"] == 2 and math.isnan(short["b"])  # unavailable, not a made-up slope


def test_qlike_known_answers():
    f = np.array([1.0, 2.0, 4.0])
    assert qlike_loss(f, f) == pytest.approx(0.0, abs=1e-15)
    # ratio 2 each: 2 - ln 2 - 1
    assert qlike_loss(2 * f, f) == pytest.approx(1 - math.log(2), rel=1e-12)
    assert math.isnan(qlike_loss(np.array([-1.0]), np.array([1.0])))  # nothing admissible


# --------------------------------------------------------------------------- #
# Streaming factor model carries the intercept factor, as the batch one does
# --------------------------------------------------------------------------- #
def test_streaming_factor_model_matches_batch_with_the_intercept_on_a_one_factor_world(tmp_path):
    """Defect guarded: the streaming estimator could drop the constant factor (only
    ``tests/test_risk_streaming.py`` noticed). The world has a large market-wide move
    every period plus one style factor with known loadings, so without an intercept the
    common move is pushed into the diagonal residuals and the factor covariance is
    wrong. The reference is a plain per-period least squares in numpy."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    from quantred.data import ParquetBarStore
    from quantred.risk import streaming_factor_risk_matrix

    rng = np.random.default_rng(21)
    t, n = 400, 8
    names = [f"S{i}" for i in range(n)]
    beta = np.linspace(-1.0, 1.0, n)
    market = rng.normal(0.0, 0.01, t)  # common to every name: the intercept factor's return
    style = rng.normal(0.0, 0.006, t)
    rets = market[:, None] + style[:, None] * beta[None, :] + rng.normal(0.0, 0.004, (t, n))
    idx = pd.date_range("2024-01-02 09:30", periods=t + 1, freq="1D", tz="America/New_York")
    bars = {}
    for j, name in enumerate(names):
        close = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets[:, j])])
        bars[name] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e5}, index=idx
        )
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = idx[-1].to_pydatetime()

    exposures = pd.DataFrame({"style": beta}, index=names)
    got = streaming_factor_risk_matrix(
        store, names, "1Day", as_of, exposures, lookback_days=10_000, periods_per_year=252.0, min_obs=60
    )
    assert got is not None and INTERCEPT in got.factor_names

    # independent reference: realised returns from the closes, f_t by lstsq, ddof=1 moments
    realised = np.array([bars[s]["close"].pct_change().dropna().to_numpy() for s in names]).T
    x = np.column_stack([np.ones(n), beta])
    f = np.array([np.linalg.lstsq(x, row, rcond=None)[0] for row in realised])
    resid = realised - f @ x.T
    f_cov = np.cov(f, rowvar=False) * 252
    spec = resid.var(axis=0, ddof=1) * 252
    ref_sigma = x @ f_cov @ x.T + np.diag(spec)
    assert got.sigma == pytest.approx(ref_sigma, rel=1e-8, abs=1e-12)
    assert got.specific_var == pytest.approx(spec, rel=1e-8)
    # the world is built so this matters: the market factor's variance is about 252 * 1e-4
    assert f_cov[0, 0] == pytest.approx(252 * 0.01**2, rel=0.3)


@pytest.mark.parametrize("n_names, expect_model", [(3, False), (4, False), (5, False), (6, False), (7, True)])
def test_factor_model_is_unavailable_when_names_leave_no_residual(n_names, expect_model):
    """With the intercept plus four style factors (K = 5), N <= 6 names leave fewer than two
    residual degrees of freedom: the regression fits every name (specific variance of order
    1e-31) and the book reads as 100% factor risk. Unavailable beats a confident fiction;
    N = K + 2 is the boundary that is accepted."""
    rng = np.random.default_rng(4)
    names = [f"S{i}" for i in range(n_names)]
    returns = pd.DataFrame(rng.normal(0, 0.01, (250, n_names)), columns=names)
    exposures = pd.DataFrame(rng.normal(size=(n_names, 4)), index=names, columns=list("abcd"))
    got = estimate_factor_model(returns, exposures, 252.0)
    assert (got is not None) is expect_model
    if got is not None:
        assert float(got.specific_var.min()) > 1e-6  # genuine residual, not float dust


# --------------------------------------------------------------------------- #
# Conditional risk: Sigma_t rebuilt from the documented formula
# --------------------------------------------------------------------------- #
def _ewma_final(x, lam, min_obs):
    """Documented recursion: seed with the raw mean square of the first ``min_obs`` rows,
    then ``v = lam * v + (1 - lam) * r**2`` row by row. Written as a plain loop."""
    v = [sum(x[i] ** 2 for i in range(min_obs)) / min_obs]
    for i in range(min_obs, len(x)):
        v.append(lam * v[-1] + (1 - lam) * x[i] ** 2)
    return v[-1]


def test_conditioned_intercept_factor_matrix_matches_the_documented_formula():
    """Intercept-only factor model: the factor return each bar is the cross-sectional mean,
    ``F_t`` is its EWMA variance (annualised), each name's specific variance is the EWMA of
    its squared deviation from that mean, and ``Sigma_t = F_t * 1 1' + diag(specific_t)``.
    Guards a conditioned matrix that conditions the factor and not the specifics (or the
    reverse), which mis-splits market from stock-specific risk."""
    from quantred.risk.conditional import condition_risk_matrix

    rng = np.random.default_rng(12)
    t, n, min_obs, lam, ppy = 300, 6, 60, 0.94, 252.0
    # a volatility regime shift in the last stretch, so conditioning moves the answer
    scale = np.where(np.arange(t) < 240, 1.0, 2.5)[:, None]
    names = [f"N{i}" for i in range(n)]
    data = (rng.normal(0.0, 0.01, (t, 1)) + rng.normal(0.0, 0.008, (t, n))) * scale
    returns = pd.DataFrame(data, columns=names)

    base = estimate_factor_model(returns, pd.DataFrame(index=names), ppy)
    got = condition_risk_matrix(base, returns, "ewma", ppy, min_obs=min_obs, lambda_=lam)

    mean_ret = [sum(row) / n for row in data.tolist()]
    resid = [[data[i][j] - mean_ret[i] for j in range(n)] for i in range(t)]
    f_var = _ewma_final(mean_ret, lam, min_obs) * ppy
    spec = [_ewma_final([resid[i][j] for i in range(t)], lam, min_obs) * ppy for j in range(n)]
    expected = [[f_var + (spec[i] if i == j else 0.0) for j in range(n)] for i in range(n)]

    assert got.factor_cov[0, 0] == pytest.approx(f_var, rel=1e-10)
    assert got.specific_var == pytest.approx(spec, rel=1e-10)
    assert got.sigma == pytest.approx(np.array(expected), rel=1e-10)
    # conditioning did something: the late regime is louder than the whole-window estimate
    assert f_var > 1.5 * base.factor_cov[0, 0]


def test_conditioned_shrinkage_matrix_holds_the_correlation_and_takes_the_ewma_diagonal():
    """``Sigma_t = D_t R D_t``: every entry is the base entry rescaled by the ratio of
    conditional to unconditional volatilities of its two names."""
    from quantred.risk.conditional import condition_risk_matrix

    rng = np.random.default_rng(13)
    t, n, min_obs, lam, ppy = 300, 5, 60, 0.97, 52.0
    names = [f"M{i}" for i in range(n)]
    data = rng.normal(0.0, 0.01, (t, 1)) + rng.normal(0.0, 0.01, (t, n)) * (1 + np.arange(n))
    returns = pd.DataFrame(data, columns=names)
    sigma, _ = SampleCovariance().estimate(returns)
    base = RiskMatrix(names, sigma * ppy)
    got = condition_risk_matrix(base, returns, "ewma", ppy, min_obs=min_obs, lambda_=lam)

    cond = [_ewma_final(data[:, j].tolist(), lam, min_obs) * ppy for j in range(n)]
    uncond = np.diag(base.sigma)
    expected = np.array(
        [
            [
                base.sigma[i, j] * math.sqrt(cond[i] * cond[j]) / math.sqrt(uncond[i] * uncond[j])
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    assert got.sigma == pytest.approx(expected, rel=1e-10)
    assert np.diag(got.sigma) == pytest.approx(cond, rel=1e-10)
    corr = lambda s: s / np.sqrt(np.outer(np.diag(s), np.diag(s)))  # noqa: E731
    assert corr(got.sigma) == pytest.approx(corr(base.sigma), abs=1e-12)  # the correlation is untouched
