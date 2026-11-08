"""The factor risk model carries a constant factor, so a market move is not "residual".

Style exposures are z-scored, so none can express "every name moved together". Without
a ones column that move landed in the residuals, whose covariance is diagonal: in a
one-factor world with 40 names the model's equal-weight volatility was about a quarter
of the truth and the mean pairwise correlation about zero.
"""

import numpy as np
import pandas as pd

from quantred.risk.exposures import FACTOR_NAMES
from quantred.risk.factor import estimate_factor_model

N, T = 40, 4000
MKT_VOL, IDIO_VOL, PPY = 0.01, 0.01, 252.0


def _world():
    rng = np.random.default_rng(11)
    names = [f"S{i:02d}" for i in range(N)]
    market = rng.normal(0, MKT_VOL, (T, 1))
    returns = pd.DataFrame(market + rng.normal(0, IDIO_VOL, (T, N)), columns=names)
    # Style exposures unrelated to the returns, z-scored like the real ones.
    x = pd.DataFrame(rng.normal(size=(N, 2)), index=names, columns=["momentum", "size"])
    x = (x - x.mean()) / x.std(ddof=0)
    return returns, x


def test_one_factor_world_equal_weight_vol_and_correlation_match_truth():
    returns, x = _world()
    m = estimate_factor_model(returns, x, PPY)

    true_var = (MKT_VOL**2 + IDIO_VOL**2 / N) * PPY
    w = np.full(N, 1.0 / N)
    vol = float(np.sqrt(w @ m.sigma @ w))
    assert abs(vol / np.sqrt(true_var) - 1.0) < 0.05

    d = np.sqrt(np.diag(m.sigma))
    corr = m.sigma / np.outer(d, d)
    mean_corr = (corr.sum() - N) / (N * (N - 1))
    true_corr = MKT_VOL**2 / (MKT_VOL**2 + IDIO_VOL**2)  # 0.5
    assert abs(mean_corr - true_corr) < 0.03


def test_specific_variance_is_not_inflated_by_the_common_move():
    returns, x = _world()
    m = estimate_factor_model(returns, x, PPY)
    # Three fitted factors on 40 names absorb ~3/40 of the idiosyncratic noise; without
    # the constant factor the common move leaked in and doubled this ratio.
    ratio = float(m.specific_var.mean() / (IDIO_VOL**2 * PPY))
    assert 0.85 < ratio < 1.05


def test_the_constant_factor_is_labelled_and_style_factors_keep_their_names():
    returns, x = _world()
    m = estimate_factor_model(returns, x, PPY)
    assert m.factor_names == ["intercept", "momentum", "size"]
    assert (m.exposures["intercept"] == 1.0).all()
    assert m.factor_cov.shape == (3, 3)
    assert "intercept" not in FACTOR_NAMES  # the constant is the model's, not a user-selectable style
