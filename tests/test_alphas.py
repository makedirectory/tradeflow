"""Tests for the continuous-alpha refinement and the scorers that feed it.

Offline and deterministic. Covers the refinement identity, cross-sectional
z-scoring, neutralisation orthogonality, the strategy/signal scorers, the
as-of/leakage discipline, and the thin-universe fallback.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.alphas import AlphaContext, panel_to_alphas, refine, refine_alpha, signal_scorer, strategy_scorer
from src.alphas.base import DEFAULT_MIN_UNIVERSE, Alpha
from src.data.panel import FeaturePanel
from src.marketdata.client import MarketDataClient
from src.services import analysis
from src.strategies import signals
from src.strategies.ma_crossover import MovingAverageCrossoverStrategy
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)


def _panel(scores: dict, vols: dict = None, betas: dict = None) -> FeaturePanel:
    panel = FeaturePanel.for_universe(AS_OF, list(scores))
    panel.set("score", scores)
    panel.set("residual_vol", vols or {s: 0.20 for s in scores})
    if betas:
        panel.set("beta", betas)
    return panel


# --- the refinement identity -------------------------------------------------
def test_scale_is_exact_sigma_ic_z():
    z = pd.Series({"A": 1.5, "B": -0.5, "C": 0.0})
    sigma = pd.Series({"A": 0.20, "B": 0.30, "C": 0.10})
    ic = 0.04
    alpha = refine.scale_to_alpha(z, sigma, ic)
    for sym in z.index:
        assert alpha[sym] == sigma[sym] * ic * z[sym]


def test_refine_applies_identity_when_not_thin():
    raw = {f"S{i}": float(i) for i in range(DEFAULT_MIN_UNIVERSE + 2)}
    vols = {s: 0.10 + 0.01 * i for i, s in enumerate(raw)}
    panel = _panel(raw, vols=vols)
    context = AlphaContext(ic=0.05)
    refine_alpha(panel, context)

    assert panel.meta["low_confidence"] is False
    for a in panel_to_alphas(panel, context):
        assert a.alpha == a.residual_vol * a.ic * a.raw_z


# --- z-score -----------------------------------------------------------------
def test_zscore_mean_zero_unit_std():
    s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 10.0, "E": -4.0})
    z = refine.zscore(s)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=0) - 1.0) < 1e-12


def test_zscore_affine_invariant_ranking():
    s = pd.Series({"A": 1.0, "B": 5.0, "C": -2.0, "D": 3.0})
    z = refine.zscore(s)
    z_affine = refine.zscore(s * 7.0 + 100.0)
    assert np.allclose(z.values, z_affine.values)
    assert list(z.sort_values().index) == list(z_affine.sort_values().index)


def test_zscore_degenerate_universe_is_all_zero():
    s = pd.Series({"A": 4.0, "B": 4.0, "C": 4.0})
    assert (refine.zscore(s) == 0.0).all()


# --- neutralisation ----------------------------------------------------------
def test_neutralize_removes_mean_and_exposure():
    z = pd.Series({"A": 1.2, "B": -0.7, "C": 0.4, "D": -0.9, "E": 1.0})
    betas = pd.DataFrame({"beta": [1.3, 0.8, 1.1, 0.5, 1.6]}, index=list("ABCDE"))
    resid = refine.neutralize(z, betas)
    assert abs(resid.mean()) < 1e-9
    assert abs(float(np.dot(resid.values, betas["beta"].values))) < 1e-9


def test_refine_neutralize_is_beta_orthogonal():
    raw = {f"S{i}": float(i % 5) - 2 for i in range(12)}
    betas = {f"S{i}": 1.0 + 0.1 * i for i in range(12)}
    panel = _panel(raw, betas=betas)
    refine_alpha(panel, AlphaContext(ic=0.04, neutralize=True))
    z = panel.get("z")
    b = panel.get("beta")
    assert abs(z.mean()) < 1e-9
    assert abs(float(np.dot(z.values, b.values))) < 1e-9


# --- scorers (the score-first migration) -------------------------------------
def test_strategy_scorer_matches_direction():
    """A migrated strategy's score sign and discrete signal agree at a BUY bar."""
    close = np.concatenate([np.linspace(100, 80, 40), np.linspace(80, 130, 40)])
    frame = make_ohlcv(n=len(close), seed=3, freq="1D")
    frame["close"] = close

    strat = MovingAverageCrossoverStrategy.create_with_defaults()
    sig = pd.Series(strat.generate_signals(strat.process_data(frame)))
    buy_bars = sig.index[sig == signals.BUY]
    assert len(buy_bars) > 0, "fixture should produce at least one BUY"

    sliced = frame.loc[frame.index <= buy_bars[0]]
    assert strategy_scorer(strat)(sliced) > 0  # continuous conviction is bullish
    assert signal_scorer(strat)(sliced) == 1.0  # discrete direction is BUY


def test_signal_scorer_maps_each_signal():
    class _FixedStrategy(MovingAverageCrossoverStrategy):
        SIGNAL = signals.HOLD

        def generate_signals(self, data):
            return {data.index[-1]: self.SIGNAL}

    frame = make_ohlcv(n=60, seed=1, freq="1D")
    for signal, expected in [
        (signals.BUY, 1.0),
        (signals.SELL, -1.0),
        (signals.HOLD, 0.0),
        (signals.CLOSE_BUY, 0.0),
    ]:
        strat = _FixedStrategy.create_with_defaults()
        strat.SIGNAL = signal
        assert signal_scorer(strat)(frame) == expected


# --- leakage discipline ------------------------------------------------------
def test_alphas_are_independent_of_post_as_of_bars():
    symbols = ["AAA", "BBB", "CCC"]
    bench = "SPY"
    full = {s: make_ohlcv(n=250, seed=i, freq="1D") for i, s in enumerate([*symbols, bench])}
    cutoff = full["AAA"].index[150]
    as_of = cutoff.to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    res_truncated = analysis.compute_alphas(
        MarketDataClient(DictMarketData(truncated)), "ma_crossover", symbols, as_of, benchmark=bench
    )
    res_full = analysis.compute_alphas(
        MarketDataClient(DictMarketData(full)), "ma_crossover", symbols, as_of, benchmark=bench
    )
    assert res_truncated["alphas"] == res_full["alphas"]
    assert res_truncated["low_confidence"] == res_full["low_confidence"]


# --- thin universe -----------------------------------------------------------
def test_thin_universe_uses_demean_and_flags_low_confidence():
    raw = {"A": 1.0, "B": 5.0, "C": -2.0}  # < DEFAULT_MIN_UNIVERSE
    panel = _panel(raw)
    refine_alpha(panel, AlphaContext(ic=0.05))

    assert panel.meta["low_confidence"] is True
    mean = np.mean(list(raw.values()))
    z = panel.get("z")
    for sym, val in raw.items():
        assert abs(z[sym] - (val - mean)) < 1e-12


def test_full_universe_is_not_low_confidence():
    raw = {f"S{i}": float(i) for i in range(DEFAULT_MIN_UNIVERSE)}
    panel = _panel(raw)
    refine_alpha(panel, AlphaContext(ic=0.03))
    assert panel.meta["low_confidence"] is False


# --- empty input -------------------------------------------------------------
def test_empty_panel_returns_no_alphas():
    panel = FeaturePanel.for_universe(AS_OF, [])
    refine_alpha(panel, AlphaContext())
    assert panel_to_alphas(panel, AlphaContext()) == []
    assert isinstance(Alpha("X", 0.0, AS_OF, 0.1, 0.03, 0.0), Alpha)
