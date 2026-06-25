"""Tests for the continuous-alpha module.

Offline and deterministic. Covers the refinement identity, cross-sectional
z-scoring, neutralisation orthogonality, the signal->score bridge, the
as-of/leakage discipline, and the thin-universe fallback.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.alphas import refine
from src.alphas.base import DEFAULT_MIN_UNIVERSE, Alpha, AlphaContext, RawScore
from src.alphas.from_score import ScoreAlphaModel
from src.alphas.from_signal import SignalAlphaModel
from src.marketdata.client import MarketDataClient
from src.services import analysis
from src.strategies import signals
from src.strategies.ma_crossover import MovingAverageCrossoverStrategy
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)


def _scores(values: dict, as_of: datetime = AS_OF) -> list:
    return [RawScore(symbol=s, score=v, as_of=as_of) for s, v in values.items()]


def _pipeline_model() -> ScoreAlphaModel:
    """A model instance whose only job here is to expose the shared .alphas()."""
    return ScoreAlphaModel(lambda frame: 0.0)


# --- the refinement identity -------------------------------------------------
def test_scale_is_exact_sigma_ic_z():
    z = pd.Series({"A": 1.5, "B": -0.5, "C": 0.0})
    sigma = pd.Series({"A": 0.20, "B": 0.30, "C": 0.10})
    ic = 0.04
    alpha = refine.scale_to_alpha(z, sigma, ic)
    for sym in z.index:
        assert alpha[sym] == sigma[sym] * ic * z[sym]


def test_pipeline_applies_identity_when_not_thin():
    # >= floor names so the unit-std path (not demean) is taken.
    raw = {f"S{i}": float(i) for i in range(DEFAULT_MIN_UNIVERSE + 2)}
    vol = {s: 0.10 + 0.01 * i for i, s in enumerate(raw)}
    ctx = AlphaContext(residual_vol=vol, ic=0.05)
    alphas = _pipeline_model().alphas(_scores(raw), ctx)

    assert not ctx.low_confidence
    for a in alphas:
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
    z_affine = refine.zscore(s * 7.0 + 100.0)  # positive scale + shift
    # A positive affine transform leaves the standardised scores unchanged.
    assert np.allclose(z.values, z_affine.values)
    assert list(z.sort_values().index) == list(z_affine.sort_values().index)


def test_zscore_degenerate_universe_is_all_zero():
    s = pd.Series({"A": 4.0, "B": 4.0, "C": 4.0})
    z = refine.zscore(s)
    assert (z == 0.0).all()


# --- neutralisation ----------------------------------------------------------
def test_neutralize_removes_mean_and_exposure():
    z = pd.Series({"A": 1.2, "B": -0.7, "C": 0.4, "D": -0.9, "E": 1.0})
    betas = pd.DataFrame({"beta": [1.3, 0.8, 1.1, 0.5, 1.6]}, index=list("ABCDE"))
    resid = refine.neutralize(z, betas)

    # Intercept in the regression => residual is mean-zero (benchmark-neutral).
    assert abs(resid.mean()) < 1e-9
    # Residual is orthogonal to the beta exposure by construction.
    assert abs(float(np.dot(resid.values, betas["beta"].values))) < 1e-9


# --- the signal -> score bridge (hidden factor 4) ----------------------------
def test_signal_alpha_score_matches_strategy_direction():
    """A migrated strategy's score sign matches its discrete signal direction."""
    # A down-then-up close series guarantees a golden cross (BUY) somewhere.
    close = np.concatenate([np.linspace(100, 80, 40), np.linspace(80, 130, 40)])
    frame = make_ohlcv(n=len(close), seed=3, freq="1D")
    frame["close"] = close

    strat = MovingAverageCrossoverStrategy.create_with_defaults()
    sig = pd.Series(strat.generate_signals(strat.process_data(frame)))
    buy_bars = sig.index[sig == signals.BUY]
    assert len(buy_bars) > 0, "fixture should produce at least one BUY"

    # Slice to a bar the strategy marks BUY; the model must score it +1.
    sliced = frame.loc[frame.index <= buy_bars[0]]
    model = SignalAlphaModel(strat)
    scores = model.raw_scores({"X": sliced}, AS_OF)
    assert scores[0].score == 1.0


def test_signal_alpha_maps_each_signal():
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
        score = SignalAlphaModel(strat).raw_scores({"X": frame}, AS_OF)[0].score
        assert score == expected


# --- leakage discipline (hidden factor 5) ------------------------------------
def test_alphas_are_independent_of_post_as_of_bars():
    symbols = ["AAA", "BBB", "CCC"]
    bench = "SPY"
    full = {s: make_ohlcv(n=250, seed=i, freq="1D") for i, s in enumerate([*symbols, bench])}
    as_of = full["AAA"].index[150].to_pydatetime()

    # Version A: only bars up to as_of exist. Version B: the full series (future bars).
    truncated = {s: f.loc[f.index <= full["AAA"].index[150]] for s, f in full.items()}

    res_truncated = analysis.compute_alphas(
        MarketDataClient(DictMarketData(truncated)), "ma_crossover", symbols, as_of, benchmark=bench
    )
    res_full = analysis.compute_alphas(
        MarketDataClient(DictMarketData(full)), "ma_crossover", symbols, as_of, benchmark=bench
    )
    # Byte-identical alpha table regardless of whether future bars were present.
    assert res_truncated["alphas"] == res_full["alphas"]
    assert res_truncated["low_confidence"] == res_full["low_confidence"]


# --- thin universe (hidden factor 3) -----------------------------------------
def test_thin_universe_uses_demean_and_flags_low_confidence():
    raw = {"A": 1.0, "B": 5.0, "C": -2.0}  # < DEFAULT_MIN_UNIVERSE
    vol = {s: 0.20 for s in raw}
    ctx = AlphaContext(residual_vol=vol, ic=0.05)
    alphas = {a.symbol: a for a in _pipeline_model().alphas(_scores(raw), ctx)}

    assert ctx.low_confidence is True
    # Demean-only: raw_z is the centred score (NOT divided by the cross-sectional std).
    mean = np.mean(list(raw.values()))
    for sym, val in raw.items():
        assert abs(alphas[sym].raw_z - (val - mean)) < 1e-12


def test_full_universe_is_not_low_confidence():
    raw = {f"S{i}": float(i) for i in range(DEFAULT_MIN_UNIVERSE)}
    ctx = AlphaContext(residual_vol={s: 0.2 for s in raw}, ic=0.03)
    _pipeline_model().alphas(_scores(raw), ctx)
    assert ctx.low_confidence is False


# --- empty input -------------------------------------------------------------
def test_empty_scores_return_empty():
    ctx = AlphaContext(residual_vol={}, ic=0.03)
    assert _pipeline_model().alphas([], ctx) == []
    assert isinstance(Alpha("X", 0.0, AS_OF, 0.1, 0.03, 0.0), Alpha)
