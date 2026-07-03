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


def test_neutralize_multi_column_orthogonal_to_each():
    """One regression on the exposure union: residual orthogonal to every column."""
    rng = np.random.default_rng(7)
    names = [f"S{i}" for i in range(20)]
    z = pd.Series(rng.normal(size=20), index=names)
    exposures = pd.DataFrame(
        {"exp_volatility": rng.normal(size=20), "exp_size": rng.normal(size=20)}, index=names
    )
    resid = refine.neutralize(z, exposures)
    assert abs(resid.mean()) < 1e-9
    for col in exposures:
        assert abs(float(np.dot(resid.values, exposures[col].values))) < 1e-9


def test_refine_factor_neutral_is_orthogonal_to_exposure_columns():
    """neutralize_factors regresses out exp_<factor> columns; beta may stay exposed."""
    rng = np.random.default_rng(11)
    raw = {f"S{i}": float(i % 5) - 2 for i in range(12)}
    panel = _panel(raw, betas={f"S{i}": 1.0 + 0.1 * i for i in range(12)})
    panel.set("exp_volatility", {s: float(v) for s, v in zip(raw, rng.normal(size=12))})
    panel.set("exp_size", {s: float(v) for s, v in zip(raw, rng.normal(size=12))})

    refine_alpha(panel, AlphaContext(ic=0.04, neutralize_factors=("volatility", "size")))
    z = panel.get("z")
    for col in ("exp_volatility", "exp_size"):
        assert abs(float(np.dot(z.values, panel.get(col).reindex(z.index).values))) < 1e-9


def test_refine_market_factor_supersedes_plain_beta():
    """With 'market' in neutralize_factors, the standardized exposure column is used
    (plain beta is skipped) and the residual is orthogonal to it."""
    raw = {f"S{i}": float(i % 5) - 2 for i in range(12)}
    betas = {f"S{i}": 1.0 + 0.1 * i for i in range(12)}
    panel = _panel(raw, betas=betas)
    exp_market = refine.zscore(pd.Series(betas))
    panel.set("exp_market", exp_market)

    refine_alpha(panel, AlphaContext(ic=0.04, neutralize=True, neutralize_factors=("market",)))
    z = panel.get("z")
    assert abs(float(np.dot(z.values, exp_market.reindex(z.index).values))) < 1e-9
    # Standardized market exposure is affine in beta, so beta-orthogonality follows too.
    assert abs(float(np.dot(z.values, pd.Series(betas).reindex(z.index).values))) < 1e-9


def test_refine_names_missing_exposures_are_mean_imputed_not_dropped():
    """A name missing a factor value gets the cross-sectional mean (0), staying in
    the regression — it must not silently lose beta neutralisation (G&K union trap)."""
    raw = {f"S{i}": float(i % 5) - 2 for i in range(12)}
    betas = {f"S{i}": 1.0 + 0.1 * i for i in range(12)}
    panel = _panel(raw, betas=betas)
    covered = [s for s in list(raw) if s != "S0"]
    rng = np.random.default_rng(3)
    panel.set("exp_size", {s: float(v) for s, v in zip(covered, rng.normal(size=len(covered)))})

    refine_alpha(panel, AlphaContext(ic=0.04, neutralize=True, neutralize_factors=("size",)))
    z = panel.get("z")
    b = pd.Series(betas).reindex(z.index)
    imputed_size = panel.get("exp_size").reindex(z.index).fillna(0.0)
    # The WHOLE cross-section (S0 included) is beta- and size-orthogonal.
    assert abs(float(np.dot(z.values, b.values))) < 1e-9
    assert abs(float(np.dot(z.values, imputed_size.values))) < 1e-9
    assert panel.meta["neutralize_imputed"] == 1
    assert panel.meta["neutralized_against"] == ["size", "beta"]


def test_refine_unusable_exposures_fall_back_to_beta():
    """All-NaN or constant exposure columns must degrade to beta neutralisation —
    never to NO neutralisation (the silent-loss bug: requested market neutrality
    on a short-history universe previously disabled beta too)."""
    raw = {f"S{i}": float(i % 5) - 2 for i in range(12)}
    betas = {f"S{i}": 1.0 + 0.1 * i for i in range(12)}
    panel = _panel(raw, betas=betas)
    panel.set("exp_market", {s: np.nan for s in raw})  # e.g. from an empty exposure build
    panel.set("exp_volatility", {s: 1.0 for s in raw})  # constant: no cross-sectional info

    refine_alpha(
        panel,
        AlphaContext(ic=0.04, neutralize=True, neutralize_factors=("market", "volatility")),
    )
    z = panel.get("z")
    assert abs(float(np.dot(z.values, pd.Series(betas).reindex(z.index).values))) < 1e-9
    assert panel.meta["neutralized_against"] == ["beta"]


def test_producer_writes_nothing_on_short_history():
    """An exposure build qualifying <2 names writes no columns, so refinement can
    fall back to beta instead of regressing on all-NaN exposures."""
    from src.data.features import add_factor_exposure_features

    names = [f"S{i}" for i in range(12)]
    bars = {s: make_ohlcv(n=40, seed=i, freq="1D") for i, s in enumerate(names)}  # < 61 bars
    bench = make_ohlcv(n=40, seed=99, freq="1D")
    panel = _panel({s: float(i % 5) - 2 for i, s in enumerate(names)})
    add_factor_exposure_features(panel, bars, bench, ["market", "volatility", "size"])
    assert not panel.has("exp_market")
    assert not panel.has("exp_volatility")


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


def test_compute_alphas_factor_neutral_end_to_end():
    """The service threads neutralize_factors: exposures built, echoed, and applied."""
    symbols = [f"S{i:02d}" for i in range(12)]
    data = {s: make_ohlcv(n=250, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    client = MarketDataClient(DictMarketData(data))
    as_of = data["S00"].index[-1].to_pydatetime()

    plain = analysis.compute_alphas(client, "ma_crossover", symbols, as_of, benchmark="SPY")
    neutral = analysis.compute_alphas(
        client,
        "ma_crossover",
        symbols,
        as_of,
        benchmark="SPY",
        neutralize_factors=("volatility", "size"),
    )
    assert neutral["neutralize_factors"] == ["volatility", "size"]
    assert neutral["neutralized_against"] == ["volatility", "size"]
    assert neutral["alphas"] and plain["alphas"]
    # The factor regression must actually change the cross-section.
    z_plain = {r["symbol"]: r["z"] for r in plain["alphas"]}
    z_neutral = {r["symbol"]: r["z"] for r in neutral["alphas"]}
    assert any(abs(z_plain[s] - z_neutral[s]) > 1e-12 for s in z_plain)


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
