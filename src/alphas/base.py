"""Core alpha types and the shared refinement pipeline.

An *alpha* is a forecast of **residual return** (return in excess of what beta to
the benchmark explains), expressed in the same units - annualised return - for
every name, so it is directly comparable across symbols and directly consumable
by a mean-variance optimiser.

The flow is two stages:

1. An :class:`AlphaModel` subclass turns some per-name view (a strategy's discrete
   signal, a scanner's continuous metric, ...) into a list of :class:`RawScore` -
   arbitrary-scale convictions, one per symbol, as of one rebalance timestamp.
2. :meth:`AlphaModel.alphas` runs the **same** refinement pipeline for every model
   (winsorize -> z-score -> optional neutralize -> scale -> cap) and returns
   :class:`Alpha` forecasts. The pipeline is shared so the scaling identity and
   the as-of/thin-universe discipline live in exactly one place.

Everything here is research-clock only: it forecasts, it never reads a realised
forward return, and it places no orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.alphas import refine

#: Conservative default information coefficient. The absolute scale of alphas is
#: only as good as this prior until IC is measured from realised outcomes; the
#: *relative* sizing across names is correct regardless (IC is a common scalar).
DEFAULT_IC = 0.03

#: Below this many names the cross-sectional z-score and winsorize quantiles are
#: unstable, so the pipeline falls back to demean-only and flags low confidence.
DEFAULT_MIN_UNIVERSE = 10


@dataclass
class RawScore:
    """One strategy/signal's pre-refinement view of a name, on any scale."""

    symbol: str
    score: float  # arbitrary units; sign = direction, magnitude = conviction
    as_of: datetime


@dataclass
class Alpha:
    """A refined residual-return forecast: annualised, benchmark-relative."""

    symbol: str
    alpha: float  # expected annualised residual return, e.g. 0.04 = +4%/yr
    as_of: datetime
    residual_vol: float  # sigma used in scaling (annualised), for audit
    ic: float  # information coefficient used in scaling
    raw_z: float  # standardised score that produced it


@dataclass
class AlphaContext:
    """Inputs to the refinement pipeline that aren't the raw scores themselves.

    ``residual_vol`` maps symbol -> annualised residual volatility (from a risk
    model when present, otherwise a trailing realised estimate). ``exposures``
    is an optional per-symbol exposure map used only when ``neutralize`` is set
    (e.g. ``{"NVDA": {"beta": 1.3}}`` for benchmark-beta neutralisation).
    """

    residual_vol: Dict[str, float]
    ic: float = DEFAULT_IC
    default_residual_vol: float = 0.20  # fallback when a name has no estimate
    neutralize: bool = False
    exposures: Optional[Dict[str, Dict[str, float]]] = None
    winsorize_limits: tuple = (0.025, 0.975)
    alpha_cap_std: float = 3.0
    min_universe: int = DEFAULT_MIN_UNIVERSE
    #: Set by :meth:`AlphaModel.alphas` to True when the thin-universe path is taken.
    low_confidence: bool = field(default=False, compare=False)


class AlphaModel(ABC):
    """Turns per-name views into scaled, comparable residual-return forecasts.

    Subclasses implement :meth:`raw_scores` (where the conviction comes from);
    :meth:`alphas` is the one shared method that refines them.
    """

    @abstractmethod
    def raw_scores(self, bars: Dict[str, pd.DataFrame], as_of: datetime) -> List[RawScore]:
        """Produce one :class:`RawScore` per scorable name, using data <= ``as_of``.

        ``bars`` maps symbol -> OHLCV frame already sliced to ``as_of`` by the
        caller; implementations must not look past ``as_of`` (no look-ahead).
        """

    def alphas(self, scores: List[RawScore], context: AlphaContext) -> List[Alpha]:
        """Refine raw scores into alphas via the shared pipeline.

        Cross-sectional at this one rebalance: winsorize, z-score, optionally
        neutralize against exposures, scale by ``sigma * IC``, then cap. On a thin
        universe (``< context.min_universe`` names) winsorize and the unit-std
        scaling are skipped in favour of demean-only, and ``context.low_confidence``
        is set.
        """
        if not scores:
            context.low_confidence = False
            return []

        s = pd.Series({sc.symbol: sc.score for sc in scores})
        as_of_by_symbol = {sc.symbol: sc.as_of for sc in scores}

        thin = len(s) < context.min_universe
        context.low_confidence = thin

        # 1-2. Winsorize then standardise. On thin universes the quantiles and the
        #      cross-sectional std are unreliable, so demean-only (no scaling).
        if thin:
            z = refine.demean(s)
        else:
            z = refine.zscore(refine.winsorize(s, *context.winsorize_limits))

        # 3. Optional neutralisation against supplied exposures.
        if context.neutralize and not thin:
            exposures = self._exposure_frame(context.exposures, s.index)
            if exposures is not None:
                z = refine.neutralize(z, exposures)

        # 4. Scale to a residual-return forecast via alpha_i = sigma_i * IC * z_i.
        vol = pd.Series({sym: context.residual_vol.get(sym, context.default_residual_vol) for sym in z.index})
        alpha = refine.scale_to_alpha(z, vol, context.ic)

        # 5. Final sanity cap (meaningless on a thin, unscaled vector).
        if not thin:
            alpha = refine.cap(alpha, context.alpha_cap_std)

        return [
            Alpha(
                symbol=sym,
                alpha=float(alpha[sym]),
                as_of=as_of_by_symbol[sym],
                residual_vol=float(vol[sym]),
                ic=context.ic,
                raw_z=float(z[sym]),
            )
            for sym in z.index
        ]

    def compute(self, bars: Dict[str, pd.DataFrame], as_of: datetime, context: AlphaContext) -> List[Alpha]:
        """Convenience: score the names then refine them in one call."""
        return self.alphas(self.raw_scores(bars, as_of), context)

    @staticmethod
    def _exposure_frame(
        exposures: Optional[Dict[str, Dict[str, float]]], index: pd.Index
    ) -> Optional[pd.DataFrame]:
        """Build a symbol-indexed exposure DataFrame from the context map."""
        if not exposures:
            return None
        frame = pd.DataFrame.from_dict(exposures, orient="index")
        frame = frame.reindex(index)
        return frame if not frame.empty else None
