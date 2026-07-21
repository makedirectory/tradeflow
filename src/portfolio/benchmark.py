"""The benchmark as a portfolio (Spec 017): loading ``w_B``, and reverse optimization.

Everywhere else in the stack a benchmark is a *return series* (e.g. SPY closes) -
betas and residual vols regress against it. This module is the other half: a
benchmark as a *holdings vector*, which is what makes "tracking error" the real
thing (``ψ = √(w_aᵀΣw_a)``, active weights ``w_a = w − w_B``) instead of the total
volatility of a cash-relative book. Pure, dependency-light functions - no data
fetching here (that's the caller's job); this module only turns a raw weights
mapping into something the optimizer and reverse-optimization report can use.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from src.risk.base import RiskMatrix


def load_benchmark_weights(source: str, symbols: Iterable[str]) -> Dict[str, float]:
    """Load ``w_B`` from ``"equal"`` (uniform over ``symbols``) or a holdings file.

    A file is CSV (``symbol,weight`` header + rows) or JSON (``{"symbol": weight}``).
    Cap-proxy weighting (shares outstanding) is deliberately not supported in v1 -
    the spec's own lean is that ``equal`` + a user-supplied file covers it (§7).

    Always renormalized to sum to 1: a benchmark that isn't fully invested (holds
    cash) is folded pro-rata into its named holdings rather than modeled as a
    genuine zero-variance asset - a v1 simplification (spec 017 hidden factor 6),
    not a hidden one - callers that care can compare against ``raw_weight_sum``
    before renormalization.
    """
    symbols = list(symbols)
    if source == "equal":
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    path = Path(source)
    if not path.exists():
        raise ValueError(f"benchmark holdings source {source!r} is neither 'equal' nor an existing file")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        weights = {str(k): float(v) for k, v in raw.items()}
    elif path.suffix.lower() == ".csv":
        weights = {}
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                weights[str(row["symbol"]).strip()] = float(row["weight"])
    else:
        raise ValueError(f"benchmark holdings file must be .csv or .json, got {path.suffix!r}")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"benchmark holdings file {source!r} has no positive weight")
    return {s: w / total for s, w in weights.items()}


def restrict_and_renormalize(
    weights: Dict[str, float], covered_symbols: Iterable[str]
) -> Tuple[Dict[str, float], float]:
    """Restrict ``w_B`` to names Σ actually covers, and renormalize to sum to 1.

    Reverse optimization and tracking error need Σ to span the benchmark (spec 017
    hidden factor 2); when it only spans the trading universe, the sanctioned v1
    alternative to extending the panel is this - restrict and renormalize, *loudly*
    (hidden factor 1): returns the coverage fraction (of raw weight mass) alongside
    the restricted weights, so a caller can warn when it's materially less than 1.
    """
    covered = set(covered_symbols)
    raw_total = sum(weights.values())
    kept = {s: w for s, w in weights.items() if s in covered}
    kept_total = sum(kept.values())
    coverage = (kept_total / raw_total) if raw_total > 0 else 0.0
    if kept_total <= 0:
        return {}, coverage
    return {s: w / kept_total for s, w in kept.items()}, coverage


def implied_returns(benchmark_weights: Dict[str, float], risk: RiskMatrix, mu_b: float) -> Dict[str, float]:
    """Reverse optimization (G&K ch. 2, eq. 2A.3): the consensus returns for which
    the benchmark ``w_B`` is itself mean-variance optimal.

    ``β = Σw_B/(w_Bᵀ Σ w_B)`` (the one canonical benchmark beta, spec 017 §4.3 -
    :meth:`RiskMatrix.implied_beta`), ``μ = β·μ_B`` for a stated benchmark premium
    ``μ_B``. Feeding ``μ`` back into :meth:`~src.portfolio.optimizer.MeanVarianceOptimizer.optimize`
    with this same ``w_B`` and zero cost returns ``w = w_B`` exactly - the sharpest
    integration test available, and the corollary this function exists to enable:
    our alphas are *deviations from* this consensus.
    """
    beta = risk.implied_beta(benchmark_weights)
    return {sym: float(beta[sym] * mu_b) for sym in risk.symbols}
