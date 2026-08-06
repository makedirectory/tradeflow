"""Cross-sectional data substrate: scan bars point-in-time, assemble a feature panel.

The single place data is gathered for cross-sectional research (alphas, risk,
portfolio construction, information analysis): :func:`~tradeflow.data.scan.BarSource`
scans a universe as of a timestamp (leakage-safe), and a :class:`~tradeflow.data.panel.FeaturePanel`
holds every name's features in one table that producers fill and consumers read.
"""

from tradeflow.data.features import Scorer, add_factor_exposure_features, add_risk_features, add_score_feature
from tradeflow.data.panel import FeaturePanel
from tradeflow.data.scan import BarSource, ClientBarSource, slice_to_as_of
from tradeflow.data.store import BAR_COLUMNS, ParquetBarStore

# Lazy out-of-core compute lives in tradeflow.data.compute / tradeflow.data.edges and
# is imported on demand — both need the optional ``store`` extra (polars/duckdb), so
# they are intentionally not pulled into the base-install import path here.

__all__ = [
    "BarSource",
    "ClientBarSource",
    "slice_to_as_of",
    "FeaturePanel",
    "Scorer",
    "add_factor_exposure_features",
    "add_risk_features",
    "add_score_feature",
    "ParquetBarStore",
    "BAR_COLUMNS",
]
