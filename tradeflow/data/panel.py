"""The cross-sectional feature panel - all the data in one place, as of one moment.

A :class:`FeaturePanel` is the single table every cross-sectional module reads from
and writes to: rows are symbols, columns are features (a signal's score, residual
volatility, beta, factor exposures, and - once refined - z and alpha). It is the
unit that makes the active-management stack cohesive: a strategy/scanner *produces*
a score column, the risk layer produces beta/residual_vol, the alpha layer consumes
those and produces z/alpha, and portfolio construction consumes the whole row.
Stacked over time these panels are the data a factor-importance or signal-combination
search runs over - "which factors matter right now" is a query against the panel.

Point-in-time by construction: a panel is built from a :mod:`tradeflow.data.scan` at one
``as_of``, so every column is leakage-safe. Today it wraps a pandas frame (small
universes); the same shape is what an Arrow/Polars columnar store would back at
scale, so the consumers above never learn where the data physically lives.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Union

import pandas as pd

Values = Union[pd.Series, Mapping[str, float]]


@dataclass
class FeaturePanel:
    """A symbol-indexed table of features for one universe at one ``as_of``."""

    as_of: datetime
    features: pd.DataFrame
    #: Cross-sectional flags set by producers/refiners (e.g. low_confidence).
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_universe(cls, as_of: datetime, symbols: List[str]) -> "FeaturePanel":
        """An empty panel whose rows are ``symbols`` (column-less until populated)."""
        index = pd.Index(list(dict.fromkeys(symbols)), name="symbol")
        return cls(as_of=as_of, features=pd.DataFrame(index=index))

    def set(self, name: str, values: Values) -> "FeaturePanel":
        """Add/replace a feature column, aligned to the panel's symbols."""
        series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
        self.features[name] = series.reindex(self.features.index)
        return self

    def get(self, name: str) -> pd.Series:
        """Return a feature column (raises ``KeyError`` if absent)."""
        return self.features[name]

    def has(self, name: str) -> bool:
        return name in self.features.columns

    @property
    def symbols(self) -> List[str]:
        return list(self.features.index)

    @property
    def columns(self) -> List[str]:
        return list(self.features.columns)
