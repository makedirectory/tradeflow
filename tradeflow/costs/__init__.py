"""Transaction costs: price the cost of trading so research metrics are net, not gross.

A parametric model (commission + half-spread + square-root market impact) charged on
every simulated fill, and an alpha haircut + linear turnover term for the optimizer.
Research-clock only - the live path gets real fills from the broker.
"""

from tradeflow.costs.base import CostModel, Trade, TradeCost
from tradeflow.costs.parametric import ParametricCostModel

__all__ = ["CostModel", "Trade", "TradeCost", "ParametricCostModel"]
