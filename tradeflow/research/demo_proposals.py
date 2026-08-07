"""A curated, replayable proposal set for ``python main.py demo-agent``.

These are *real proposals in the real format* - the same :class:`Proposal` objects
an LLM emits - replayed deterministically so the demo produces the same narrative
every run, with no API key and no model-provider variance. Swap in a live model
with ``--provider anthropic`` and the loop is identical; only the source of the
proposals changes.

They are ordered to exercise one guardrail each, because the guardrails are the
product:

1. ``UNSAFE_CODE`` - reaches for the filesystem. Rejected by the sandbox's import
   allow-list before a single bar is loaded.
2. ``OVERPARAMETERIZED_CODE`` - a plausible mechanism with 8 knobs. Rejected by the
   contract check: more knobs = more overfit surface.
3. ``AGGRESSIVE_TUNE`` - the in-sample beauty queen: fast EMAs and a tight
   stop/target pair that prints a gorgeous backtest (Sharpe ~1.9 in-sample on a
   recent large-cap universe) and does not survive out-of-sample.
4. ``VOLATILITY_SCALED_CODE`` - clean, contract-valid, and economically sensible.
   It is admitted to walk-forward validation and *genuinely holds up* out-of-sample
   - and is refused anyway, on sample size and drawdown degradation. This is the
   most important beat: the engine refusing something that looks good, rather than
   something that was obviously noise all along.
"""

from tradeflow.research.proposer import Proposal

#: Reaches outside the sandbox. Rejected at import time by the allow-list.
UNSAFE_CODE = '''
import os
import pandas as pd
from tradeflow.strategies.base import Strategy


class ExfiltratingStrategy(Strategy):
    """Trend follower that also caches its universe to disk for later reuse."""

    TIMEFRAME = "1Day"
    PARAM_RANGES = {
        "ema_period": {"type": "int", "min": 5, "max": 40, "step": 1, "default": 20,
                       "description": "Trend EMA period"},
    }

    def initialize(self):
        os.makedirs("/tmp/strategy_cache", exist_ok=True)

    def process_data(self, data):
        return data

    def calculate_scores(self, data):
        return data["close"].pct_change()

    def calculate_required_lookback(self):
        return self.config["ema_period"] + 1
'''

#: A reasonable idea strangled by 8 tunable knobs. Rejected by the parameter cap.
OVERPARAMETERIZED_CODE = '''
import pandas as pd
from tradeflow.indicators import indicators
from tradeflow.strategies.base import Strategy


class MultiFactorConfluenceStrategy(Strategy):
    """Confluence of trend, momentum, and volatility filters, each independently tunable."""

    TIMEFRAME = "1Day"
    PARAM_RANGES = {
        "fast_ema": {"type": "int", "min": 3, "max": 15, "step": 1, "default": 8,
                     "description": "Fast EMA"},
        "slow_ema": {"type": "int", "min": 20, "max": 60, "step": 1, "default": 30,
                     "description": "Slow EMA"},
        "rsi_period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14,
                       "description": "RSI period"},
        "rsi_floor": {"type": "int", "min": 20, "max": 50, "step": 5, "default": 40,
                      "description": "RSI floor"},
        "atr_period": {"type": "int", "min": 7, "max": 21, "step": 1, "default": 14,
                       "description": "ATR period"},
        "atr_scale": {"type": "float", "min": 0.5, "max": 3.0, "step": 0.5, "default": 1.5,
                      "description": "ATR scaling"},
        "risk_per_trade": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.02,
                           "description": "Risk per trade"},
        "stop_loss": {"type": "float", "min": 0.01, "max": 0.08, "step": 0.01, "default": 0.03,
                      "description": "Stop distance"},
    }

    def initialize(self):
        pass

    def process_data(self, data):
        enriched = data.copy()
        enriched["fast"] = indicators.calculate_ema(data["close"], self.config["fast_ema"])
        enriched["slow"] = indicators.calculate_ema(data["close"], self.config["slow_ema"])
        enriched["rsi"] = indicators.calculate_rsi(data["close"], self.config["rsi_period"])
        return enriched

    def calculate_scores(self, data):
        gap = (data["fast"] - data["slow"]) / data["slow"]
        return gap.where(data["rsi"] > self.config["rsi_floor"], 0.0)

    def calculate_required_lookback(self):
        return max(self.config["slow_ema"], self.config["rsi_period"]) + 1
'''

#: Contract-valid and economically sensible. Admitted to walk-forward validation.
VOLATILITY_SCALED_CODE = '''
import numpy as np
import pandas as pd
from tradeflow.indicators import indicators
from tradeflow.strategies.base import Strategy


class VolatilityScaledTrendStrategy(Strategy):
    """Trend strength per unit of risk.

    A raw EMA gap treats a 2% move in a quiet name and a 2% move in a violent one as
    equal evidence, so position sizing ends up dominated by whichever symbol happens
    to be most volatile. Dividing the normalized gap by trailing realized volatility
    expresses conviction in risk-adjusted units, which should travel better
    out-of-sample: the ranking stops being a proxy for "which symbol moves most".
    """

    TIMEFRAME = "1Day"
    PARAM_RANGES = {
        "fast_ema_period": {"type": "int", "min": 5, "max": 20, "step": 1, "default": 10,
                            "description": "Fast EMA period"},
        "slow_ema_period": {"type": "int", "min": 21, "max": 60, "step": 1, "default": 30,
                            "description": "Slow EMA baseline"},
        "vol_window": {"type": "int", "min": 10, "max": 40, "step": 5, "default": 20,
                       "description": "Realized-volatility lookback"},
        "stop_loss": {"type": "float", "min": 0.01, "max": 0.06, "step": 0.01, "default": 0.03,
                      "description": "Stop-loss distance from entry (fraction)"},
        "take_profit": {"type": "float", "min": 0.02, "max": 0.12, "step": 0.02, "default": 0.06,
                        "description": "Take-profit distance from entry (fraction)"},
        # Fixed, not searched: spending the searchable budget on the mechanism rather
        # than on sizing keeps the overfit surface pointed at the actual hypothesis.
        "risk_per_trade": {"type": "float", "default": 0.02,
                           "description": "Capital fraction risked per trade"},
    }

    def __init__(self, config):
        config["timeframe"] = self.TIMEFRAME
        super().__init__(config)

    def initialize(self):
        if self.config["fast_ema_period"] >= self.config["slow_ema_period"]:
            raise ValueError("fast_ema_period must be < slow_ema_period")

    def process_data(self, data):
        if data.empty:
            return pd.DataFrame()
        enriched = data.copy()
        enriched["fast_ema"] = indicators.calculate_ema(data["close"], self.config["fast_ema_period"])
        enriched["slow_ema"] = indicators.calculate_ema(data["close"], self.config["slow_ema_period"])
        returns = data["close"].pct_change()
        enriched["realized_vol"] = returns.rolling(self.config["vol_window"]).std()
        return enriched

    def calculate_scores(self, data):
        if data.empty:
            return pd.Series(dtype=float)
        gap = (data["fast_ema"] - data["slow_ema"]) / data["slow_ema"]
        vol = data["realized_vol"].replace(0.0, np.nan)
        return (gap / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def calculate_required_lookback(self):
        return max(self.config["slow_ema_period"], self.config["vol_window"]) + 1
'''


#: The replay script. Each entry is exactly what a proposer returns.
DEMO_PROPOSALS = [
    Proposal(
        hypothesis=(
            "Trend signals decay when recomputed from scratch each run. Caching the processed "
            "universe to local disk lets the strategy reuse warm state across sessions, which "
            "should stabilize signals near session boundaries."
        ),
        kind="code",
        code=UNSAFE_CODE,
        tokens_used=612,
    ),
    Proposal(
        hypothesis=(
            "No single filter separates trend from chop reliably. Requiring confluence across "
            "trend (EMA gap), momentum (RSI floor), and volatility (ATR scaling) should suppress "
            "false entries in ranging markets while preserving genuine breakouts."
        ),
        kind="code",
        code=OVERPARAMETERIZED_CODE,
        tokens_used=1043,
    ),
    Proposal(
        hypothesis=(
            "The incumbent holds losers too long and lets winners round-trip. Faster EMAs enter "
            "trends earlier, and a tight stop paired with a modest take-profit harvests the move "
            "before mean reversion claws it back - raising both trade frequency and the win rate."
        ),
        kind="tune",
        strategy="ma_crossover",
        params={
            "fast_ema_period": 5,
            "slow_ema_period": 21,
            "stop_loss": 0.02,
            "take_profit": 0.04,
        },
        tuned_params=["fast_ema_period", "slow_ema_period", "stop_loss", "take_profit"],
        tokens_used=498,
    ),
    Proposal(
        hypothesis=(
            "A raw EMA gap conflates trend strength with volatility, so the cross-sectional "
            "ranking degenerates into a volatility ranking and the highest-conviction names are "
            "simply the noisiest. Scaling the normalized gap by trailing realized volatility "
            "expresses conviction in risk-adjusted units, which should generalize out-of-sample "
            "because the volatility normalization is estimated from data available at each bar."
        ),
        kind="code",
        code=VOLATILITY_SCALED_CODE,
        tokens_used=1387,
    ),
]
