"""Risk model: the covariance matrix Σ and the quantities built on it.

Active management needs a structural answer to "how risky is this portfolio?" that
respects the fact that risk is not additive. This module estimates an annualized,
well-conditioned, invertible Σ over a universe (Ledoit–Wolf shrinkage in v1) and
exposes portfolio variance, tracking error, and marginal contribution to risk.
Research-clock only: Σ sizes conviction, it never reaches the order path.
"""

from tradeflow.risk.base import RiskMatrix, RiskModel, build_return_panel, build_risk_matrix
from tradeflow.risk.conditional import (
    condition_risk_matrix,
    default_lambda,
    evaluate_vol_forecasts,
    mincer_zarnowitz,
    qlike_loss,
    turnover_risk_share,
)
from tradeflow.risk.exposures import FACTOR_NAMES, build_factor_exposures
from tradeflow.risk.factor import FactorRiskMatrix, build_factor_risk_matrix, estimate_factor_model
from tradeflow.risk.sample import LedoitWolfCovariance, SampleCovariance
from tradeflow.risk.streaming import streaming_factor_risk_matrix, streaming_sample_covariance

#: Conditional-vol backends accepted by ``build_risk_matrix``/``build_factor_risk_matrix``.
CONDITIONAL_VOL_MODELS = ("ewma", "har")

#: Statistical estimators by name (the `RiskModel.estimate` interface).
RISK_MODELS = {
    "shrinkage": LedoitWolfCovariance,
    "sample": SampleCovariance,
}

#: Every covariance model name the surfaces accept ("factor" is structural, built
#: from exposures rather than a plain estimator).
COVARIANCE_MODELS = (*RISK_MODELS, "factor")

__all__ = [
    "RiskMatrix",
    "RiskModel",
    "build_return_panel",
    "build_risk_matrix",
    "LedoitWolfCovariance",
    "SampleCovariance",
    "RISK_MODELS",
    "COVARIANCE_MODELS",
    "FACTOR_NAMES",
    "build_factor_exposures",
    "FactorRiskMatrix",
    "estimate_factor_model",
    "build_factor_risk_matrix",
    "streaming_sample_covariance",
    "streaming_factor_risk_matrix",
    "condition_risk_matrix",
    "default_lambda",
    "evaluate_vol_forecasts",
    "mincer_zarnowitz",
    "qlike_loss",
    "turnover_risk_share",
    "CONDITIONAL_VOL_MODELS",
]
