"""
model_go: Unified Headless Production Meteorological Forecasting Engine.
Merges Model Alpha (GBDT), Hybrid Physics-ML, and IMD 4D-Var Assimilation.
"""

from model_go.sdk.client import Forecaster
from model_go.fusion.orchestrator import ModelGoOrchestrator
from model_go.schemas import (
    ForecastPointRequest,
    ForecastRegionalRequest,
    ForecastResponse,
    HorizonForecast,
    UnifiedAlert,
    AlertLevel
)

__version__ = "1.0.0"
__all__ = [
    "Forecaster",
    "ModelGoOrchestrator",
    "ForecastPointRequest",
    "ForecastRegionalRequest",
    "ForecastResponse",
    "HorizonForecast",
    "UnifiedAlert",
    "AlertLevel"
]
