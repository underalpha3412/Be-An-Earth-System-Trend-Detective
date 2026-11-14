"""
Standardized Pydantic v2 schemas for model_go meteorological forecasting engine.
Enforces type safety, strict schema validation, and clean serialization.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    EXTREME_RED_ALERT = "EXTREME_RED_ALERT"


class DominantEngine(str, Enum):
    MODEL_ALPHA_GBDT = "MODEL_ALPHA_GBDT"
    HYBRID_PHYSICS_STUDENT = "HYBRID_PHYSICS_STUDENT"
    SUBSEASONAL_CLIMATE_HEAD = "SUBSEASONAL_CLIMATE_HEAD"
    IMD_4DVAR_NUDGED = "IMD_4DVAR_NUDGED"


class ForecastPointRequest(BaseModel):
    latitude: float = Field(default=28.6139, ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(default=77.2090, ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    station_name: Optional[str] = Field(default="Delhi (Safdarjung)", description="Optional station name")
    horizons_hours: List[int] = Field(default=[24, 48, 72, 168, 720], description="Forecast horizons in hours")
    include_synoptics: bool = Field(default=True, description="Whether to include synoptic fluid vectors")


class ForecastRegionalRequest(BaseModel):
    min_lat: float = Field(default=27.5, ge=-90.0, le=90.0)
    max_lat: float = Field(default=30.0, ge=-90.0, le=90.0)
    min_lon: float = Field(default=76.0, ge=-180.0, le=180.0)
    max_lon: float = Field(default=78.5, ge=-180.0, le=180.0)
    resolution_deg: float = Field(default=0.25, ge=0.05, le=2.5)
    horizons_hours: List[int] = Field(default=[24, 48, 72])


class HorizonForecast(BaseModel):
    lead_time_hours: int = Field(description="Lead time from t0 in hours")
    horizon_label: str = Field(description="Horizon key e.g. +24h, +48h, +7d")
    valid_time_utc: str = Field(default="", description="Target validity ISO-8601 UTC timestamp")
    valid_time_ist: str = Field(default="", description="Target validity ISO-8601 IST timestamp")
    temperature_c: float = Field(description="Predicted 2-meter air temperature (°C)")
    temp_uncertainty_90pct: Tuple[float, float] = Field(description="Calibrated 90% confidence interval [lower, upper] (°C)")
    msl_pressure_hpa: float = Field(description="Mean sea-level pressure (hPa)")
    wind_speed_ms: float = Field(description="10-meter wind speed (m/s)")
    relative_humidity_pct: float = Field(description="2-meter relative humidity (%)")
    precipitation_expected_mm: float = Field(description="Expected rainfall accumulation (mm)")
    rain_probability: float = Field(description="Two-stage hurdle probability of rain [0.0 - 1.0]")
    rain_hurdle_classification: str = Field(description="Hurdle rainfall classification")
    dominant_engine: DominantEngine = Field(description="Primary model engine routed for this horizon")


class SynopticContext(BaseModel):
    active_drivers: List[str] = Field(default_factory=list, description="Active meteorological drivers (e.g. Western Disturbance, Monsoon Trough)")
    vorticity_advection_vector: Tuple[float, float] = Field(default=(0.0, 0.0), description="Upper-air vorticity transport vector (u, v)")
    geostrophic_adjustment_status: str = Field(default="BALANCED", description="Geostrophic wind-pressure balance state")
    meridional_thermal_gradient_c_per_deg: float = Field(default=-0.04, description="North-south temperature gradient")


class UnifiedAlert(BaseModel):
    category: str = Field(description="Alert category: 'YAMUNA_HYDROLOGY', 'GLOBAL_HYDROLOGY', or 'IMD_STATUTORY'")
    level: AlertLevel = Field(description="Standardized alert severity level")
    title: str = Field(description="Short alert title")
    description: str = Field(description="Detailed meteorological/hydrological advisory")
    gauge_level_m: Optional[float] = Field(default=None, description="Current / projected River Yamuna gauge level at Old Railway Bridge")
    danger_mark_m: float = Field(default=205.33, description="Official CWC danger mark level")
    runoff_risk_index: Optional[float] = Field(default=None, description="0-100 Flash-Flood Risk Index for global coordinates")
    issued_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    issued_at_ist: str = Field(default="", description="IST issue timestamp string")


class ForecastMetadata(BaseModel):
    model_version: str = Field(default="2.0.0-headless")
    profile: str = Field(default="server")
    degraded_mode: bool = Field(default=False, description="True if operating in circuit-breaker fallback mode")
    adapter_status: str = Field(default="specialized", description="Adaptation tier: 'specialized', 'warming_up', or 'zero_shot'")
    conformal_coverage_guarantee: str = Field(default="90% (Split Conformal)", description="Conformal coverage guarantee")
    persistence_blend_weight: float = Field(default=0.0, description="Dynamic weight assigned to persistence blending")
    t0_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    t0_timestamp_ist: str = Field(default="", description="ISO-8601 Indian Standard Time string")
    t0_assimilation_sources: List[str] = Field(description="Active data feeds assimilated at t0")
    active_engines: List[str] = Field(description="Dispatched forecasting expert engines")
    inference_latency_ms: float = Field(description="End-to-end inference execution time in milliseconds")
    spending_incurred: str = Field(default="₹0.00")


class AdaptRequest(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    lookback_years: int = Field(default=3, ge=1, le=10, description="Historical training window in years")
    station_name: Optional[str] = Field(default=None, description="Optional city or station name")


class AdaptResponse(BaseModel):
    status: str = "SUCCESS"
    grid_key: str
    station_name: str
    adapter_status: str
    training_time_sec: float
    sample_count: int
    train_mae_c: float
    val_mae_c: float
    conformal_quantile_90_c: float
    adapter_size_bytes: int


class ForecastResponse(BaseModel):
    status: str = "SUCCESS"
    target_location: Dict[str, Any] = Field(description="Location metadata (lat, lon, station)")
    metadata: ForecastMetadata
    horizons: Dict[str, HorizonForecast]
    synoptic_context: SynopticContext
    alerts: List[UnifiedAlert]


class ConnectorHealth(BaseModel):
    name: str
    status: str  # "HEALTHY", "DEGRADED", "OFFLINE", "REQUIRES_AUTH"
    latency_ms: float
    last_sync_utc: Optional[datetime] = None
    notes: str


class SystemHealthResponse(BaseModel):
    status: str
    service: str = "model_go"
    version: str = "2.0.0"
    profile: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    connectors: Dict[str, ConnectorHealth]
    active_alerts_count: int
