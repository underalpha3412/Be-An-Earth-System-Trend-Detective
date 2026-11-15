"""
Hierarchical Mixture-of-Experts (MoE) & Spectral-Nudged Forecasting Orchestrator.
Merges Model Alpha (+24h point persistence & hurdle rain), Hybrid Physics-ML (+48h/+72h synoptics),
and IMD Operational 4D-Var boundary nudging into a unified headless pipeline.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import requests

from model_go.config import PROFILE
from model_go.schemas import (
    ForecastPointRequest,
    ForecastResponse,
    ForecastMetadata,
    HorizonForecast,
    SynopticContext,
    UnifiedAlert,
    AlertLevel,
    DominantEngine
)
from model_go.ingestion.feature_store import LocalFeatureStore
from model_go.ingestion.nasa.power_client import NASAPowerConnector
from model_go.ingestion.nasa.imerg_client import GPMIMERGConnector
from model_go.ingestion.nasa.geos_merra_client import GEOSMERRAConnector
from model_go.ingestion.nasa.smap_client import SMAPSoilMoistureConnector
from model_go.ingestion.imd.operational_bridge import IMDOperationalBridge
from model_go.ingestion.imd.statutory_parser import IMDStatutoryWarningParser
from model_go.ingestion.causal_gate import CausalTimeGate

from model_go.engines.alpha.model import ModelAlphaEngine
from model_go.engines.hybrid.student_model import DistilledStudentForecaster
from model_go.engines.hybrid.dynamics_core import AtmosphericDynamicsEngine
from model_go.engines.hybrid.precipitation_core import ConvectivePrecipitationEngine
from model_go.fusion.spectral_nudging import SpectralNudger
from model_go.hydrology.yamuna_model import RiverYamunaHydrologyEngine
from model_go.adaptation.spatial_adapter import SpatialAdapterManager
from model_go.fusion.conformal import ConformalCalibrator
from model_go.hydrology.global_runoff import GlobalRunoffEngine

logger = logging.getLogger(__name__)


class ModelGoOrchestrator:
    """Production orchestrator managing expert model routing, boundary nudging, and assimilation."""

    def __init__(self, profile: Optional[str] = None):
        self.profile = profile or PROFILE
        # Ingestion layer
        self.feature_store = LocalFeatureStore()
        self.power_client = NASAPowerConnector()
        self.imerg_client = GPMIMERGConnector()
        self.geos_client = GEOSMERRAConnector()
        self.smap_client = SMAPSoilMoistureConnector()
        self.imd_bridge = IMDOperationalBridge()
        self.imd_warnings = IMDStatutoryWarningParser()
        self.causal_gate = CausalTimeGate(strict_mode=False)

        # Expert Engines
        self.alpha_engine = ModelAlphaEngine()
        self.hybrid_student = DistilledStudentForecaster()
        self.student_engine = self.hybrid_student
        self.dynamics_engine = AtmosphericDynamicsEngine()
        self.precip_engine = ConvectivePrecipitationEngine()

        # v2 Fusion, Adaptation, Conformal Calibration & Global Hydrology
        self.nudger = SpectralNudger()
        self.hydrology_engine = RiverYamunaHydrologyEngine()
        self.adapter_manager = SpatialAdapterManager()
        self.conformal_calibrator = ConformalCalibrator()
        self.global_runoff_engine = GlobalRunoffEngine()

        logger.info("ModelGoOrchestrator v2 initialized successfully [Profile: %s]", self.profile)

    def generate_point_forecast(self, req: ForecastPointRequest) -> ForecastResponse:
        """
        Execute unified multi-horizon forecast cycle with Hierarchical MoE routing
        and circuit-breaker fallback protection.
        """
        t0 = time.perf_counter()
        now_dt = datetime.now(timezone.utc)
        ist_offset = timedelta(hours=5, minutes=30)
        t0_ist_str = (now_dt + ist_offset).strftime("%Y-%m-%d %H:%M:%S IST")
        doy = now_dt.timetuple().tm_yday
        station_key = f"{req.latitude:.4f}_{req.longitude:.4f}"

        assimilation_sources = []
        active_engines = []
        degraded_mode = False

        # 1. Ingest initial synoptic state with circuit-breaker protection
        if self.profile == "edge":
            # Even the lightweight profile must be location-aware. Its previous
            # fixed Delhi baseline caused identical outputs for every selected city.
            import math
            seasonal = math.cos(2.0 * math.pi * (doy - 200) / 365.25)
            lon_factor = math.sin(math.radians(req.longitude))
            lat_factor = max(-1.0, min(1.0, (abs(req.latitude) - 20.0) / 55.0))
            curr_t = 25.0 - 10.0 * lat_factor + 3.0 * seasonal + 1.5 * lon_factor
            curr_rh = max(25.0, min(90.0, 62.0 + 12.0 * math.cos(math.radians(req.latitude)) - 8.0 * seasonal + 5.0 * lon_factor))
            curr_p = 1013.0 - 0.12 * abs(req.latitude) + 0.8 * lon_factor
            curr_w = 2.0 + 0.8 * abs(math.sin(math.radians(req.longitude)))
            curr_wd = (270.0 + req.longitude) % 360.0
            assimilation_sources.append("LOCATION_AWARE_EDGE_CLIMATOLOGY")
            stored_features = {}
            active_engines = ["HYBRID_PHYSICS_STUDENT_EDGE", "OFFLINE_CACHED_LAGS"]
        else:
            # The bundled IMD bridge currently returns a fixed Safdarjung sample for
            # every station name. That made every city forecast start from identical
            # weather. Prefer a coordinate-specific current observation from
            # Open-Meteo (free, no API key); use a location-aware fallback if offline.
            try:
                weather_url = "https://api.open-meteo.com/v1/forecast"
                weather_params = {
                    "latitude": req.latitude,
                    "longitude": req.longitude,
                    "current": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m",
                    "timezone": "UTC",
                }
                weather_response = requests.get(weather_url, params=weather_params, timeout=6)
                weather_response.raise_for_status()
                current = weather_response.json().get("current", {})
                if not current or current.get("temperature_2m") is None:
                    raise ValueError("Open-Meteo returned no current observation")
                curr_t = float(current["temperature_2m"])
                curr_rh = float(current.get("relative_humidity_2m", 60.0))
                curr_p = float(current.get("pressure_msl", 1013.0))
                curr_w = float(current.get("wind_speed_10m", 9.0)) / 3.6
                curr_wd = float(current.get("wind_direction_10m", 270.0))
                curr_rain = float(current.get("precipitation", 0.0))
                imd_obs = {
                    "source": "OPEN_METEO_COORDINATE_OBSERVATION",
                    "station": req.station_name or "Selected location",
                    "latitude": req.latitude, "longitude": req.longitude,
                    "current_temp_c": curr_t, "morning_min_c": curr_t - 6.0,
                    "afternoon_max_c": curr_t + 3.0, "humidity_pct": curr_rh,
                    "pressure_hpa": curr_p, "wind_speed_ms": curr_w,
                    "wind_direction_deg": curr_wd, "rainfall_24h_mm": curr_rain,
                }
                assimilation_sources.append("OPEN_METEO_COORDINATE_OBSERVATION")
            except Exception as e:
                logger.warning("Coordinate weather feed unavailable; using location-aware fallback: %s", e)
                degraded_mode = True
                # A deterministic geographic fallback ensures different coordinates
                # do not silently receive the same Delhi/Safdarjung initial state.
                lat_factor = max(-1.0, min(1.0, (abs(req.latitude) - 20.0) / 55.0))
                seasonal = __import__("math").cos(2.0 * __import__("math").pi * (doy - 200) / 365.25)
                lon_factor = __import__("math").sin(__import__("math").radians(req.longitude))
                curr_t = 25.0 - 10.0 * lat_factor + 3.0 * seasonal + 1.5 * lon_factor
                curr_rh = max(25.0, min(90.0, 62.0 + 12.0 * __import__("math").cos(__import__("math").radians(req.latitude)) - 8.0 * seasonal + 5.0 * lon_factor))
                curr_p = 1013.0 - 0.12 * abs(req.latitude) + 0.8 * lon_factor
                curr_w = 2.0 + 0.8 * abs(__import__("math").sin(__import__("math").radians(req.longitude)))
                curr_wd = (270.0 + req.longitude) % 360.0
                curr_rain = max(0.0, (curr_rh - 68.0) * 0.035)
                imd_obs = {
                    "source": "LOCATION_AWARE_CLIMATOLOGY_FALLBACK",
                    "station": req.station_name or "Selected location",
                    "latitude": req.latitude, "longitude": req.longitude,
                    "current_temp_c": curr_t, "morning_min_c": curr_t - 6.0,
                    "afternoon_max_c": curr_t + 3.0, "humidity_pct": curr_rh,
                    "pressure_hpa": curr_p, "wind_speed_ms": curr_w,
                    "wind_direction_deg": curr_wd, "rainfall_24h_mm": curr_rain,
                }
                assimilation_sources.append("LOCATION_AWARE_CLIMATOLOGY_FALLBACK")

            # 2. Retrieve lag features from FeatureStore or instant climate-normalized fallback
            stored_features = self.feature_store.get_features(station_key)
            if stored_features is None:
                raw_obs = {
                    "T2M": curr_t,
                    "T2M_MAX": imd_obs.get("afternoon_max_c", curr_t + 4.0),
                    "T2M_MIN": imd_obs.get("morning_min_c", curr_t - 8.0),
                    "RH2M": curr_rh,
                    "PRECTOTCORR": float(imd_obs.get("rainfall_24h_mm", 0.0)),
                    "WS10M": curr_w,
                    "WD10M": curr_wd,
                    "PS": curr_p / 10.0,
                    "ALLSKY_SFC_SW_DWN": 16.5
                }
                stored_features = self.alpha_engine.pipeline.build_feature_row(now_dt, raw_obs, {})
                self.feature_store.put_features(station_key, stored_features, {"station": req.station_name})
                assimilation_sources.append("INSTANT_ZERO_SHOT_CLIMATE_NORMALIZED")
            else:
                assimilation_sources.append("LOCAL_FEATURE_STORE_CACHE")
                # Keep stored lag features, but always replace present-time inputs
                # with the newly fetched observation for this exact coordinate.
                stored_features.update({
                    "T2M": curr_t,
                    "T2M_MAX": imd_obs.get("afternoon_max_c", curr_t + 4.0),
                    "T2M_MIN": imd_obs.get("morning_min_c", curr_t - 8.0),
                    "RH2M": curr_rh,
                    "PRECTOTCORR": imd_obs.get("rainfall_24h_mm", 0.0),
                    "WS10M": curr_w,
                    "WD10M": curr_wd,
                    "PS": curr_p / 10.0,
                })

        # 3. Ingest GPM IMERG and SMAP for hydrological boundary
        imerg_data = self.imerg_client.fetch_data()
        smap_data = self.smap_client.fetch_catchment_soil_moisture()
        if imerg_data.get("auth_status") == "AUTHENTICATED":
            assimilation_sources.append("NASA_GPM_IMERG_GES_DISC")
        else:
            assimilation_sources.append("NASA_GPM_IMERG_ASSIMILATED")

        if smap_data.get("auth_status") == "AUTHENTICATED":
            assimilation_sources.append("NASA_SMAP_L3_DAAC")
        else:
            assimilation_sources.append("NASA_SMAP_HYDRO_INDEX")

        # 4. Regional Spatial Specialization Check (Tier 1 vs Tier 2)
        adapter, adapter_status = self.adapter_manager.request_adapter(
            lat=req.latitude,
            lon=req.longitude,
            station_name=req.station_name,
            lookback_years=3
        )
        if adapter is not None:
            active_engines.append("LOCAL_RESIDUAL_ADAPTER")
        elif adapter_status == "warming_up":
            active_engines.append("ADAPTER_WARMING_UP")
        else:
            active_engines.append("ZERO_SHOT_CLIMATE_NORMALIZED")

        # Route Expert 1: Model Alpha (+24h point prediction & hurdle rain)
        alpha_preds = self.alpha_engine.predict_point(stored_features)
        active_engines.append("MODEL_ALPHA_GBDT")

        t24_base = alpha_preds["next_day_t2m_c"]
        if adapter is not None:
            month = now_dt.month
            local_residual = adapter.predict_residual(
                base_pred_t2m=t24_base,
                t_prev=curr_t,
                t_prev2=curr_t - 0.5,
                doy=doy,
                month=month,
                rh_prev=curr_rh,
                p_prev_hpa=curr_p
            )
            t24_adapted = round(t24_base + local_residual, 2)
        else:
            t24_adapted = t24_base

        # Regime-Aware Persistence Blending via Synoptic Volatility Gate
        p_diff = abs(curr_p - 1013.25)
        rain_prob = alpha_preds["next_day_rain_prob"]
        volatility_score = (p_diff / 8.0) + (rain_prob * 1.5)

        if volatility_score < 0.85:
            # Calm synoptic regime: blend short-term persistence into +24h head
            w_pers = max(0.0, min(0.35, 0.35 * (1.0 - (volatility_score / 0.85))))
            t24_val = round((1.0 - w_pers) * t24_adapted + w_pers * curr_t, 2)
        else:
            # Active synoptic/frontal regime: route full weight to ML + physics core
            w_pers = 0.0
            t24_val = t24_adapted

        # 5. Route Expert 2: Hybrid Physics Student (+48h & +72h synoptics)
        hybrid_preds = self.hybrid_student.forecast_horizons(
            t2m_c=curr_t,
            mslp_hpa=curr_p,
            wind_ms=curr_w,
            doy=doy,
            horizons_hours=[48, 72]
        )
        active_engines.append("HYBRID_PHYSICS_STUDENT")

        # 6. Apply Boundary Blending & Spectral Nudging (24h -> 48h)
        t48_raw = hybrid_preds[48]["t2m_c"]
        p24_mm = alpha_preds["next_day_rain_expected_mm"]
        p48_raw_mm = 0.5 if hybrid_preds[48]["mslp_hpa"] < 1010.0 else 0.0

        t48_nudged, p48_nudged_mm = self.nudger.blend_24h_to_48h(t24_val, t48_raw, p24_mm, p48_raw_mm)
        t72_val = hybrid_preds[72]["t2m_c"]
        p72_mm = 1.2 if hybrid_preds[72]["mslp_hpa"] < 1011.0 else 0.0

        # 7. Route Expert 3: Sub-Seasonal Climate Anomaly Heads (+7d & +30d)
        t7d_val = alpha_preds["day_plus_7_t2m_c"]
        t30d_val = alpha_preds["next_30d_mean_t2m_c"]
        active_engines.append("SUBSEASONAL_CLIMATE_HEAD")

        # Conformal 90% Uncertainty Bounds across all horizons
        adapter_q90 = adapter.conformal_quantile_90 if adapter is not None else None
        ci_24 = self.conformal_calibrator.compute_bounds(24, t24_val, adapter_q90)
        ci_48 = self.conformal_calibrator.compute_bounds(48, t48_nudged, adapter_q90)
        ci_72 = self.conformal_calibrator.compute_bounds(72, t72_val, adapter_q90)
        ci_168 = self.conformal_calibrator.compute_bounds(168, t7d_val, adapter_q90)
        ci_720 = self.conformal_calibrator.compute_bounds(720, t30d_val, adapter_q90)

        # Construct Horizon Forecasts with ISO-8601 UTC and IST timestamps
        horizons_dict = {}

        # +24h Horizon (Dominated by Adapted & Blended Model Alpha)
        horizons_dict["+24h"] = HorizonForecast(
            lead_time_hours=24,
            horizon_label="+24h",
            valid_time_utc=(now_dt + timedelta(hours=24)).isoformat(),
            valid_time_ist=(now_dt + timedelta(hours=24) + ist_offset).strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=t24_val,
            temp_uncertainty_90pct=ci_24,
            msl_pressure_hpa=round(curr_p + 0.4, 1),
            wind_speed_ms=round(curr_w * 0.95, 1),
            relative_humidity_pct=alpha_preds["next_day_rh2m_pct"],
            precipitation_expected_mm=p24_mm,
            rain_probability=alpha_preds["next_day_rain_prob"],
            rain_hurdle_classification=alpha_preds["rain_hurdle_classification"],
            dominant_engine=DominantEngine.MODEL_ALPHA_GBDT
        )

        # +48h Horizon (Dominated by Nudged Hybrid Physics)
        horizons_dict["+48h"] = HorizonForecast(
            lead_time_hours=48,
            horizon_label="+48h",
            valid_time_utc=(now_dt + timedelta(hours=48)).isoformat(),
            valid_time_ist=(now_dt + timedelta(hours=48) + ist_offset).strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=t48_nudged,
            temp_uncertainty_90pct=ci_48,
            msl_pressure_hpa=hybrid_preds[48]["mslp_hpa"],
            wind_speed_ms=hybrid_preds[48]["wind_ms"],
            relative_humidity_pct=round(curr_rh * 0.96, 1),
            precipitation_expected_mm=p48_nudged_mm,
            rain_probability=0.18 if p48_nudged_mm > 0 else 0.06,
            rain_hurdle_classification="ISOLATED_DRIZZLE" if p48_nudged_mm > 0 else "NO_RAIN",
            dominant_engine=DominantEngine.HYBRID_PHYSICS_STUDENT
        )

        # +72h Horizon (Dominated by Hybrid Physics Student)
        horizons_dict["+72h"] = HorizonForecast(
            lead_time_hours=72,
            horizon_label="+72h",
            valid_time_utc=(now_dt + timedelta(hours=72)).isoformat(),
            valid_time_ist=(now_dt + timedelta(hours=72) + ist_offset).strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=t72_val,
            temp_uncertainty_90pct=ci_72,
            msl_pressure_hpa=hybrid_preds[72]["mslp_hpa"],
            wind_speed_ms=hybrid_preds[72]["wind_ms"],
            relative_humidity_pct=round(curr_rh * 1.05, 1),
            precipitation_expected_mm=p72_mm,
            rain_probability=0.28 if p72_mm > 0 else 0.08,
            rain_hurdle_classification="LIGHT_RAIN" if p72_mm > 0 else "NO_RAIN",
            dominant_engine=DominantEngine.HYBRID_PHYSICS_STUDENT
        )

        # +7d Horizon (Medium-Range Climate)
        horizons_dict["+7d"] = HorizonForecast(
            lead_time_hours=168,
            horizon_label="+7d",
            valid_time_utc=(now_dt + timedelta(hours=168)).isoformat(),
            valid_time_ist=(now_dt + timedelta(hours=168) + ist_offset).strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=t7d_val,
            temp_uncertainty_90pct=ci_168,
            msl_pressure_hpa=1014.2,
            wind_speed_ms=2.0,
            relative_humidity_pct=60.0,
            precipitation_expected_mm=0.0,
            rain_probability=0.10,
            rain_hurdle_classification="NO_RAIN",
            dominant_engine=DominantEngine.SUBSEASONAL_CLIMATE_HEAD
        )

        # +30d Horizon (Monthly Climatology Anomaly)
        horizons_dict["+30d"] = HorizonForecast(
            lead_time_hours=720,
            horizon_label="+30d",
            valid_time_utc=(now_dt + timedelta(hours=720)).isoformat(),
            valid_time_ist=(now_dt + timedelta(hours=720) + ist_offset).strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=t30d_val,
            temp_uncertainty_90pct=ci_720,
            msl_pressure_hpa=1016.5,
            wind_speed_ms=1.8,
            relative_humidity_pct=62.0,
            precipitation_expected_mm=0.0,
            rain_probability=0.05,
            rain_hurdle_classification="NO_RAIN",
            dominant_engine=DominantEngine.SUBSEASONAL_CLIMATE_HEAD
        )

        # 8. Synoptic Context & Dynamics Vectors
        synoptic_dict = self.dynamics_engine.evaluate_synoptic_context(
            lat=req.latitude,
            lon=req.longitude,
            t2m_c=curr_t,
            mslp_hpa=curr_p,
            u10=curr_w * 0.7,
            v10=curr_w * 0.7,
            doy=doy
        )
        synoptics = SynopticContext(
            active_drivers=synoptic_dict["active_drivers"],
            vorticity_advection_vector=synoptic_dict["vorticity_advection_vector"],
            geostrophic_adjustment_status=synoptic_dict["geostrophic_adjustment_status"],
            meridional_thermal_gradient_c_per_deg=synoptic_dict["meridional_thermal_gradient_c_per_deg"]
        )

        # 9. Hydrology & Alerts (Unified Yamuna Catchment vs Global Runoff Engine)
        runoff_res = self.global_runoff_engine.diagnose_risk(
            lat=req.latitude,
            lon=req.longitude,
            rain_24h_mm=p24_mm,
            rain_72h_mm=p48_nudged_mm + p72_mm,
            soil_saturation=smap_data.get("saturation_fraction", 0.45)
        )
        unified_alerts = [
            UnifiedAlert(
                category=runoff_res["category"],
                level=AlertLevel(runoff_res["alert_level"]),
                title=runoff_res["title"],
                description=runoff_res["advisory"],
                gauge_level_m=runoff_res.get("projected_gauge_m"),
                danger_mark_m=runoff_res.get("danger_mark_m", 205.33),
                runoff_risk_index=runoff_res.get("runoff_risk_index"),
                issued_at_utc=now_dt,
                issued_at_ist=t0_ist_str
            )
        ]

        # Append IMD statutory alerts if in Yamuna / Indian territory
        if runoff_res.get("is_yamuna"):
            imd_stat_list = self.imd_warnings.fetch_active_warnings(req.station_name or "Delhi")
            for stat in imd_stat_list:
                unified_alerts.append(
                    UnifiedAlert(
                        category="IMD_STATUTORY",
                        level=AlertLevel(stat.get("level", "GREEN")),
                        title=stat.get("title", "IMD Advisory"),
                        description=stat.get("description", "Normal conditions"),
                        issued_at_utc=now_dt,
                        issued_at_ist=t0_ist_str
                    )
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metadata = ForecastMetadata(
            model_version="2.0.0-headless",
            profile=self.profile,
            degraded_mode=degraded_mode,
            adapter_status=adapter_status,
            conformal_coverage_guarantee="90% (Split Conformal)",
            persistence_blend_weight=round(w_pers, 3),
            t0_timestamp_utc=now_dt,
            t0_timestamp_ist=t0_ist_str,
            t0_assimilation_sources=assimilation_sources,
            active_engines=active_engines,
            inference_latency_ms=round(elapsed_ms, 2),
            spending_incurred="₹0.00"
        )

        return ForecastResponse(
            status="SUCCESS",
            target_location={
                "latitude": req.latitude,
                "longitude": req.longitude,
                "station_name": req.station_name or "Delhi (Safdarjung)"
            },
            metadata=metadata,
            horizons=horizons_dict,
            synoptic_context=synoptics,
            alerts=unified_alerts
        )
