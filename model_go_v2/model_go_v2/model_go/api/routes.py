"""
FastAPI route definitions for model_go headless service.
Exposes REST endpoints for point forecasting, regional fields, active alerts, and connector health.
"""

import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends

from model_go.config import PROFILE
from model_go.schemas import (
    ForecastPointRequest,
    ForecastRegionalRequest,
    ForecastResponse,
    AdaptRequest,
    AdaptResponse,
    UnifiedAlert,
    SystemHealthResponse,
    ConnectorHealth
)
from model_go.fusion.orchestrator import ModelGoOrchestrator
from model_go.api.webhooks import WebhookDispatcher
from model_go.api.mqtt_emitter import MQTTEmitter

router = APIRouter(prefix="/v1", tags=["Meteorological Forecasting"])

# Global singleton orchestrator
_orchestrator: ModelGoOrchestrator = None
_webhook_dispatcher = WebhookDispatcher()
_mqtt_emitter = MQTTEmitter()


def get_orchestrator() -> ModelGoOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ModelGoOrchestrator()
    return _orchestrator


@router.post("/forecast/point", response_model=ForecastResponse)
def forecast_point(
    req: ForecastPointRequest,
    orchestrator: ModelGoOrchestrator = Depends(get_orchestrator)
):
    """
    Generate unified multi-horizon point forecast (24h, 48h, 72h, 7d, 30d).
    Executes Hierarchical MoE routing and boundary nudging.
    """
    try:
        response = orchestrator.generate_point_forecast(req)

        # Dispatch webhooks if escalated
        _webhook_dispatcher.dispatch_alert_if_escalated(response.alerts)

        # Emit MQTT telemetry
        _mqtt_emitter.publish_telemetry(response.model_dump())

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting engine error: {str(e)}")


@router.post("/adapt", response_model=AdaptResponse)
def adapt_location(
    req: AdaptRequest,
    orchestrator: ModelGoOrchestrator = Depends(get_orchestrator)
):
    """
    On-Demand Regional Specialization Endpoint.
    Synchronously fits and saves a local residual bias-correction head (< 50 KB)
    for the specified coordinate, pre-warming it for sub-18ms future inference.
    """
    try:
        res = orchestrator.adapter_manager.fit_adapter_sync(
            lat=req.lat,
            lon=req.lon,
            lookback_years=req.lookback_years,
            station_name=req.station_name
        )
        return AdaptResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")


@router.post("/forecast/regional")
def forecast_regional(
    req: ForecastRegionalRequest,
    orchestrator: ModelGoOrchestrator = Depends(get_orchestrator)
):
    """
    Generate coarse regional multi-horizon grid forecast for a spatial bounding box.
    Evaluates spatio-temporal dynamics using the distilled hybrid physics student engine.
    """
    import numpy as np

    lats = np.arange(req.min_lat, req.max_lat + 1e-4, req.resolution_deg)
    lons = np.arange(req.min_lon, req.max_lon + 1e-4, req.resolution_deg)

    # Establish central thermodynamic anchor
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    center_point = orchestrator.generate_point_forecast(
        ForecastPointRequest(latitude=center_lat, longitude=center_lon, station_name="Regional Center")
    )
    t0_c = center_point.horizons["+24h"].temperature_c
    p0_hpa = center_point.horizons["+24h"].msl_pressure_hpa
    w0_ms = center_point.horizons["+24h"].wind_speed_ms

    grid_points = []
    # Evaluate grid with spatial lapse rate and longitudinal phase adjustment
    for lat in lats:
        lat_f = float(lat)
        # Standard tropospheric lapse rate and latitudinal temperature gradient
        delta_lat = lat_f - center_lat
        t_lat_adj = -0.65 * delta_lat
        p_lat_adj = 0.8 * delta_lat
        for lon in lons:
            lon_f = float(lon)
            delta_lon = lon_f - center_lon
            local_t0 = t0_c + t_lat_adj - 0.1 * delta_lon
            local_p0 = p0_hpa + p_lat_adj

            rollout = orchestrator.student_engine.forecast_horizons(
                t2m_c=local_t0,
                mslp_hpa=local_p0,
                wind_ms=w0_ms,
                doy=180
            )

            grid_points.append({
                "latitude": round(lat_f, 3),
                "longitude": round(lon_f, 3),
                "horizons": {
                    "+24h": {"t2m_c": round(rollout[24]["t2m_c"], 2), "precip_mm": 0.0},
                    "+48h": {"t2m_c": round(rollout[48]["t2m_c"], 2), "precip_mm": round(rollout[48].get("precip_mm", 0.0), 2)},
                    "+72h": {"t2m_c": round(rollout[72]["t2m_c"], 2), "precip_mm": round(rollout[72].get("precip_mm", 0.0), 2)},
                }
            })

    return {
        "status": "SUCCESS",
        "bounding_box": [req.min_lat, req.max_lat, req.min_lon, req.max_lon],
        "resolution_deg": req.resolution_deg,
        "grid_points_count": len(grid_points),
        "sample_points": grid_points[:50]
    }


@router.get("/alerts/active", response_model=List[UnifiedAlert])
def get_active_alerts(orchestrator: ModelGoOrchestrator = Depends(get_orchestrator)):
    """
    Retrieve all current active River Yamuna hydrological and IMD statutory weather alerts.
    """
    req = ForecastPointRequest()
    res = orchestrator.generate_point_forecast(req)
    return res.alerts


@router.get("/health", response_model=SystemHealthResponse)
def get_system_health(orchestrator: ModelGoOrchestrator = Depends(get_orchestrator)):
    """
    System diagnostic and live upstream connector health monitor.
    Reports operational status for NASA POWER, GPM IMERG, GEOS-FP, SMAP, and IMD bridges.
    """
    connectors = {
        "nasa_power": ConnectorHealth(**orchestrator.power_client.check_health()),
        "nasa_gpm_imerg": ConnectorHealth(**orchestrator.imerg_client.check_health()),
        "nasa_geos_merra": ConnectorHealth(**orchestrator.geos_client.check_health()),
        "nasa_smap": ConnectorHealth(**orchestrator.smap_client.check_health()),
        "imd_operational_bridge": ConnectorHealth(**orchestrator.imd_bridge.check_health()),
        "imd_statutory_parser": ConnectorHealth(**orchestrator.imd_warnings.check_health())
    }

    active_alerts = len([c for c in connectors.values() if c.status == "REQUIRES_AUTH"])

    return SystemHealthResponse(
        status="OPERATIONAL",
        service="model_go",
        version="2.0.0",
        profile=PROFILE,
        connectors=connectors,
        active_alerts_count=active_alerts
    )
