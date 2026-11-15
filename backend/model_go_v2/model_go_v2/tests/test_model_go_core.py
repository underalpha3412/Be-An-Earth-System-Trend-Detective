"""
Comprehensive unit and integration test suite for model_go.
"""

import pytest
from fastapi.testclient import TestClient

from model_go import Forecaster, ModelGoOrchestrator
from model_go.schemas import ForecastPointRequest, ForecastResponse, AlertLevel
from model_go.api.app import app
from model_go.fusion.spectral_nudging import SpectralNudger
from model_go.hydrology.yamuna_model import RiverYamunaHydrologyEngine


@pytest.fixture
def test_client():
    return TestClient(app)


def test_sdk_in_process_forecaster():
    fc = Forecaster()
    res = fc.predict_point(lat=28.6139, lon=77.2090, station_name="Delhi (Safdarjung)")
    assert res["status"] == "SUCCESS"
    assert "+24h" in res["horizons"]
    assert "+48h" in res["horizons"]
    assert "+72h" in res["horizons"]
    assert "+7d" in res["horizons"]
    assert "+30d" in res["horizons"]

    # Verify +24h point prediction characteristics
    h24 = res["horizons"]["+24h"]
    assert h24["dominant_engine"] == "MODEL_ALPHA_GBDT"
    assert 15.0 <= h24["temperature_c"] <= 45.0
    assert 0.0 <= h24["rain_probability"] <= 1.0

    # Verify +48h & +72h synoptic rollouts
    h48 = res["horizons"]["+48h"]
    assert h48["dominant_engine"] == "HYBRID_PHYSICS_STUDENT"
    h72 = res["horizons"]["+72h"]
    assert h72["dominant_engine"] == "HYBRID_PHYSICS_STUDENT"

    # Verify uncertainty bounds
    assert h24["temp_uncertainty_90pct"][0] < h24["temperature_c"] < h24["temp_uncertainty_90pct"][1]

    # Verify alerts
    assert len(res["alerts"]) >= 1


def test_api_health_endpoint(test_client):
    resp = test_client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "model_go"
    assert data["status"] == "OPERATIONAL"
    assert "nasa_power" in data["connectors"]
    assert "imd_operational_bridge" in data["connectors"]


def test_api_forecast_point(test_client):
    payload = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "station_name": "Delhi Test Station"
    }
    resp = test_client.post("/v1/forecast/point", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["target_location"]["station_name"] == "Delhi Test Station"
    assert data["metadata"]["inference_latency_ms"] > 0.0


def test_api_active_alerts(test_client):
    resp = test_client.get("/v1/alerts/active")
    assert resp.status_code == 200
    alerts = resp.json()
    assert isinstance(alerts, list)
    assert any(a["category"] == "YAMUNA_HYDROLOGY" for a in alerts)


def test_spectral_nudger_blending():
    nudger = SpectralNudger()
    # Test discontinuity clamping
    t48_nudged, p48_blended = nudger.blend_24h_to_48h(
        t24_alpha=25.0,
        t48_hybrid=35.0,  # Unrealistic +10C jump
        p24_alpha_mm=0.0,
        p48_hybrid_mm=5.0
    )
    # Should clamp jump to 3.8C
    assert t48_nudged == 28.8
    assert p48_blended > 0.0


def test_yamuna_hydrology_extreme_flood():
    hydro = RiverYamunaHydrologyEngine()
    # Test catastrophic flood response (150 mm rain on 85% saturated ground)
    diag = hydro.diagnose_catchment_risk(rain_24h_mm=150.0, soil_saturation=0.85, antecedent_gauge_m=204.0)
    assert diag["alert_level"] in ("ORANGE", "EXTREME_RED_ALERT")
    assert diag["projected_gauge_m"] > 205.33


def test_api_forecast_regional(test_client):
    payload = {
        "min_lat": 28.0,
        "max_lat": 29.0,
        "min_lon": 76.5,
        "max_lon": 77.8,
        "resolution_deg": 0.5
    }
    resp = test_client.post("/v1/forecast/regional", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["grid_points_count"] > 0
    assert len(data["sample_points"]) > 0
    sample = data["sample_points"][0]
    assert "+24h" in sample["horizons"]
    assert "+48h" in sample["horizons"]


def test_edge_profile_mode():
    orchestrator_edge = ModelGoOrchestrator(profile="edge")
    assert orchestrator_edge.profile == "edge"
    req = ForecastPointRequest(latitude=28.6139, longitude=77.2090, station_name="Delhi Edge")
    resp = orchestrator_edge.generate_point_forecast(req)
    assert resp.status == "SUCCESS"
    assert "HYBRID_PHYSICS_STUDENT_EDGE" in resp.metadata.active_engines
    assert resp.metadata.inference_latency_ms < 50.0  # Under 50ms on edge


def test_feature_store_caching():
    from model_go.ingestion.feature_store import LocalFeatureStore
    fs = LocalFeatureStore()
    fs.put_features("test_key", {"t2m_lag_1d": 28.5})
    val = fs.get_features("test_key")
    assert val is not None
    assert val["t2m_lag_1d"] == 28.5


def test_causal_time_gate():
    from datetime import datetime, timezone, timedelta
    from model_go.ingestion.causal_gate import CausalTimeGate, CausalLeakageException
    gate = CausalTimeGate(strict_mode=True, power_meteo_lag_days=2)
    issuance = datetime(2024, 8, 10, 6, 0, tzinfo=timezone.utc)

    # 1. Past dates should pass
    raw = {
        "T2M": {
            "20240805": 28.5,
            "20240806": 28.2,
            "20240807": 27.9,
            "20240808": 28.1,
        }
    }
    filtered = gate.filter_power_timeseries(issuance, raw)
    assert "20240808" in filtered["T2M"]

    # 2. Future peeking should raise CausalLeakageException in strict mode
    raw_leakage = {
        "T2M": {
            "20240811": 31.0
        }
    }
    with pytest.raises(CausalLeakageException):
        gate.filter_power_timeseries(issuance, raw_leakage)

    # 3. GPM IMERG 4-hour latency check
    future_granule = issuance - timedelta(hours=2)
    with pytest.raises(CausalLeakageException):
        gate.validate_gpm_granule(issuance, future_granule)


def test_circuit_breaker_timeout_fallback(monkeypatch, test_client):
    """Verify circuit-breaker intercepts timeouts and returns HTTP 200 with degraded_mode=True."""
    from model_go.ingestion.imd.operational_bridge import IMDOperationalBridge
    import requests

    def mock_fetch_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection to upstream timed out after 10000ms")

    monkeypatch.setattr(IMDOperationalBridge, "fetch_station_observation", mock_fetch_timeout)

    payload = {"latitude": 28.6139, "longitude": 77.2090, "station_name": "Delhi Safdarjung"}
    resp = test_client.post("/v1/forecast/point", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["metadata"]["degraded_mode"] is True
    assert "CIRCUIT_BREAKER_CLIMATOLOGY" in data["metadata"]["t0_assimilation_sources"]


def test_webhook_retry_and_dead_letter(tmp_path):
    from model_go.api.webhooks import WebhookDispatcher
    from model_go.schemas import UnifiedAlert, AlertLevel

    dead_letter_file = tmp_path / "dead_letter.json"
    # Target unreachable URL to trigger retries and dead-letter queue
    dispatcher = WebhookDispatcher(
        target_urls=["http://127.0.0.1:59999/unreachable_webhook"],
        max_retries=2,
        initial_backoff_sec=0.05,
        dead_letter_file=dead_letter_file
    )

    alert = UnifiedAlert(
        category="YAMUNA_HYDROLOGY",
        level=AlertLevel.EXTREME_RED_ALERT,
        title="Catastrophic Surge",
        description="Embankment overtopping test"
    )

    res = dispatcher.dispatch_alert_if_escalated([alert])
    assert res["attempted"] == 1
    assert res["successful"] == 0
    assert res["dead_lettered"] == 1
    assert dead_letter_file.exists()


def test_background_sync_worker():
    from model_go.ingestion.background_worker import IngestionBackgroundWorker
    test_stations = [
        {"station_id": "test_stn", "name": "Test Station", "latitude": 28.6139, "longitude": 77.2090}
    ]
    worker = IngestionBackgroundWorker(stations=test_stations)
    res = worker.sync_station(test_stations[0])
    assert res is True
    assert worker.feature_store.has_station("test_stn")


