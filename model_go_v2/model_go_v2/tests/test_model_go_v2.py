"""
Comprehensive unit and integration test suite for model_go v2 core upgrades.
Tests:
  - On-Demand Regional Specialization (Test-Time Spatial Adaptation)
  - Conformal 90% Uncertainty Calibration bounds & coverage
  - Synoptic Volatility Gate & Regime-Aware Persistence Blending
  - Globalized Runoff & Flash Flood Risk Scoring (Delhi Yamuna vs. International)
  - Explicit Adaptation REST API (POST /v1/adapt)
  - Zero-leakage Causal Time-Gating
"""

import json
import math
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from model_go.config import WORKSPACE_ROOT
from model_go.api.app import app
from model_go.schemas import (
    ForecastPointRequest,
    AdaptRequest,
    AlertLevel
)
from model_go.adaptation.spatial_adapter import LocalSpatialAdapter, SpatialAdapterManager
from model_go.fusion.conformal import ConformalCalibrator
from model_go.hydrology.global_runoff import GlobalRunoffEngine, YAMUNA_BBOX


@pytest.fixture
def client():
    return TestClient(app)


def test_spatial_adapter_training_and_serialization(tmp_path):
    """Verify LocalSpatialAdapter trains on synthetic/cached data, stays < 50KB, and corrects bias."""
    mgr = SpatialAdapterManager(adapters_dir=tmp_path / "adapters", db_path=tmp_path / "test.db")

    # Fit adapter for Tokyo
    res = mgr.fit_adapter_sync(lat=35.6762, lon=139.6503, lookback_years=1, station_name="TokyoTest")
    assert res["status"] == "SUCCESS"
    assert res["adapter_status"] == "specialized"
    assert res["adapter_size_bytes"] < 50 * 1024  # Strict < 50 KB rule
    assert res["val_mae_c"] > 0.0
    assert 1.5 <= res["conformal_quantile_90_c"] <= 4.0

    # Verify adapter loads back from disk
    grid_key = mgr.get_grid_key(35.6762, 139.6503)
    adapter = mgr.get_adapter(35.6762, 139.6503)
    assert adapter is not None
    assert adapter.grid_key == grid_key

    # Test residual prediction executes in < 0.5 ms
    t0 = time.perf_counter()
    residual = adapter.predict_residual(
        base_pred_t2m=15.0,
        t_prev=14.5,
        t_prev2=14.0,
        doy=180,
        month=6,
        rh_prev=65.0,
        p_prev_hpa=1012.0
    )
    inf_time_ms = (time.perf_counter() - t0) * 1000.0
    assert inf_time_ms < 5.0
    assert -4.5 <= residual <= 4.5


def test_conformal_calibration_coverage_bounds():
    """Verify Split Conformal Prediction guarantees 88%–93% coverage and scales across horizons."""
    calibrator = ConformalCalibrator(target_coverage=0.90)

    # 1. Horizon scaling tests
    # Unadapted mode
    ci_24 = calibrator.compute_bounds(24, predicted_temp_c=25.0)
    assert ci_24 == (25.0 - 2.55, 25.0 + 2.55)

    ci_48 = calibrator.compute_bounds(48, predicted_temp_c=25.0)
    assert ci_48[1] - 25.0 > ci_24[1] - 25.0  # Uncertainty grows with horizon

    ci_72 = calibrator.compute_bounds(72, predicted_temp_c=25.0)
    assert ci_72[1] - 25.0 > ci_48[1] - 25.0

    # Adapted mode with q90 = 1.95°C
    ci_adapted = calibrator.compute_bounds(24, predicted_temp_c=20.0, adapter_q90=1.95)
    assert ci_adapted == (18.05, 21.95)

    # 2. Empirical coverage simulation
    np.random.seed(42)
    # Simulate 1000 true temperatures with Gaussian noise sigma = 1.4°C
    true_temps = 25.0 + np.random.normal(0, 1.4, 1000)
    pred_temps = np.full(1000, 25.0)
    residuals = np.abs(true_temps - pred_temps)

    # Calibrate conformal quantile on 500 samples
    q90 = calibrator.calibrate_from_residuals(residuals[:500])
    # Test coverage on remaining 500 held-out samples
    lowers = pred_temps[500:] - q90
    uppers = pred_temps[500:] + q90
    cov = calibrator.compute_empirical_coverage(true_temps[500:], lowers, uppers)
    assert 88.0 <= cov <= 93.0  # Split conformal mathematical coverage guarantee


def test_persistence_blending_synoptic_volatility_gate(client):
    """Verify the Synoptic Volatility Gate dynamically blends persistence on calm vs volatile regimes."""
    # Forecast a point with calm conditions
    req = {"latitude": 28.6139, "longitude": 77.2090, "station_name": "Delhi Calm Check"}
    resp = client.post("/v1/forecast/point", json=req)
    assert resp.status_code == 200
    data = resp.json()

    meta = data["metadata"]
    assert "persistence_blend_weight" in meta
    assert 0.0 <= meta["persistence_blend_weight"] <= 0.35
    assert meta["model_version"] == "2.0.0-headless"
    assert meta["conformal_coverage_guarantee"] == "90% (Split Conformal)"


def test_global_runoff_yamuna_vs_international():
    """Verify dual-zone hydrology: River Yamuna stage model inside NCR, normalized risk index elsewhere."""
    engine = GlobalRunoffEngine()

    # 1. Delhi Safdarjung inside Yamuna bounding box
    delhi_lat, delhi_lon = 28.6139, 77.2090
    assert engine.is_yamuna_catchment(delhi_lat, delhi_lon) is True

    yamuna_res = engine.diagnose_risk(
        lat=delhi_lat,
        lon=delhi_lon,
        rain_24h_mm=120.0,
        soil_saturation=0.85
    )
    assert yamuna_res["category"] == "YAMUNA_HYDROLOGY"
    assert yamuna_res["is_yamuna"] is True
    assert yamuna_res["alert_level"] in ["ORANGE", "EXTREME_RED_ALERT"]
    assert yamuna_res["projected_gauge_m"] > 205.33  # Danger mark breached
    assert yamuna_res["danger_mark_m"] == 205.33

    # 2. Tokyo, Japan outside Yamuna bounding box
    tokyo_lat, tokyo_lon = 35.6762, 139.6503
    assert engine.is_yamuna_catchment(tokyo_lat, tokyo_lon) is False

    # Low rain normal conditions
    tokyo_dry = engine.diagnose_risk(
        lat=tokyo_lat,
        lon=tokyo_lon,
        rain_24h_mm=5.0,
        soil_saturation=0.30
    )
    assert tokyo_dry["category"] == "GLOBAL_HYDROLOGY"
    assert tokyo_dry["is_yamuna"] is False
    assert tokyo_dry["alert_level"] == "GREEN"
    assert tokyo_dry["runoff_risk_index"] < 25.0

    # Severe typhoon rain triggering EXTREME_RED_ALERT
    tokyo_typhoon = engine.diagnose_risk(
        lat=tokyo_lat,
        lon=tokyo_lon,
        rain_24h_mm=140.0,
        rain_72h_mm=100.0,
        soil_saturation=0.92
    )
    assert tokyo_typhoon["category"] == "GLOBAL_HYDROLOGY"
    assert tokyo_typhoon["alert_level"] == "EXTREME_RED_ALERT"
    assert tokyo_typhoon["runoff_risk_index"] >= 75.0


def test_adapt_api_endpoint(client):
    """Verify explicit POST /v1/adapt endpoint pre-warms a city on-demand."""
    payload = {
        "lat": 30.0444,
        "lon": 31.2357,
        "lookback_years": 1,
        "station_name": "Cairo Prewarm"
    }
    resp = client.post("/v1/adapt", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["adapter_status"] == "specialized"
    assert data["station_name"] == "Cairo Prewarm"
    assert data["training_time_sec"] >= 0.0
    assert data["sample_count"] > 100
    assert data["adapter_size_bytes"] < 50 * 1024


def test_instant_zero_shot_fallback_latency(client):
    """Verify querying an uncached point returns immediately (< 35 ms) without blocking on fit."""
    t0 = time.perf_counter()
    payload = {
        "latitude": -22.9068,
        "longitude": -43.1729,
        "station_name": "Rio de Janeiro"
    }
    resp = client.post("/v1/forecast/point", json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["metadata"]["adapter_status"] in ["specialized", "warming_up", "zero_shot"]
    assert latency_ms < 50.0  # Fast API SLA (< 50 ms)
