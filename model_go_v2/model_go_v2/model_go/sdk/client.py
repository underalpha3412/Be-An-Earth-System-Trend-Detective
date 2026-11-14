"""
Python SDK Client for model_go.
Enables seamless internal embedding:
    from model_go import Forecaster
    fc = Forecaster()
    result = fc.predict_point(lat=28.6139, lon=77.2090)
"""

from typing import Dict, Any, List, Optional
import requests

from model_go.schemas import ForecastPointRequest, ForecastResponse


class Forecaster:
    """Python SDK client for model_go in-process or remote HTTP execution."""

    def __init__(self, remote_url: Optional[str] = None):
        self.remote_url = remote_url.rstrip("/") if remote_url else None
        self._local_orchestrator = None

        if not self.remote_url:
            from model_go.fusion.orchestrator import ModelGoOrchestrator
            self._local_orchestrator = ModelGoOrchestrator()

    def predict_point(
        self,
        lat: float = 28.6139,
        lon: float = 77.2090,
        station_name: str = "Delhi (Safdarjung)",
        horizons_hours: List[int] = [24, 48, 72, 168, 720]
    ) -> Dict[str, Any]:
        """Generate point forecast either in-memory or via remote REST endpoint."""
        req_obj = ForecastPointRequest(
            latitude=lat,
            longitude=lon,
            station_name=station_name,
            horizons_hours=horizons_hours
        )

        if self.remote_url:
            url = f"{self.remote_url}/v1/forecast/point"
            resp = requests.post(url, json=req_obj.model_dump(), timeout=15)
            resp.raise_for_status()
            return resp.json()
        else:
            res: ForecastResponse = self._local_orchestrator.generate_point_forecast(req_obj)
            return res.model_dump()

    def get_health(self) -> Dict[str, Any]:
        """Check system and upstream connector health."""
        if self.remote_url:
            resp = requests.get(f"{self.remote_url}/v1/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        else:
            from model_go.api.routes import get_system_health
            return get_system_health(self._local_orchestrator).model_dump()
