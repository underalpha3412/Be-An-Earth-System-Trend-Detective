"""
Autonomous NASA POWER API client for model_go ingestion layer.
Retrieves unauthenticated point meteorology and solar parameters for Model Alpha's lag pipeline.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import requests

from model_go.config import NASA_POWER_BASE
from model_go.ingestion.base import BaseIngestionConnector

logger = logging.getLogger(__name__)


class NASAPowerConnector(BaseIngestionConnector):
    """Zero-authentication connector for NASA POWER point daily observations."""

    def __init__(self, cache_ttl_seconds: int = 14400):
        super().__init__(name="nasa_power", cache_ttl_seconds=cache_ttl_seconds)

    def fetch_data(
        self,
        latitude: float = 28.6139,
        longitude: float = 77.2090,
        days_back: int = 7,
        force_fresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch multi-day meteorological time-series for a point."""
        cache_key = f"power_{latitude:.4f}_{longitude:.4f}_{days_back}"
        if not force_fresh:
            cached = self.read_cache(cache_key)
            if cached:
                return cached

        t0 = time.perf_counter()
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS10M,WD10M,ALLSKY_SFC_SW_DWN,PS",
            "community": "AG",
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": start_str,
            "end": end_str,
            "format": "JSON",
        }

        try:
            resp = requests.get(NASA_POWER_BASE, params=params, timeout=12)
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_sync_utc = datetime.now(timezone.utc)

            if resp.status_code == 200:
                data = resp.json()
                series = data.get("properties", {}).get("parameter", {})
                payload = {
                    "source": "NASA_POWER_API",
                    "latitude": latitude,
                    "longitude": longitude,
                    "parameters": series,
                    "fetched_at": self.last_sync_utc.isoformat(),
                    "latency_ms": round(self.last_latency_ms, 2)
                }
                self.write_cache(cache_key, payload)
                return payload
            else:
                logger.warning("NASA POWER API returned HTTP %s: %s", resp.status_code, resp.text[:100])
        except Exception as e:
            logger.warning("NASA POWER API connection failed: %s", e)

        # Fallback to local cached historical observation if live call fails
        cached_fallback = self.read_cache(cache_key)
        if cached_fallback:
            return cached_fallback

        return {
            "source": "NASA_POWER_FALLBACK",
            "latitude": latitude,
            "longitude": longitude,
            "parameters": {},
            "status": "OFFLINE"
        }

    def check_health(self) -> Dict[str, Any]:
        """Ping NASA POWER API to verify live status."""
        t0 = time.perf_counter()
        try:
            resp = requests.get("https://power.larc.nasa.gov/api/system/manager/parameters", timeout=5)
            latency = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                return {
                    "name": self.name,
                    "status": "HEALTHY",
                    "latency_ms": round(latency, 2),
                    "notes": "Unauthenticated public endpoint online"
                }
        except Exception as e:
            return {
                "name": self.name,
                "status": "DEGRADED",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "notes": f"Network error: {str(e)[:100]}"
            }
        return {"name": self.name, "status": "OFFLINE", "latency_ms": 0.0, "notes": "No response"}
