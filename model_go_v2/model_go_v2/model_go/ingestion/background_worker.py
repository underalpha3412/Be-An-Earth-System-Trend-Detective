"""
Asynchronous Background Sync Worker for model_go.
Decouples live HTTP queries to NASA and IMD APIs from the user-facing request path.
Pre-aggregates and writes Model Alpha's 100+ lag features into Redis and SQLite
so POST /v1/forecast/point executes in under 20 milliseconds.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from model_go.ingestion.feature_store import LocalFeatureStore
from model_go.ingestion.nasa.power_client import NASAPowerConnector
from model_go.ingestion.imd.operational_bridge import IMDOperationalBridge
from model_go.ingestion.causal_gate import CausalTimeGate
from model_go.engines.alpha.feature_pipeline import AlphaFeaturePipeline

logger = logging.getLogger(__name__)

DEFAULT_MONITORED_STATIONS = [
    {"station_id": "28.6139_77.2090", "name": "Delhi (Safdarjung)", "latitude": 28.6139, "longitude": 77.2090},
    {"station_id": "28.6692_77.4538", "name": "Ghaziabad (Hindon)", "latitude": 28.6692, "longitude": 77.4538},
    {"station_id": "28.4595_77.0266", "name": "Gurugram (Aravalli Ridge)", "latitude": 28.4595, "longitude": 77.0266},
    {"station_id": "28.9845_77.7064", "name": "Meerut (Agricultural Plain)", "latitude": 28.9845, "longitude": 77.7064},
]


class IngestionBackgroundWorker:
    """Scheduled background worker pre-caching lag features for fast inference."""

    def __init__(
        self,
        stations: Optional[List[Dict[str, Any]]] = None,
        sync_interval_seconds: int = 14400  # Sync every 4 hours
    ):
        self.stations = stations or DEFAULT_MONITORED_STATIONS
        self.sync_interval = sync_interval_seconds
        self.feature_store = LocalFeatureStore()
        self.power_client = NASAPowerConnector()
        self.imd_bridge = IMDOperationalBridge()
        self.causal_gate = CausalTimeGate(strict_mode=False)
        self.pipeline = AlphaFeaturePipeline()
        self._is_running = False

    def sync_station(self, station: Dict[str, Any]) -> bool:
        """Fetch upstream observations and pre-compute 100+ lag features for one station."""
        lat = station["latitude"]
        lon = station["longitude"]
        sid = station.get("station_id", f"{lat:.4f}_{lon:.4f}")
        name = station.get("name", sid)
        now_dt = datetime.now(timezone.utc)

        logger.info("Syncing features for station: %s (%s)", name, sid)
        try:
            # 1. Fetch AWS ground telemetry
            imd_obs = self.imd_bridge.fetch_station_observation(name)
            curr_t = float(imd_obs.get("current_temp_c", 25.0))
            curr_rh = float(imd_obs.get("humidity_pct", 65.0))
            curr_p = float(imd_obs.get("pressure_hpa", 1013.0))
            curr_w = float(imd_obs.get("wind_speed_ms", 2.5))
            curr_wd = float(imd_obs.get("wind_direction_deg", 270.0))

            # 2. Fetch NASA POWER historical lag series (7 days back)
            power_data = self.power_client.fetch_data(lat, lon, days_back=7)
            raw_params = power_data.get("parameters", {})

            # 3. Apply causal time gate to enforce satellite latency cutoff
            causal_params = self.causal_gate.filter_power_timeseries(
                issuance_time_utc=now_dt,
                power_parameters=raw_params,
                station_aws_history={
                    (now_dt - timedelta(days=1)).strftime("%Y%m%d"): {
                        "T2M": curr_t - 0.5,
                        "RH2M": curr_rh,
                        "PRECTOTCORR": float(imd_obs.get("rainfall_24h_mm", 0.0)),
                        "WS10M": curr_w
                    }
                }
            )

            # 4. Construct feature row
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
            features = self.pipeline.build_feature_row(now_dt, raw_obs, causal_params)

            # 5. Store in FeatureStore
            self.feature_store.put_features(
                sid,
                features,
                {"station_name": name, "synced_at": now_dt.isoformat(), "lat": lat, "lon": lon}
            )
            logger.info("Successfully cached features for %s into feature store.", name)
            return True
        except Exception as e:
            logger.warning("Failed background sync for %s: %s", name, e)
            return False

    def sync_all_stations(self) -> Dict[str, bool]:
        """Perform full sync across all monitored stations."""
        results = {}
        for stn in self.stations:
            results[stn.get("name", stn["station_id"])] = self.sync_station(stn)
        return results

    async def run_sync_daemon(self):
        """Asynchronous periodic execution loop."""
        self._is_running = True
        logger.info("Background Ingestion Worker started (interval: %ds)", self.sync_interval)
        while self._is_running:
            try:
                self.sync_all_stations()
            except Exception as e:
                logger.error("Error in background sync cycle: %s", e)
            await asyncio.sleep(self.sync_interval)

    def stop(self):
        self._is_running = False
