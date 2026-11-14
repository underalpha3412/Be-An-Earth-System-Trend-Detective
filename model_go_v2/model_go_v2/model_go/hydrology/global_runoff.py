"""
Globalized Hydrology & Flash-Flood Runoff Risk Scoring Engine for model_go.
Preserves specialized River Yamuna catchment stage modeling (205.33m Danger Mark)
for Delhi NCR while providing Normalized Flash-Flood Risk Index (0-100) worldwide.
"""

import logging
from typing import Dict, Any, Optional

from model_go.hydrology.yamuna_model import (
    RiverYamunaHydrologyEngine,
    DANGER_MARK_M,
    WARNING_MARK_M,
    BASELINE_DRY_LEVEL_M
)

logger = logging.getLogger(__name__)

# Bounding box for River Yamuna catchment basin (Wazirabad/Hathnikund to Okhla)
YAMUNA_BBOX = {
    "min_lat": 28.0,
    "max_lat": 31.5,
    "min_lon": 76.5,
    "max_lon": 78.5
}


class GlobalRunoffEngine:
    """Hydrological risk classifier routing between local river stage models and global runoff index."""

    def __init__(self):
        self.yamuna_engine = RiverYamunaHydrologyEngine()

    @staticmethod
    def is_yamuna_catchment(lat: float, lon: float) -> bool:
        """Check if coordinates fall inside the River Yamuna drainage basin."""
        return (
            YAMUNA_BBOX["min_lat"] <= lat <= YAMUNA_BBOX["max_lat"] and
            YAMUNA_BBOX["min_lon"] <= lon <= YAMUNA_BBOX["max_lon"]
        )

    def diagnose_risk(
        self,
        lat: float,
        lon: float,
        rain_24h_mm: float,
        rain_72h_mm: float = 0.0,
        soil_saturation: float = 0.45,
        antecedent_gauge_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate hydrological risk based on spatial location.
        If in Yamuna catchment -> RiverYamunaHydrologyEngine.
        If outside -> Normalized Global Flash-Flood Risk Index (0–100).
        """
        if self.is_yamuna_catchment(lat, lon):
            base_res = self.yamuna_engine.diagnose_catchment_risk(
                rain_24h_mm=rain_24h_mm,
                soil_saturation=soil_saturation,
                antecedent_gauge_m=antecedent_gauge_m or BASELINE_DRY_LEVEL_M
            )
            base_res["category"] = "YAMUNA_HYDROLOGY"
            base_res["is_yamuna"] = True
            base_res["danger_mark_m"] = DANGER_MARK_M
            base_res["runoff_risk_index"] = None
            return base_res

        # Outside Yamuna basin: evaluate Normalized Global Runoff & Flash-Flood Index (0-100)
        # Runoff increases non-linearly with soil saturation and multi-day accumulation
        runoff_coeff = 0.20 + 0.70 * (soil_saturation ** 1.5)
        effective_runoff_depth = rain_24h_mm * runoff_coeff + 0.25 * rain_72h_mm * soil_saturation

        # Reference extreme runoff: 75mm effective runoff in 24h triggers 100/100 risk
        risk_index = float(min(100.0, max(0.0, (effective_runoff_depth / 75.0) * 100.0)))
        risk_index = round(risk_index, 1)

        if risk_index >= 75.0:
            alert_level = "EXTREME_RED_ALERT"
            title = f"CATASTROPHIC FLASH FLOOD ALERT: Extreme Runoff Index {risk_index}/100"
            advisory = (
                f"Severe convective precipitation ({rain_24h_mm:.1f} mm) combined with saturated ground "
                f"({soil_saturation * 100:.0f}% soil moisture) poses immediate life-threatening flash-flooding."
            )
        elif risk_index >= 50.0:
            alert_level = "ORANGE"
            title = f"FLASH FLOOD WARNING: High Runoff Risk Index {risk_index}/100"
            advisory = (
                f"High runoff generation expected ({rain_24h_mm:.1f} mm rain). Inundation of low-lying terrain, "
                f"stormwater surcharge, and tributary overflow likely."
            )
        elif risk_index >= 25.0:
            alert_level = "YELLOW"
            title = f"HYDROLOGICAL WATCH: Moderate Runoff Risk Index {risk_index}/100"
            advisory = (
                f"Moderate surface ponding and localized drainage stress ({rain_24h_mm:.1f} mm rain, "
                f"saturation {soil_saturation * 100:.0f}%). Normal river courses maintained."
            )
        else:
            alert_level = "GREEN"
            title = f"NORMAL RUNOFF: Low Hydrological Risk Index {risk_index}/100"
            advisory = "Precipitation and soil moisture within nominal drainage carrying capacities."

        return {
            "category": "GLOBAL_HYDROLOGY",
            "is_yamuna": False,
            "alert_level": alert_level,
            "runoff_risk_index": risk_index,
            "title": title,
            "advisory": advisory,
            "projected_gauge_m": None,
            "danger_mark_m": 0.0
        }
