"""
River Yamuna Catchment Runoff & Convective Flood Risk Engine.
Couples GPM IMERG satellite precipitation, SMAP antecedent soil saturation,
and predicted convective rainfall to project Yamuna gauge levels at Old Railway Bridge (Delhi).
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# CWC Official Hydrological Thresholds for Delhi (Old Railway Bridge)
WARNING_MARK_M = 204.50
DANGER_MARK_M = 205.33
RECORD_FLOOD_JULY_2023_M = 208.66
BASELINE_DRY_LEVEL_M = 203.10


class RiverYamunaHydrologyEngine:
    """Predicts catchment runoff and gauge stages for Delhi NCT."""

    def __init__(self):
        # Catchment area upstream of Delhi (Wazirabad/Hathnikund to Okhla) ~ 19,000 km^2
        self.catchment_area_km2 = 19400.0

    def diagnose_catchment_risk(
        self,
        rain_24h_mm: float,
        soil_saturation: float = 0.48,
        antecedent_gauge_m: float = BASELINE_DRY_LEVEL_M
    ) -> Dict[str, Any]:
        """
        Compute projected River Yamuna water level and assign emergency alert tier.
        """
        # Runoff coefficient increases non-linearly with antecedent soil moisture
        runoff_coeff = 0.20 + 0.60 * (soil_saturation ** 2)
        effective_runoff_depth_mm = rain_24h_mm * runoff_coeff

        # Empirical stage-discharge-runoff response for Old Railway Bridge:
        # Every 10 mm of basin-wide effective runoff yields ~0.35m stage rise within 24-36h
        stage_rise_m = (effective_runoff_depth_mm / 10.0) * 0.35
        projected_gauge_m = antecedent_gauge_m + stage_rise_m

        if projected_gauge_m >= 208.0:
            alert_level = "EXTREME_RED_ALERT"
            title = "CATASTROPHIC FLOOD ALERT: Yamuna Embankment Overtopping Risk"
            advisory = f"Projected gauge {projected_gauge_m:.2f}m exceeds record levels (208.66m July 2023). Immediate evacuation of low-lying floodplains required."
        elif projected_gauge_m >= DANGER_MARK_M:
            alert_level = "ORANGE"
            title = "FLOOD WARNING: Yamuna River Flowing Above Danger Mark"
            advisory = f"Projected gauge {projected_gauge_m:.2f}m breaches official Danger Mark (205.33m). High runoff from upstream Hathnikund Barrage."
        elif projected_gauge_m >= WARNING_MARK_M:
            alert_level = "YELLOW"
            title = "FLOOD WATCH: Yamuna Water Level Approaching Warning Mark"
            advisory = f"Projected gauge {projected_gauge_m:.2f}m approaching Warning Mark (204.50m). Enhanced monitoring at Old Railway Bridge."
        else:
            alert_level = "GREEN"
            title = "NORMAL FLOW: River Yamuna Flowing Below Warning Level"
            advisory = f"Projected gauge {projected_gauge_m:.2f}m is safely below Warning Mark (204.50m). Drainage networks functioning nominally."

        return {
            "alert_level": alert_level,
            "title": title,
            "advisory": advisory,
            "projected_gauge_m": round(projected_gauge_m, 2),
            "effective_runoff_depth_mm": round(effective_runoff_depth_mm, 2),
            "runoff_coefficient": round(runoff_coeff, 3)
        }
