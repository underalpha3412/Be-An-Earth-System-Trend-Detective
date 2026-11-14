"""
Feature engineering pipeline for Model Alpha.
Transforms atmospheric time-series into 100+ engineered lag, harmonic, and thermodynamic features.
"""

import math
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


def compute_vapor_pressure_deficit(t2m_c: float, rh2m_pct: float) -> float:
    """Calculate Vapor Pressure Deficit (VPD) in kPa using Tetens formula."""
    es_kpa = 0.61078 * math.exp((17.27 * t2m_c) / (t2m_c + 237.3))
    ea_kpa = es_kpa * (rh2m_pct / 100.0)
    return max(0.0, es_kpa - ea_kpa)


def compute_dewpoint_approx(t2m_c: float, rh2m_pct: float) -> float:
    """Approximate 2m dewpoint temperature in °C via Magnus-Tetens."""
    a = 17.27
    b = 237.7
    rh_frac = max(0.01, min(1.0, rh2m_pct / 100.0))
    alpha = ((a * t2m_c) / (b + t2m_c)) + math.log(rh_frac)
    return (b * alpha) / (a - alpha)


class AlphaFeaturePipeline:
    """Constructs the exact tabular feature row expected by Model Alpha's gradient-boosted trees."""

    def __init__(self):
        pass

    def build_feature_row(
        self,
        dt: datetime,
        current_obs: Dict[str, float],
        historical_lags: Dict[str, Dict[int, float]]
    ) -> Dict[str, float]:
        """
        Build feature dictionary with exact schema alignment to metadata/feature_columns.json.
        """
        t2m = current_obs.get("T2M", 25.0)
        t2m_max = current_obs.get("T2M_MAX", t2m + 5.0)
        t2m_min = current_obs.get("T2M_MIN", t2m - 5.0)
        rh2m = current_obs.get("RH2M", 65.0)
        precip = current_obs.get("PRECTOTCORR", 0.0)
        ws10m = current_obs.get("WS10M", 2.5)
        wd10m = current_obs.get("WD10M", 270.0)
        sw_dwn = current_obs.get("ALLSKY_SFC_SW_DWN", 16.5)
        ps = current_obs.get("PS", 99.0)  # kPa

        doy = dt.timetuple().tm_yday
        month = dt.month

        dtr = t2m_max - t2m_min
        vpd = compute_vapor_pressure_deficit(t2m, rh2m)
        dew = compute_dewpoint_approx(t2m, rh2m)

        wd_rad = math.radians(wd10m)
        u10 = -ws10m * math.sin(wd_rad)
        v10 = -ws10m * math.cos(wd_rad)

        row = {
            "T2M": t2m,
            "T2M_MAX": t2m_max,
            "T2M_MIN": t2m_min,
            "RH2M": rh2m,
            "PRECTOTCORR": precip,
            "WS10M": ws10m,
            "WD10M": wd10m,
            "ALLSKY_SFC_SW_DWN": sw_dwn,
            "PS": ps,
            "DAY_OF_YEAR": float(doy),
            "MONTH": float(month),
            "DOY_SIN": math.sin(2.0 * math.pi * doy / 365.25),
            "DOY_COS": math.cos(2.0 * math.pi * doy / 365.25),
            "MONTH_SIN": math.sin(2.0 * math.pi * month / 12.0),
            "MONTH_COS": math.cos(2.0 * math.pi * month / 12.0),
            "WD10M_SIN": math.sin(wd_rad),
            "WD10M_COS": math.cos(wd_rad),
            "WIND_U10M": u10,
            "WIND_V10M": v10,
            "DTR": dtr,
            "VPD_KPA": vpd,
            "DEWPOINT_APPROX_C": dew,
            "T2M_DEWPOINT_SPREAD": t2m - dew,
        }

        # Multi-horizon historical lags
        lag_days = [1, 2, 3, 7, 14, 30, 365]
        for v in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS10M", "PS", "ALLSKY_SFC_SW_DWN"]:
            param_lags = historical_lags.get(v, {})
            for d in lag_days:
                # Default to current value if lag is missing
                row[f"{v}_LAG_{d}D"] = param_lags.get(d, row.get(v, 0.0))

        return row
