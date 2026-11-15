"""
Causal Latency Gate & Time-Gating Enforcer for model_go.
Guarantees zero-leakage live operational forecasting by enforcing real-world
satellite downlink delays, processing latencies, and sensor cutoffs at issuance time t0.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Real-world operational latency definitions
NASA_POWER_METEO_LATENCY_DAYS = 2    # ~48-72h publication delay for daily meteo
NASA_POWER_SOLAR_LATENCY_DAYS = 5    # ~5-7 days for CERES/MERRA-2 radiative clearance
NASA_GPM_IMERG_LATENCY_HOURS = 4     # >= 4h early-run downlink/gridding delay
NASA_SMAP_LATENCY_HOURS = 24         # ~24-48h for L3/L4 radiometer assimilation


class CausalLeakageException(Exception):
    """Raised when an evaluation or ingestion module attempts to peek into the future."""
    pass


class CausalTimeGate:
    """
    Enforces causal availability boundaries at forecast issuance timestamp t0.
    Bridges recent observational gaps (t-1d, t-2d) cleanly via IMD AWS ground telemetry
    and GEOS-FP analysis rather than peeking at post-facto daily satellite products.
    """

    def __init__(
        self,
        strict_mode: bool = True,
        power_meteo_lag_days: int = NASA_POWER_METEO_LATENCY_DAYS,
        power_solar_lag_days: int = NASA_POWER_SOLAR_LATENCY_DAYS,
        gpm_lag_hours: int = NASA_GPM_IMERG_LATENCY_HOURS,
        smap_lag_hours: int = NASA_SMAP_LATENCY_HOURS
    ):
        self.strict_mode = strict_mode
        self.power_meteo_lag_days = power_meteo_lag_days
        self.power_solar_lag_days = power_solar_lag_days
        self.gpm_lag_hours = gpm_lag_hours
        self.smap_lag_hours = smap_lag_hours

    def filter_power_timeseries(
        self,
        issuance_time_utc: datetime,
        power_parameters: Dict[str, Dict[str, float]],
        station_aws_history: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Enforce causal cutoff on NASA POWER time series.
        Removes any dates within [t0 - lag_days, t0] and bridges recent gaps
        using real IMD AWS station telemetry if available.
        """
        causal_meteo_cutoff = (issuance_time_utc - timedelta(days=self.power_meteo_lag_days)).strftime("%Y%m%d")
        causal_solar_cutoff = (issuance_time_utc - timedelta(days=self.power_solar_lag_days)).strftime("%Y%m%d")

        filtered_params: Dict[str, Dict[str, float]] = {}

        for param_name, date_dict in power_parameters.items():
            filtered_params[param_name] = {}
            cutoff = causal_solar_cutoff if param_name == "ALLSKY_SFC_SW_DWN" else causal_meteo_cutoff
            
            for date_key, val in date_dict.items():
                if date_key <= cutoff:
                    filtered_params[param_name][date_key] = val
                elif self.strict_mode and date_key > issuance_time_utc.strftime("%Y%m%d"):
                    raise CausalLeakageException(
                        f"Temporal Leakage: Date {date_key} exceeds forecast issuance {issuance_time_utc.strftime('%Y%m%d')}"
                    )

        # Bridge recent gap (t-1d, t-2d) via IMD AWS observations if provided
        if station_aws_history:
            for day_offset in range(1, self.power_meteo_lag_days + 1):
                gap_date = (issuance_time_utc - timedelta(days=day_offset)).strftime("%Y%m%d")
                if gap_date in station_aws_history:
                    aws_day = station_aws_history[gap_date]
                    if "T2M" in filtered_params and "T2M" in aws_day:
                        filtered_params["T2M"][gap_date] = aws_day["T2M"]
                    if "RH2M" in filtered_params and "RH2M" in aws_day:
                        filtered_params["RH2M"][gap_date] = aws_day["RH2M"]
                    if "PRECTOTCORR" in filtered_params and "PRECTOTCORR" in aws_day:
                        filtered_params["PRECTOTCORR"][gap_date] = aws_day["PRECTOTCORR"]
                    if "WS10M" in filtered_params and "WS10M" in aws_day:
                        filtered_params["WS10M"][gap_date] = aws_day.get("WS10M", 2.5)

        return filtered_params

    def validate_gpm_granule(
        self,
        issuance_time_utc: datetime,
        granule_timestamp_utc: datetime
    ) -> bool:
        """
        Verify that GPM IMERG Early Run granule timestamp is at least 4 hours
        prior to forecast issuance time t0.
        """
        allowed_max = issuance_time_utc - timedelta(hours=self.gpm_lag_hours)
        if granule_timestamp_utc > allowed_max:
            if self.strict_mode:
                raise CausalLeakageException(
                    f"GPM IMERG Leakage: Granule at {granule_timestamp_utc.isoformat()} "
                    f"is within the {self.gpm_lag_hours}h downlink latency window of issuance {issuance_time_utc.isoformat()}"
                )
            return False
        return True

    def validate_smap_soil_moisture(
        self,
        issuance_time_utc: datetime,
        smap_timestamp_utc: datetime
    ) -> bool:
        """
        Verify SMAP L3/L4 soil moisture pass satisfies operational latency (>= 24h).
        """
        allowed_max = issuance_time_utc - timedelta(hours=self.smap_lag_hours)
        if smap_timestamp_utc > allowed_max:
            if self.strict_mode:
                raise CausalLeakageException(
                    f"SMAP Leakage: Soil moisture observation at {smap_timestamp_utc.isoformat()} "
                    f"is within the {self.smap_lag_hours}h latency window of issuance {issuance_time_utc.isoformat()}"
                )
            return False
        return True
