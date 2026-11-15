"""
Convective Thermodynamics and Moisture Convergence Engine.
Calculates thermodynamic saturation deficit, moisture flux convergence,
and precipitation rates to support multi-horizon rollouts and runoff modeling.
"""

import math
from typing import Dict, Any, Tuple
import numpy as np


class ConvectivePrecipitationEngine:
    """Evaluates atmospheric moisture flux and convective potential."""

    def __init__(self):
        pass

    def compute_convective_potential(
        self,
        t2m_c: float,
        rh_pct: float,
        mslp_hpa: float,
        wind_ms: float
    ) -> Dict[str, float]:
        """
        Estimate thermodynamic moisture convergence and convective rain potential.
        """
        # Saturation specific humidity q_sat (g/kg) via Magnus formula
        es_hpa = 6.112 * math.exp((17.67 * t2m_c) / (t2m_c + 243.5))
        q_sat_g_kg = (622.0 * es_hpa) / max(500.0, mslp_hpa)
        q_actual_g_kg = q_sat_g_kg * (rh_pct / 100.0)

        # Saturation deficit (dryness of air)
        saturation_deficit = max(0.0, q_sat_g_kg - q_actual_g_kg)

        # Convective Available Potential Energy (CAPE) proxy from low-level thermal buoyancy
        # Higher temperature + high humidity in warm season yields strong instability
        cape_proxy = max(0.0, (t2m_c - 18.0) * 85.0 * (rh_pct / 100.0))

        # Moisture convergence proxy
        moisture_convergence = (q_actual_g_kg * wind_ms * 0.15) if rh_pct > 70.0 else 0.0

        return {
            "q_sat_g_kg": round(q_sat_g_kg, 2),
            "q_actual_g_kg": round(q_actual_g_kg, 2),
            "saturation_deficit_g_kg": round(saturation_deficit, 2),
            "cape_proxy_j_kg": round(cape_proxy, 1),
            "moisture_convergence_proxy": round(moisture_convergence, 2)
        }
