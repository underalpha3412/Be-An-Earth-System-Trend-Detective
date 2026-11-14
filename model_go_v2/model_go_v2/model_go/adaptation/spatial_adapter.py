"""
On-Demand Regional Specialization & Test-Time Spatial Adaptation Engine.
Provides instant zero-shot fallback (< 20 ms) with async background warmup,
and fast local residual bias-correction (< 50 KB serialized adapter) keyed by grid cell.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import requests

from model_go.config import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

ADAPTERS_DIR = WORKSPACE_ROOT / "data" / "adapters"
HISTORICAL_DB_PATH = WORKSPACE_ROOT / "data" / "historical_audit_cache.sqlite3"


class LocalSpatialAdapter:
    """Lightweight (< 50 KB) localized residual head and climatology normalizer."""

    def __init__(
        self,
        grid_key: str,
        latitude: float,
        longitude: float,
        station_name: str,
        monthly_climatology: Dict[str, List[float]],
        ridge_weights: List[float],
        ridge_intercept: float,
        binned_residuals: Dict[str, float],
        conformal_quantile_90: float,
        sample_count: int = 0,
        train_mae_c: float = 0.0,
        val_mae_c: float = 0.0,
        fit_time_sec: float = 0.0,
        fitted_at: Optional[str] = None
    ):
        self.grid_key = grid_key
        self.latitude = latitude
        self.longitude = longitude
        self.station_name = station_name
        self.monthly_climatology = monthly_climatology  # {"means": [12 floats], "stds": [12 floats]}
        self.ridge_weights = ridge_weights              # ~8-9 coefficients
        self.ridge_intercept = ridge_intercept
        self.binned_residuals = binned_residuals        # non-linear quantile corrections
        self.conformal_quantile_90 = conformal_quantile_90
        self.sample_count = sample_count
        self.train_mae_c = round(train_mae_c, 3)
        self.val_mae_c = round(val_mae_c, 3)
        self.fit_time_sec = round(fit_time_sec, 3)
        self.fitted_at = fitted_at or datetime.now(timezone.utc).isoformat()

    def get_monthly_clim(self, month: int) -> Tuple[float, float]:
        """Return (mean, std) for given month (1..12)."""
        idx = max(0, min(11, month - 1))
        m = self.monthly_climatology.get("means", [25.0] * 12)[idx]
        s = self.monthly_climatology.get("stds", [3.0] * 12)[idx]
        return float(m), float(s)

    def extract_features(
        self,
        base_pred_t2m: float,
        t_prev: float,
        t_prev2: float,
        doy: int,
        month: int,
        rh_prev: float = 60.0,
        p_prev_hpa: float = 1013.25
    ) -> np.ndarray:
        """Construct normalized feature vector for the residual ridge regressor."""
        clim_mean, clim_std = self.get_monthly_clim(month)
        clim_anomaly = t_prev - clim_mean
        diurnal_momentum = t_prev - t_prev2
        sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
        cos_doy = math.cos(2.0 * math.pi * doy / 365.25)
        rh_norm = (rh_prev - 50.0) / 25.0
        p_norm = (p_prev_hpa - 1013.25) / 10.0

        return np.array([
            base_pred_t2m,
            t_prev,
            diurnal_momentum,
            clim_anomaly,
            sin_doy,
            cos_doy,
            rh_norm,
            p_norm
        ], dtype=np.float64)

    def predict_residual(
        self,
        base_pred_t2m: float,
        t_prev: float,
        t_prev2: float,
        doy: int,
        month: int,
        rh_prev: float = 60.0,
        p_prev_hpa: float = 1013.25
    ) -> float:
        """Evaluate local bias correction in < 0.1 ms."""
        x = self.extract_features(base_pred_t2m, t_prev, t_prev2, doy, month, rh_prev, p_prev_hpa)
        w = np.array(self.ridge_weights, dtype=np.float64)
        linear_residual = float(np.dot(w, x) + self.ridge_intercept)

        # Apply non-linear bin correction if available
        bin_key = f"bin_{int(math.floor(base_pred_t2m / 5.0) * 5)}"
        non_linear_adj = self.binned_residuals.get(bin_key, 0.0)

        total_residual = linear_residual + 0.35 * non_linear_adj
        # Clamp maximum residual correction to prevent erratic runaway (< 4.5°C)
        return float(np.clip(total_residual, -4.5, 4.5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_key": self.grid_key,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "station_name": self.station_name,
            "monthly_climatology": self.monthly_climatology,
            "ridge_weights": self.ridge_weights,
            "ridge_intercept": self.ridge_intercept,
            "binned_residuals": self.binned_residuals,
            "conformal_quantile_90": self.conformal_quantile_90,
            "sample_count": self.sample_count,
            "train_mae_c": self.train_mae_c,
            "val_mae_c": self.val_mae_c,
            "fit_time_sec": self.fit_time_sec,
            "fitted_at": self.fitted_at,
            "adapter_size_bytes": 0  # Calculated dynamically on save
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalSpatialAdapter":
        return cls(
            grid_key=data["grid_key"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            station_name=data.get("station_name", "Global Station"),
            monthly_climatology=data["monthly_climatology"],
            ridge_weights=data["ridge_weights"],
            ridge_intercept=data["ridge_intercept"],
            binned_residuals=data.get("binned_residuals", {}),
            conformal_quantile_90=data.get("conformal_quantile_90", 2.25),
            sample_count=data.get("sample_count", 0),
            train_mae_c=data.get("train_mae_c", 0.0),
            val_mae_c=data.get("val_mae_c", 0.0),
            fit_time_sec=data.get("fit_time_sec", 0.0),
            fitted_at=data.get("fitted_at")
        )

    def save_json(self, path: Path) -> int:
        d = self.to_dict()
        serialized = json.dumps(d, indent=2)
        size_bytes = len(serialized.encode("utf-8"))
        d["adapter_size_bytes"] = size_bytes
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return size_bytes


class SpatialAdapterManager:
    """Manages adapter persistence, on-demand caching, and asynchronous background training."""

    def __init__(
        self,
        adapters_dir: Path = ADAPTERS_DIR,
        db_path: Path = HISTORICAL_DB_PATH
    ):
        self.adapters_dir = adapters_dir
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._memory_cache: Dict[str, LocalSpatialAdapter] = {}
        self._warmup_in_progress: Set[str] = set()
        self._lock = threading.Lock()
        self._load_existing_adapters()

    @staticmethod
    def get_grid_key(lat: float, lon: float) -> str:
        """Key coordinates by rounded 0.25-degree regional cell."""
        lat_rounded = round(lat * 4.0) / 4.0
        lon_rounded = round(lon * 4.0) / 4.0
        return f"{lat_rounded:+.2f}_{lon_rounded:+.2f}"

    def _load_existing_adapters(self):
        """Pre-populate memory cache from disk on startup."""
        for p in self.adapters_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                adapter = LocalSpatialAdapter.from_dict(data)
                self._memory_cache[adapter.grid_key] = adapter
            except Exception as e:
                logger.warning("Failed loading adapter %s: %s", p.name, e)
        logger.info("Loaded %d specialized spatial adapters from disk", len(self._memory_cache))

    def get_adapter(self, lat: float, lon: float) -> Optional[LocalSpatialAdapter]:
        key = self.get_grid_key(lat, lon)
        return self._memory_cache.get(key)

    def request_adapter(
        self,
        lat: float,
        lon: float,
        station_name: Optional[str] = None,
        lookback_years: int = 3
    ) -> Tuple[Optional[LocalSpatialAdapter], str]:
        """
        Tier 1 lookup (< 0.1 ms).
        Returns (adapter, "specialized") if ready.
        If uncached, enqueues async background warmup and returns (None, "warming_up").
        """
        key = self.get_grid_key(lat, lon)
        adapter = self._memory_cache.get(key)
        if adapter is not None:
            return adapter, "specialized"

        with self._lock:
            if key not in self._warmup_in_progress:
                self._warmup_in_progress.add(key)
                thread = threading.Thread(
                    target=self._async_fit_worker,
                    args=(lat, lon, station_name or f"Loc_{key}", lookback_years, key),
                    daemon=True
                )
                thread.start()
                logger.info("Enqueued async background specialization for %s [%s]", key, station_name)

        return None, "warming_up"

    def _async_fit_worker(self, lat: float, lon: float, station_name: str, lookback_years: int, key: str):
        try:
            self.fit_adapter_sync(lat=lat, lon=lon, lookback_years=lookback_years, station_name=station_name)
        except Exception as e:
            logger.error("Async spatial specialization failed for %s: %s", key, e, exc_info=True)
        finally:
            with self._lock:
                self._warmup_in_progress.discard(key)

    def fetch_historical_series(
        self,
        lat: float,
        lon: float,
        start_str: str,
        end_str: str,
        station_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Fetch NASA POWER historical daily series, caching in SQLite."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_observations (
                    station_key TEXT,
                    date_str TEXT,
                    t2m REAL,
                    t2m_max REAL,
                    t2m_min REAL,
                    rh2m REAL,
                    precip_mm REAL,
                    wind_speed REAL,
                    solar_rad REAL,
                    pressure_kpa REAL,
                    PRIMARY KEY (station_key, date_str)
                )
            """)
            conn.commit()

            cursor = conn.cursor()
            cursor.execute("""
                SELECT date_str, t2m, t2m_max, t2m_min, rh2m, precip_mm, wind_speed, solar_rad, pressure_kpa
                FROM daily_observations
                WHERE station_key = ? AND date_str >= ? AND date_str <= ?
                ORDER BY date_str ASC
            """, (station_name, start_str, end_str))
            rows = cursor.fetchall()

            d_start = datetime.strptime(start_str, "%Y%m%d")
            d_end = datetime.strptime(end_str, "%Y%m%d")
            expected_days = (d_end - d_start).days + 1

            if len(rows) >= expected_days * 0.90:
                result = {
                    "T2M": {}, "T2M_MAX": {}, "T2M_MIN": {}, "RH2M": {},
                    "PRECTOTCORR": {}, "WS10M": {}, "ALLSKY_SFC_SW_DWN": {}, "PS": {}
                }
                for r in rows:
                    ds, t, tmax, tmin, rh, p, w, sol, ps = r
                    result["T2M"][ds] = t
                    result["T2M_MAX"][ds] = tmax
                    result["T2M_MIN"][ds] = tmin
                    result["RH2M"][ds] = rh
                    result["PRECTOTCORR"][ds] = p
                    result["WS10M"][ds] = w
                    result["ALLSKY_SFC_SW_DWN"][ds] = sol
                    result["PS"][ds] = ps
                return result

        # Fetch from NASA POWER API
        logger.info("Fetching NASA POWER daily data for %s (%s to %s)...", station_name, start_str, end_str)
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS10M,WD10M,ALLSKY_SFC_SW_DWN,PS",
            "community": "AG",
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "start": start_str,
            "end": end_str,
            "format": "JSON"
        }
        resp = requests.get(url, params=params, timeout=40)
        if resp.status_code != 200:
            raise RuntimeError(f"NASA POWER API error HTTP {resp.status_code} for ({lat}, {lon})")

        data = resp.json()["properties"]["parameter"]
        dates = sorted(list(data["T2M"].keys()))

        with sqlite3.connect(self.db_path) as conn:
            for d in dates:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_observations (
                        station_key, date_str, t2m, t2m_max, t2m_min, rh2m,
                        precip_mm, wind_speed, solar_rad, pressure_kpa
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    station_name, d,
                    float(data["T2M"].get(d, 25.0)),
                    float(data["T2M_MAX"].get(d, 30.0)),
                    float(data["T2M_MIN"].get(d, 20.0)),
                    float(data["RH2M"].get(d, 60.0)),
                    float(data["PRECTOTCORR"].get(d, 0.0)),
                    float(data["WS10M"].get(d, 2.5)),
                    float(data["ALLSKY_SFC_SW_DWN"].get(d, 15.0)),
                    float(data["PS"].get(d, 99.0))
                ))
            conn.commit()

        return data

    def fit_adapter_sync(
        self,
        lat: float,
        lon: float,
        lookback_years: int = 3,
        station_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tier 2 explicit / synchronous adaptation (~2–5 seconds execution).
        Trains Ridge residual head + monthly climatology + conformal quantile.
        """
        t_start = time.perf_counter()
        grid_key = self.get_grid_key(lat, lon)
        name = station_name or f"Loc_{grid_key}"

        # 1. Determine training date window (strictly historical, e.g. 2020 to 2022)
        end_dt = datetime(2022, 12, 31, tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_years * 365 + 10)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        hist_data = self.fetch_historical_series(lat, lon, start_str, end_str, name)
        t2m_series = hist_data["T2M"]
        rh_series = hist_data.get("RH2M", {})
        ps_series = hist_data.get("PS", {})

        dates = sorted(list(t2m_series.keys()))
        if len(dates) < 300:
            raise ValueError(f"Insufficient historical samples ({len(dates)}) for {grid_key}")

        # 2. Compute 12-month local climatology
        monthly_temps: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        for d in dates:
            m = int(d[4:6])
            monthly_temps[m].append(float(t2m_series[d]))

        monthly_means = [float(np.mean(monthly_temps[m])) if monthly_temps[m] else 25.0 for m in range(1, 13)]
        monthly_stds = [float(np.std(monthly_temps[m])) if monthly_temps[m] else 3.0 for m in range(1, 13)]
        monthly_clim = {"means": monthly_means, "stds": monthly_stds}

        # 3. Build training matrix for residual correction
        # Base model proxy on general coordinate:
        # T_base = 0.70 * T_prev + 0.30 * Clim(month)
        X_list = []
        y_true_list = []
        base_preds = []

        for i in range(2, len(dates) - 1):
            d_curr = dates[i]
            d_prev1 = dates[i - 1]
            d_prev2 = dates[i - 2]
            d_next = dates[i + 1]

            t_next = float(t2m_series[d_next])
            t_curr = float(t2m_series[d_curr])
            t_prev1 = float(t2m_series[d_prev1])
            t_prev2 = float(t2m_series[d_prev2])
            rh_curr = float(rh_series.get(d_curr, 60.0))
            ps_curr = float(ps_series.get(d_curr, 101.3)) * 10.0  # kPa to hPa

            dt_curr = datetime.strptime(d_curr, "%Y%m%d")
            doy = dt_curr.timetuple().tm_yday
            month = dt_curr.month

            # Base predictor
            clim_m = monthly_means[month - 1]
            base_t = 0.72 * t_curr + 0.28 * clim_m

            # Extract normalized feature vector
            clim_anom = t_curr - clim_m
            diurnal_mom = t_curr - t_prev1
            sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
            cos_doy = math.cos(2.0 * math.pi * doy / 365.25)
            rh_norm = (rh_curr - 50.0) / 25.0
            p_norm = (ps_curr - 1013.25) / 10.0

            feats = [base_t, t_curr, diurnal_mom, clim_anom, sin_doy, cos_doy, rh_norm, p_norm]
            X_list.append(feats)
            y_true_list.append(t_next)
            base_preds.append(base_t)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_true_list, dtype=np.float64)
        y_base = np.array(base_preds, dtype=np.float64)
        r = y - y_base  # Residual to learn

        # 4. Train / Val split (75% / 25%)
        n_samples = len(y)
        split_idx = int(n_samples * 0.75)

        X_train, X_val = X[:split_idx], X[split_idx:]
        r_train, r_val = r[:split_idx], r[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        y_base_train, y_base_val = y_base[:split_idx], y_base[split_idx:]

        # 5. Fit Ridge Regression on residuals: w = (X^T X + lambda I)^-1 X^T r
        # Add intercept column
        X_train_ext = np.column_stack([np.ones(len(X_train)), X_train])
        X_val_ext = np.column_stack([np.ones(len(X_val)), X_val])

        lambda_reg = 1.5
        reg_matrix = lambda_reg * np.eye(X_train_ext.shape[1])
        reg_matrix[0, 0] = 0.0  # Do not penalize intercept

        w_ext = np.linalg.solve(X_train_ext.T @ X_train_ext + reg_matrix, X_train_ext.T @ r_train)
        intercept = float(w_ext[0])
        weights = [float(x) for x in w_ext[1:]]

        # 6. Compute non-linear temperature bin residuals
        r_pred_train = X_train_ext @ w_ext
        r_pred_val = X_val_ext @ w_ext

        binned_residuals = {}
        for base_val, r_actual in zip(y_base_train, r_train):
            b_key = f"bin_{int(math.floor(base_val / 5.0) * 5)}"
            if b_key not in binned_residuals:
                binned_residuals[b_key] = []
            binned_residuals[b_key].append(float(r_actual))
        binned_corrections = {
            k: float(np.clip(np.mean(v) * 0.4, -1.5, 1.5)) for k, v in binned_residuals.items() if len(v) >= 5
        }

        # 7. Evaluate out-of-fold validation metrics & Split Conformal 90% Quantile
        y_val_adapted = y_base_val + r_pred_val
        val_errors = np.abs(y_val - y_val_adapted)
        train_mae = float(np.mean(np.abs(y_train - (y_base_train + r_pred_train))))
        val_mae = float(np.mean(val_errors))

        # Split conformal quantile with finite sample correction: (1 - alpha) * (1 + 1/n)
        alpha = 0.10
        n_val = len(val_errors)
        conformal_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_val))
        q90 = float(np.quantile(val_errors, conformal_level))
        # Bounded conformal interval [1.65, 3.20]
        q90 = float(np.clip(q90, 1.65, 3.20))

        fit_time = time.perf_counter() - t_start

        adapter = LocalSpatialAdapter(
            grid_key=grid_key,
            latitude=lat,
            longitude=lon,
            station_name=name,
            monthly_climatology=monthly_clim,
            ridge_weights=weights,
            ridge_intercept=intercept,
            binned_residuals=binned_corrections,
            conformal_quantile_90=round(q90, 2),
            sample_count=n_samples,
            train_mae_c=train_mae,
            val_mae_c=val_mae,
            fit_time_sec=fit_time
        )

        # Save to disk
        adapter_file = self.adapters_dir / f"{grid_key}.json"
        size_bytes = adapter.save_json(adapter_file)

        # Update in-memory cache
        self._memory_cache[grid_key] = adapter

        logger.info(
            "Specialized adapter fitted for %s: val_mae=%.3f°C, q90=±%.2f°C, size=%d bytes in %.2fs",
            name, val_mae, q90, size_bytes, fit_time
        )

        return {
            "status": "SUCCESS",
            "grid_key": grid_key,
            "station_name": name,
            "adapter_status": "specialized",
            "sample_count": n_samples,
            "training_time_sec": round(fit_time, 2),
            "train_mae_c": round(train_mae, 3),
            "val_mae_c": round(val_mae, 3),
            "conformal_quantile_90_c": round(q90, 2),
            "adapter_size_bytes": size_bytes
        }
