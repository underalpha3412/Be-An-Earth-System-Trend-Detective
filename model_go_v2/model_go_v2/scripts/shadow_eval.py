"""
7-Day Live Shadow Evaluation Harness for model_go.
Logs daily shadow forecasts for Ghaziabad (28.6692°N, 77.4538°E) and Safdarjung (28.6139°N, 77.2090°E)
into SQLite/Parquet, and automatically scores predictions against ground truth after +24h, +48h, and +72h.
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from model_go import Forecaster
from model_go.schemas import ForecastPointRequest
from model_go.ingestion.nasa.power_client import NASAPowerConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shadow_eval")

SHADOW_DB_PATH = WORKSPACE_ROOT / "data" / "shadow_eval.sqlite3"

SHADOW_STATIONS = [
    {"name": "Delhi (Safdarjung)", "latitude": 28.6139, "longitude": 77.2090},
    {"name": "Ghaziabad (Hindon)", "latitude": 28.6692, "longitude": 77.4538},
]


class ShadowEvaluationHarness:
    """Manages continuous shadow forecast logging and automated lagging ground truth scoring."""

    def __init__(self, db_path: Path = SHADOW_DB_PATH):
        self.db_path = db_path
        self.forecaster = Forecaster()
        self.power_client = NASAPowerConnector()
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    issued_at_utc TEXT,
                    lead_time_hours INTEGER,
                    valid_time_utc TEXT,
                    pred_t2m_c REAL,
                    pred_t_lower_90 REAL,
                    pred_t_upper_90 REAL,
                    pred_precip_mm REAL,
                    pred_rain_prob REAL,
                    actual_t2m_c REAL DEFAULT NULL,
                    actual_precip_mm REAL DEFAULT NULL,
                    t2m_abs_error REAL DEFAULT NULL,
                    is_within_90ci INTEGER DEFAULT NULL,
                    scored_at_utc TEXT DEFAULT NULL
                )
            """)
            conn.commit()

    def log_daily_shadow_forecasts(self) -> List[Dict[str, Any]]:
        """Issue and record forward forecasts (+24h, +48h, +72h) for monitored stations."""
        now_utc = datetime.now(timezone.utc)
        records_logged = []

        with sqlite3.connect(self.db_path) as conn:
            for stn in SHADOW_STATIONS:
                logger.info("Generating shadow forecast for %s...", stn["name"])
                res = self.forecaster.predict_point(
                    lat=stn["latitude"],
                    lon=stn["longitude"],
                    station_name=stn["name"]
                )

                horizons = res.get("horizons", {})
                for lead_h, h_key in [(24, "+24h"), (48, "+48h"), (72, "+72h")]:
                    if h_key in horizons:
                        h = horizons[h_key]
                        valid_dt = now_utc + timedelta(hours=lead_h)
                        t_bounds = h.get("temp_uncertainty_90pct", (h["temperature_c"] - 1.5, h["temperature_c"] + 1.5))

                        conn.execute("""
                            INSERT INTO shadow_forecasts (
                                station_name, latitude, longitude, issued_at_utc,
                                lead_time_hours, valid_time_utc, pred_t2m_c,
                                pred_t_lower_90, pred_t_upper_90, pred_precip_mm,
                                pred_rain_prob
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            stn["name"], stn["latitude"], stn["longitude"],
                            now_utc.isoformat(), lead_h, valid_dt.isoformat(),
                            h["temperature_c"], t_bounds[0], t_bounds[1],
                            h.get("precipitation_expected_mm", 0.0),
                            h.get("rain_probability", 0.0)
                        ))
                        records_logged.append({
                            "station": stn["name"],
                            "lead_h": lead_h,
                            "valid_time": valid_dt.isoformat(),
                            "pred_t": h["temperature_c"]
                        })

            conn.commit()
        logger.info("Successfully recorded %d shadow forecasts into %s", len(records_logged), self.db_path)
        return records_logged

    def auto_score_unverified_records(self) -> Dict[str, Any]:
        """Check for elapsed forecasts and score against newly arrived NASA ground truth."""
        now_utc = datetime.now(timezone.utc)
        scored_count = 0
        total_err_24, total_err_48, total_err_72 = [], [], []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Find unscored records whose valid time is in the past by at least 24h
            cursor.execute("""
                SELECT id, station_name, latitude, longitude, valid_time_utc, lead_time_hours,
                       pred_t2m_c, pred_t_lower_90, pred_t_upper_90, pred_precip_mm
                FROM shadow_forecasts
                WHERE actual_t2m_c IS NULL AND valid_time_utc <= ?
            """, ((now_utc - timedelta(hours=24)).isoformat(),))

            rows = cursor.fetchall()
            for row in rows:
                rec_id, stn_name, lat, lon, v_time_str, lead_h, p_t, p_lo, p_hi, p_rain = row
                v_dt = datetime.fromisoformat(v_time_str)
                v_date_str = v_dt.strftime("%Y%m%d")

                # Fetch verified observation from NASA POWER
                try:
                    obs_data = self.power_client.fetch_data(lat, lon, days_back=7)
                    params = obs_data.get("parameters", {})
                    t2m_series = params.get("T2M", {})
                    rain_series = params.get("PRECTOTCORR", {})

                    if v_date_str in t2m_series:
                        actual_t = float(t2m_series[v_date_str])
                        actual_p = float(rain_series.get(v_date_str, 0.0))
                        abs_err = abs(p_t - actual_t)
                        in_ci = 1 if (p_lo <= actual_t <= p_hi) else 0

                        conn.execute("""
                            UPDATE shadow_forecasts
                            SET actual_t2m_c = ?, actual_precip_mm = ?,
                                t2m_abs_error = ?, is_within_90ci = ?,
                                scored_at_utc = ?
                            WHERE id = ?
                        """, (actual_t, actual_p, abs_err, in_ci, now_utc.isoformat(), rec_id))

                        scored_count += 1
                        if lead_h == 24: total_err_24.append(abs_err)
                        elif lead_h == 48: total_err_48.append(abs_err)
                        elif lead_h == 72: total_err_72.append(abs_err)
                except Exception as e:
                    logger.debug("Ground truth not yet available for %s on %s: %s", stn_name, v_date_str, e)

            conn.commit()

        summary = {
            "records_scored_in_cycle": scored_count,
            "mae_24h": round(float(sum(total_err_24) / len(total_err_24)), 3) if total_err_24 else None,
            "mae_48h": round(float(sum(total_err_48) / len(total_err_48)), 3) if total_err_48 else None,
            "mae_72h": round(float(sum(total_err_72) / len(total_err_72)), 3) if total_err_72 else None,
        }
        logger.info("Shadow verification complete: %s", summary)
        return summary

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate high-level scorecard across all recorded shadow evaluations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(t2m_abs_error), AVG(is_within_90ci) FROM shadow_forecasts WHERE actual_t2m_c IS NOT NULL")
            total_scored, avg_mae, ci_coverage = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts WHERE actual_t2m_c IS NULL")
            pending_count = cursor.fetchone()[0]

            return {
                "total_scored_forecasts": total_scored or 0,
                "pending_validation_count": pending_count or 0,
                "overall_t2m_mae_c": round(float(avg_mae), 3) if avg_mae else None,
                "ci_90pct_coverage": round(float(ci_coverage * 100.0), 1) if ci_coverage else None,
                "database_path": str(self.db_path)
            }


def main():
    parser = argparse.ArgumentParser(description="model_go 7-Day Live Shadow Evaluation Harness")
    parser.add_argument("--issue", action="store_true", help="Issue and log shadow forecasts for today")
    parser.add_argument("--score", action="store_true", help="Score pending forecasts against newly available ground truth")
    parser.add_argument("--summary", action="store_true", help="Print summary scorecard")

    args = parser.parse_args()
    harness = ShadowEvaluationHarness()

    if args.issue:
        records = harness.log_daily_shadow_forecasts()
        print(f"\n[+] Issued {len(records)} shadow forecasts for Safdarjung & Ghaziabad.")
    elif args.score:
        res = harness.auto_score_unverified_records()
        print(f"\n[+] Scoring Cycle Complete: {json.dumps(res, indent=2)}")
    elif args.summary:
        card = harness.get_summary_report()
        print("\n=======================================================")
        print(" MODEL_GO 7-DAY SHADOW EVALUATION SCORECARD")
        print("=======================================================")
        print(f" Total Scored Forecasts:     {card['total_scored_forecasts']}")
        print(f" Pending Validation:         {card['pending_validation_count']}")
        print(f" Overall Temperature MAE:    {card['overall_t2m_mae_c']} °C" if card['overall_t2m_mae_c'] else " Overall Temperature MAE:    Awaiting verification")
        print(f" 90% CI Calibration Coverage: {card['ci_90pct_coverage']} %" if card['ci_90pct_coverage'] else " 90% CI Calibration Coverage: Awaiting verification")
        print(f" Database:                   {card['database_path']}")
        print("=======================================================\n")
    else:
        # Default: run issue and print summary
        harness.log_daily_shadow_forecasts()
        card = harness.get_summary_report()
        print(f"\nShadow forecast cycle executed. Database: {card['database_path']}\n")


if __name__ == "__main__":
    main()
