"""
Configuration management for model_go meteorological forecasting engine.
Supports dual profiles: 'server' (full ensemble & live sync) and 'edge' (sub-1MB lightweight).
"""

import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR

# Profile: 'server' (default) or 'edge'
PROFILE = os.getenv("MODEL_GO_PROFILE", "server").lower()
IS_EDGE = PROFILE == "edge"

# Storage paths
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained model paths
ALPHA_ROOT = BASE_DIR / "test_model_alpha"
if not ALPHA_ROOT.exists():
    fallback_alpha = Path("D:/projects/khula_panja/nasa/test_model_alpha")
    if fallback_alpha.exists():
        ALPHA_ROOT = fallback_alpha

STUDENT_CHECKPOINT_PATH = OUTPUTS_DIR / "student_model_trained.json"

# NASA API Endpoints
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_GPM_IMERG_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07"
NASA_GEOS_FP_BASE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
NASA_SMAP_BASE = "https://n5eil01u.ecs.nsidc.org/SMAP"

# Optional credentials (read from environment, never hardcoded)
NASA_EARTHDATA_TOKEN: Optional[str] = os.getenv("NASA_EARTHDATA_TOKEN")
NASA_EARTHDATA_USER: Optional[str] = os.getenv("NASA_EARTHDATA_USER")
NASA_EARTHDATA_PASS: Optional[str] = os.getenv("NASA_EARTHDATA_PASS")
IMD_API_KEY: Optional[str] = os.getenv("IMD_API_KEY")
IMD_STATION_FEED_URL: Optional[str] = os.getenv("IMD_STATION_FEED_URL", "https://mausam.imd.gov.in/api")

# Webhook & Telemetry Configuration
WEBHOOK_URLS = [url.strip() for url in os.getenv("MODEL_GO_WEBHOOKS", "").split(",") if url.strip()]
MQTT_BROKER = os.getenv("MODEL_GO_MQTT_BROKER", "")
MQTT_PORT = int(os.getenv("MODEL_GO_MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MODEL_GO_MQTT_TOPIC", "model_go/forecasts")

# Default Target
DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090
DEFAULT_ELEVATION_M = 216.0
DEFAULT_STATION_NAME = "Delhi (Safdarjung)"
