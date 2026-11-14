"""
Abstract base class and caching utilities for model_go ingestion connectors.
"""

import abc
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from model_go.config import CACHE_DIR

logger = logging.getLogger(__name__)


class BaseIngestionConnector(abc.ABC):
    """Abstract connector interface for atmospheric and meteorological data providers."""

    def __init__(self, name: str, cache_ttl_seconds: int = 3600):
        self.name = name
        self.cache_ttl_seconds = cache_ttl_seconds
        self.last_sync_utc: Optional[datetime] = None
        self.last_latency_ms: float = 0.0

    @abc.abstractmethod
    def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch fresh observations or fields from the upstream source."""
        pass

    @abc.abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """Verify network connectivity and credential/authorization status."""
        pass

    def _get_cache_path(self, cache_key: str) -> Path:
        hashed = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        return CACHE_DIR / f"{self.name}_{hashed}.json"

    def read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = self._get_cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = data.get("_cached_at", 0)
            if (time.time() - cached_time) < self.cache_ttl_seconds:
                return data.get("payload")
        except Exception as e:
            logger.warning("Cache read error for %s: %s", self.name, e)
        return None

    def write_cache(self, cache_key: str, payload: Any):
        path = self._get_cache_path(cache_key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "_cached_at": time.time(),
                    "_cached_iso": datetime.now(timezone.utc).isoformat(),
                    "payload": payload
                }, f, indent=2)
        except Exception as e:
            logger.warning("Cache write error for %s: %s", self.name, e)
