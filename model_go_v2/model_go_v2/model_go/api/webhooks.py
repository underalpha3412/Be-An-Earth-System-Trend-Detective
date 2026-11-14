"""
Reliable Webhook Event Dispatcher for model_go with Exponential Backoff & Dead-Letter Queue.
Pushes instant JSON alert payloads to external emergency services when severity escalates
to ORANGE or EXTREME_RED_ALERT, with automated retry logic and dead-letter persistence.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

from model_go.config import WEBHOOK_URLS, DATA_DIR
from model_go.schemas import UnifiedAlert

logger = logging.getLogger(__name__)

DEAD_LETTER_PATH = DATA_DIR / "webhook_dead_letter.json"


class WebhookDispatcher:
    """Dispatches emergency alerts with exponential backoff and dead-letter queue logging."""

    def __init__(
        self,
        target_urls: Optional[List[str]] = None,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.5,
        dead_letter_file: Path = DEAD_LETTER_PATH
    ):
        self.target_urls = target_urls if target_urls is not None else WEBHOOK_URLS
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.dead_letter_file = dead_letter_file

    def dispatch_alert_if_escalated(self, alerts: List[UnifiedAlert]) -> Dict[str, int]:
        """
        Check if any alert is ORANGE or EXTREME_RED_ALERT and push immediately.
        Executes exponential backoff on transient errors and writes to dead-letter queue if failed.
        """
        escalated = [a for a in alerts if a.level.value in ("ORANGE", "EXTREME_RED_ALERT") or str(a.level) in ("ORANGE", "EXTREME_RED_ALERT")]
        if not escalated or not self.target_urls:
            return {"attempted": 0, "successful": 0, "dead_lettered": 0}

        payload = {
            "event": "METEOROLOGICAL_ALERT_ESCALATION",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "escalated_alerts": [a.model_dump(mode="json") for a in escalated],
            "count": len(escalated)
        }

        successful = 0
        dead_lettered = 0

        for url in self.target_urls:
            delivered = False
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.post(url, json=payload, timeout=3.0)
                    if resp.status_code in (200, 201, 202):
                        delivered = True
                        successful += 1
                        logger.info("Delivered alert webhook to %s (attempt %d)", url, attempt)
                        break
                    else:
                        logger.warning("Webhook endpoint %s returned HTTP %d", url, resp.status_code)
                except Exception as e:
                    logger.warning("Webhook attempt %d failed for %s: %s", attempt, url, e)

                if attempt < self.max_retries:
                    time.sleep(self.initial_backoff_sec * (2 ** (attempt - 1)))

            if not delivered:
                dead_lettered += 1
                self._record_dead_letter(url, payload)

        return {"attempted": len(self.target_urls), "successful": successful, "dead_lettered": dead_lettered}

    def _record_dead_letter(self, url: str, payload: Dict[str, Any]):
        """Persist undelivered payload to dead-letter storage for post-incident audit."""
        try:
            self.dead_letter_file.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if self.dead_letter_file.exists():
                try:
                    with open(self.dead_letter_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []

            existing.append({
                "target_url": url,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "payload": payload
            })

            # Keep latest 100 dead letters
            existing = existing[-100:]
            with open(self.dead_letter_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            logger.error("Logged undeliverable webhook to dead-letter queue: %s", url)
        except Exception as e:
            logger.critical("Failed writing to webhook dead-letter queue: %s", e)
