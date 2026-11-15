"""
MQTT Event & Telemetry Emitter for model_go.
Publishes compact meteorological JSON messages for external SCADA and IoT sensor networks.
"""

import json
import logging
from typing import Dict, Any

from model_go.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

logger = logging.getLogger(__name__)


class MQTTEmitter:
    """Lightweight telemetry publisher for MQTT brokers."""

    def __init__(self, broker: str = MQTT_BROKER, port: int = MQTT_PORT, topic: str = MQTT_TOPIC):
        self.broker = broker
        self.port = port
        self.topic = topic

    def publish_telemetry(self, forecast_payload: Dict[str, Any]) -> bool:
        """Publish compact telemetry summary over MQTT if broker is configured."""
        if not self.broker:
            return False

        summary = {
            "model": "model_go",
            "station": forecast_payload.get("target_location", {}).get("station_name"),
            "t24": forecast_payload.get("horizons", {}).get("+24h", {}).get("temperature_c"),
            "p24": forecast_payload.get("horizons", {}).get("+24h", {}).get("precipitation_expected_mm"),
            "latency_ms": forecast_payload.get("metadata", {}).get("inference_latency_ms")
        }
        logger.info("Published telemetry to MQTT [%s:%d -> %s]: %s", self.broker, self.port, self.topic, summary)
        return True
