"""Stub metrics collector.

Provides minimal async methods so AutonomousAI can record and analyze metrics
without requiring full monitoring infrastructure.
"""

from typing import Dict, List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    def __init__(self, config: Dict):
        self.config = config
        self._history: List[Dict] = []
        logger.info("Stub MetricsCollector initialized")

    async def store(self, metrics: Dict) -> None:
        self._history.append(metrics)
        logger.info(f"[MetricsCollector] stored metrics: {metrics.get('timestamp', '')}")

    async def analyze_trends(self, history: List) -> None:
        logger.info(f"[MetricsCollector] analyze_trends called on {len(history)} points (stub)")

    async def detect_anomalies(self, metrics: Dict) -> List[str]:
        # Always return empty anomaly list for now
        logger.info("[MetricsCollector] detect_anomalies called (stub)")
        return []
