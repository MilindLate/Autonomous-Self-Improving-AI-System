"""Stub graph database used by AutonomousAI.

Represents knowledge as a simple in-memory list so count_concepts works.
"""

from typing import Dict, Any, List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphDatabase:
    def __init__(self, config: Dict):
        self.config = config
        self._knowledge: List[Dict[str, Any]] = []
        logger.info("Stub GraphDatabase initialized")

    async def add_knowledge(self, item: Dict[str, Any]) -> None:
        self._knowledge.append(item)
        logger.info("[GraphDatabase] add_knowledge (stub)")

    async def count_concepts(self) -> int:
        # Very rough: one concept per item
        return len(self._knowledge)
