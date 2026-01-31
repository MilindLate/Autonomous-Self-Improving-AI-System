"""Stub vector database used by AutonomousAI.

In the full system this would wrap Chroma, Pinecone, etc. For now it's just an
in-memory store so the rest of the code can run.
"""

from typing import Dict, Any, List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorDatabase:
    def __init__(self, config: Dict):
        self.config = config
        self._items: List[Dict[str, Any]] = []
        logger.info("Stub VectorDatabase initialized")

    async def add_knowledge(self, item: Dict[str, Any]) -> None:
        self._items.append(item)
        logger.info("[VectorDatabase] add_knowledge (stub)")
