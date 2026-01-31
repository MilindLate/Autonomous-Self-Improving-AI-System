"""Stub dashboard implementation.

Represents a monitoring UI in the full system; here it's just an async loop.
"""

import asyncio
from typing import Dict, Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Dashboard:
    def __init__(self, ai: Any, config: Dict):
        self.ai = ai
        self.config = config
        self._running = False

    async def start(self) -> None:
        logger.info("[Dashboard] Stub dashboard starting (no web UI)")
        self._running = True
        while self._running:
            await asyncio.sleep(3600)

    async def stop(self) -> None:
        logger.info("[Dashboard] Stub dashboard stopping")
        self._running = False
