"""Stub API server.

This provides the APIServer class expected by main.py but does not start a
real HTTP server yet. It just keeps an async loop alive.
"""

import asyncio
from typing import Dict, Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class APIServer:
    def __init__(self, ai: Any, config: Dict):
        self.ai = ai
        self.config = config
        self._running = False

    async def start(self) -> None:
        logger.info("[APIServer] Stub API server starting (no real HTTP endpoints yet)")
        self._running = True
        while self._running:
            await asyncio.sleep(3600)

    async def stop(self) -> None:
        logger.info("[APIServer] Stub API server stopping")
        self._running = False
