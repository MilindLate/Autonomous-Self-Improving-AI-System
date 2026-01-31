"""Stub solution generator.

Provides novelty and creativity scores without real models.
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SolutionGenerator:
    def __init__(self, config):
        self.config = config
        logger.info("Stub SolutionGenerator initialized")

    async def measure_novelty(self) -> float:
        logger.info("[SolutionGenerator] measure_novelty -> 0.5 (stub)")
        return 0.5

    async def measure_creativity(self) -> float:
        logger.info("[SolutionGenerator] measure_creativity -> 0.5 (stub)")
        return 0.5
