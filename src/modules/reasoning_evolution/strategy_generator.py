"""Stub strategy generator.

Generates and evaluates reasoning strategies in a minimal way.
"""

from typing import Dict, Any, List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StrategyGenerator:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Stub StrategyGenerator initialized")

    async def design_algorithms(self, task: str, knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"[StrategyGenerator] design_algorithms task={task} (stub)")
        return [
            {"name": f"stub_algorithm_for_{task}", "details": "no-op algorithm"}
        ]

    async def measure_depth(self) -> float:
        logger.info("[StrategyGenerator] measure_depth -> 1.0 (stub)")
        return 1.0
