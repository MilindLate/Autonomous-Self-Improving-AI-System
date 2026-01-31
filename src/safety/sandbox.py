"""Stub sandbox implementation.

Real implementation would isolate and safely execute code. Here we just log
calls and return benign results.
"""

from dataclasses import dataclass
from typing import Any, Dict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SandboxResult:
    performance: float = 1.0
    performance_gain: float = 0.0
    area: str = "generic"
    details: str = "stub result"


class Sandbox:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Stub Sandbox initialized")

    async def execute(self, plan: Dict) -> SandboxResult:
        logger.info(f"[Sandbox] execute plan: {plan.get('name', 'unknown')} (stub)")
        return SandboxResult()

    async def test_algorithm(self, algorithm: Any) -> SandboxResult:
        logger.info("[Sandbox] test_algorithm called (stub)")
        return SandboxResult()

    async def verify_integrity(self) -> bool:
        logger.info("[Sandbox] verify_integrity -> True (stub)")
        return True
