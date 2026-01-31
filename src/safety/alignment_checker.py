"""Stub alignment checker.

Always reports aligned for now so the system can run.
"""

from dataclasses import dataclass
from typing import Dict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AlignmentStatus:
    is_aligned: bool
    details: str = ""


class AlignmentChecker:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Stub AlignmentChecker initialized")

    async def verify_alignment(self) -> AlignmentStatus:
        logger.info("[AlignmentChecker] verify_alignment -> aligned (stub)")
        return AlignmentStatus(is_aligned=True, details="stub aligned")

    async def check_plan(self, plan: Dict) -> bool:
        logger.info(f"[AlignmentChecker] check_plan for {plan.get('name', 'unknown')} -> True (stub)")
        return True
