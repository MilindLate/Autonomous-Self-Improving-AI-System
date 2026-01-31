"""Stub architecture designer.

In full system this would orchestrate NAS; here it just logs calls.
"""

from typing import Dict, Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ArchitectureDesigner:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Stub ArchitectureDesigner initialized")

    async def propose_architecture(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[ArchitectureDesigner] propose_architecture for {task} (stub)")
        return {"name": f"stub_architecture_for_{task}", "constraints": constraints}
