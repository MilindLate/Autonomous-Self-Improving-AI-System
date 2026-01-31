"""
Autonomous Self-Improving AI System - Main Entry Point

This is the main entry point for the autonomous AI system.
"""

import asyncio
import sys
import os
import logging
import signal
from pathlib import Path
from typing import Dict
import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.autonomous_ai import AutonomousAI, AutonomyLevel
from src.utils.logger import setup_logger
from src.interfaces.api_server import APIServer
from src.interfaces.dashboard import Dashboard

logger = setup_logger(__name__)


class AutonomousAISystem:
    """Main system controller"""
    
    def __init__(self):
        """Initialize the system"""
        logger.info("=" * 80)
        logger.info("🧠 AUTONOMOUS SELF-IMPROVING AI SYSTEM")
        logger.info("=" * 80)
        
        # Load environment
        load_dotenv()
        
        # Load configuration
        self.config = self.load_configuration()
        
        # Initialize core AI
        self.ai = AutonomousAI(self.config)
        
        # Initialize interfaces
        self.api_server = None
        self.dashboard = None
        
        if self.config.get("api_enabled", True):
            self.api_server = APIServer(self.ai, self.config)
        
        if self.config.get("dashboard_enabled", True):
            self.dashboard = Dashboard(self.ai, self.config)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("✅ System initialized successfully")
    
    def load_configuration(self) -> Dict:
        """Load system configuration"""
        config_file = Path("config/system_config.yaml")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning("Config file not found, using defaults")
            config = self.get_default_config()
        
        # Override with environment variables
        config = self.apply_env_overrides(config)
        
        return config
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "autonomy_level": "supervised",
            "autonomous_learning": True,
            "self_modification_enabled": True,
            "api_enabled": True,
            "dashboard_enabled": True,
            "llm_provider": "openai",
            "llm_model": "gpt-4-turbo-preview"
        }
    
    def apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides"""
        
        # LLM configuration
        if os.getenv("PRIMARY_LLM_PROVIDER"):
            config["llm_provider"] = os.getenv("PRIMARY_LLM_PROVIDER")
        
        if os.getenv("PRIMARY_LLM_MODEL"):
            config["llm_model"] = os.getenv("PRIMARY_LLM_MODEL")
        
        if os.getenv("OPENAI_API_KEY"):
            config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            config["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        # System configuration
        if os.getenv("AUTONOMY_LEVEL"):
            config["autonomy_level"] = os.getenv("AUTONOMY_LEVEL")
        
        if os.getenv("AUTONOMOUS_LEARNING"):
            config["autonomous_learning"] = os.getenv("AUTONOMOUS_LEARNING").lower() == "true"
        
        if os.getenv("SELF_MODIFICATION_ENABLED"):
            config["self_modification_enabled"] = os.getenv("SELF_MODIFICATION_ENABLED").lower() == "true"
        
        return config
    
    async def start(self):
        """Start the autonomous AI system"""
        try:
            logger.info("🚀 Starting Autonomous AI System...")
            
            # Display system status
            self.display_status()
            
            # Start components in parallel
            tasks = []
            
            # Start core AI
            tasks.append(self.ai.start())
            
            # Start API server
            if self.api_server:
                tasks.append(self.api_server.start())
            
            # Start dashboard
            if self.dashboard:
                tasks.append(self.dashboard.start())
            
            # Run all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"❌ Critical error starting system: {e}")
            await self.shutdown()
    
    def display_status(self):
        """Display system status"""
        logger.info("\n" + "=" * 80)
        logger.info("SYSTEM STATUS")
        logger.info("=" * 80)
        logger.info(f"Version: {self.ai.version}")
        logger.info(f"Autonomy Level: {self.ai.autonomy_level.value}")
        logger.info(f"Autonomous Learning: {self.config.get('autonomous_learning', False)}")
        logger.info(f"Self-Modification: {self.config.get('self_modification_enabled', False)}")
        logger.info(f"LLM Provider: {self.config.get('llm_provider', 'unknown')}")
        logger.info(f"LLM Model: {self.config.get('llm_model', 'unknown')}")
        logger.info("=" * 80 + "\n")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"\n⚠️ Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Autonomous AI System...")
        
        try:
            # Stop API server
            if self.api_server:
                await self.api_server.stop()
            
            # Stop dashboard
            if self.dashboard:
                await self.dashboard.stop()
            
            # Stop core AI (saves state)
            if self.ai:
                await self.ai.emergency_stop()
            
            logger.info("✅ Shutdown complete")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        finally:
            sys.exit(0)


async def main():
    """Main entry point"""
    
    # Create and start system
    system = AutonomousAISystem()
    await system.start()


if __name__ == "__main__":
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
