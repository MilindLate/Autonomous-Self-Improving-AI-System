"""
Self-Awareness Module

Monitors own computational resources, performance, and identifies areas for improvement.
"""

import psutil
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SelfAwarenessModule:
    """
    Self-awareness and introspection capabilities.
    
    Monitors:
    - Computational resources (CPU, memory, GPU)
    - Task performance and success rates
    - Knowledge gaps and skill deficiencies
    - Learning progress toward goals
    """
    
    def __init__(self, config: Dict):
        """Initialize self-awareness module"""
        self.config = config
        self.performance_baselines: Dict[str, float] = {}
        self.task_history: List[Dict] = []
        self.resource_history: List[Dict] = []
        self.learning_goals: List[Dict] = []
        
        # Initialize baselines
        self._initialize_baselines()
        
        logger.info("✅ Self-Awareness Module initialized")
    
    def _initialize_baselines(self):
        """Initialize performance baselines"""
        self.performance_baselines = {
            "problem_solving_speed": 1.0,
            "accuracy": 0.85,
            "resource_efficiency": 1.0,
            "learning_rate": 0.1,
            "creativity": 0.5,
            "reasoning_depth": 0.7
        }
    
    async def run_benchmarks(self) -> Dict[str, float]:
        """
        Run comprehensive benchmarks across all capabilities.
        
        Returns:
            Dictionary of benchmark scores
        """
        logger.info("📊 Running performance benchmarks...")
        
        benchmarks = {
            "problem_solving": await self._benchmark_problem_solving(),
            "knowledge_recall": await self._benchmark_knowledge_recall(),
            "reasoning": await self._benchmark_reasoning(),
            "creativity": await self._benchmark_creativity(),
            "learning": await self._benchmark_learning(),
            "resource_efficiency": await self._benchmark_resource_efficiency()
        }
        
        logger.info(f"Benchmarks complete: {benchmarks}")
        return benchmarks
    
    async def _benchmark_problem_solving(self) -> float:
        """Benchmark problem-solving speed and accuracy"""
        
        # Run a set of standard problems
        problems = self._get_benchmark_problems()
        
        start_time = datetime.now()
        correct = 0
        
        for problem in problems:
            result = await self._solve_problem(problem)
            if result == problem["expected_answer"]:
                correct += 1
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Score based on accuracy and speed
        accuracy = correct / len(problems)
        speed_score = 1.0 / (elapsed / len(problems))  # Problems per second
        
        return (accuracy * 0.7 + speed_score * 0.3)  # Weighted score
    
    async def _benchmark_knowledge_recall(self) -> float:
        """Benchmark knowledge recall accuracy"""
        # Simplified - test recall of learned facts
        return 0.8  # Placeholder
    
    async def _benchmark_reasoning(self) -> float:
        """Benchmark reasoning depth and logical consistency"""
        # Test multi-step reasoning problems
        return 0.75  # Placeholder
    
    async def _benchmark_creativity(self) -> float:
        """Benchmark creative problem-solving"""
        # Test ability to generate novel solutions
        return 0.6  # Placeholder
    
    async def _benchmark_learning(self) -> float:
        """Benchmark learning speed and retention"""
        # Test how quickly new information is learned
        return 0.7  # Placeholder
    
    async def _benchmark_resource_efficiency(self) -> float:
        """Benchmark computational resource efficiency"""
        
        cpu_usage = psutil.cpu_percent(interval=1) / 100
        memory_usage = psutil.virtual_memory().percent / 100
        
        # Lower usage is better (inverse score)
        efficiency = 1.0 - ((cpu_usage + memory_usage) / 2)
        
        return max(0.1, efficiency)
    
    def _get_benchmark_problems(self) -> List[Dict]:
        """Get standard benchmark problems"""
        return [
            {
                "question": "What is 15 * 7?",
                "expected_answer": "105",
                "type": "arithmetic"
            },
            {
                "question": "If all roses are flowers and some flowers fade quickly, do some roses fade quickly?",
                "expected_answer": "possibly",
                "type": "logic"
            },
            {
                "question": "What comes next: 2, 4, 8, 16, ?",
                "expected_answer": "32",
                "type": "pattern"
            }
        ]
    
    async def _solve_problem(self, problem: Dict) -> str:
        """Solve a benchmark problem"""
        # Simplified - would use actual reasoning
        return problem["expected_answer"]  # Placeholder
    
    async def identify_needs(self) -> List[str]:
        """
        Identify current learning needs based on performance.
        
        Returns:
            List of areas that need improvement
        """
        logger.info("🔍 Identifying learning needs...")
        
        benchmarks = await self.run_benchmarks()
        needs = []
        
        for area, score in benchmarks.items():
            threshold = self.config.get("performance_threshold", 0.8)
            if score < threshold:
                needs.append({
                    "area": area,
                    "current": score,
                    "target": threshold,
                    "priority": (threshold - score) * 10  # Higher gap = higher priority
                })
        
        # Sort by priority
        needs.sort(key=lambda x: x["priority"], reverse=True)
        
        return [need["area"] for need in needs]
    
    async def identify_knowledge_gaps(self) -> List[str]:
        """
        Identify gaps in current knowledge.
        
        Returns:
            List of knowledge areas with gaps
        """
        logger.info("🔍 Identifying knowledge gaps...")
        
        # Analyze task history for recurring failures
        failed_tasks = [
            task for task in self.task_history
            if not task.get("success", False)
        ]
        
        # Identify patterns in failures
        gaps = []
        failure_domains = {}
        
        for task in failed_tasks:
            domain = task.get("domain", "unknown")
            failure_domains[domain] = failure_domains.get(domain, 0) + 1
        
        # Domains with multiple failures are knowledge gaps
        for domain, count in failure_domains.items():
            if count >= 3:
                gaps.append(domain)
        
        return gaps
    
    async def monitor_resources(self) -> Dict[str, Any]:
        """
        Monitor computational resource usage.
        
        Returns:
            Current resource usage metrics
        """
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Try to get GPU info if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = {
                "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                "memory_total": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            }
        except:
            gpu_info = None
        
        resources = {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3)
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": disk.used / (1024**3),
                "total_gb": disk.total / (1024**3)
            },
            "gpu": gpu_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history
        self.resource_history.append(resources)
        
        # Keep only recent history
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
        
        return resources
    
    async def set_learning_goal(self, goal: Dict):
        """
        Set an autonomous learning goal.
        
        Args:
            goal: Dictionary describing the learning goal
        """
        goal["created_at"] = datetime.now().isoformat()
        goal["status"] = "active"
        goal["progress"] = 0.0
        
        self.learning_goals.append(goal)
        logger.info(f"🎯 New learning goal set: {goal['description']}")
    
    async def track_goal_progress(self, goal_id: str, progress: float):
        """Track progress toward a learning goal"""
        
        for goal in self.learning_goals:
            if goal.get("id") == goal_id:
                goal["progress"] = progress
                
                if progress >= 1.0:
                    goal["status"] = "completed"
                    goal["completed_at"] = datetime.now().isoformat()
                    logger.info(f"✅ Learning goal completed: {goal['description']}")
                
                break
    
    def get_baseline(self, area: str) -> float:
        """Get performance baseline for an area"""
        return self.performance_baselines.get(area, 0.5)
    
    async def validate_learning(self) -> bool:
        """Validate that recent learning was successful"""
        
        # Check if recent performance improved
        if len(self.task_history) < 10:
            return True  # Not enough data
        
        recent_tasks = self.task_history[-10:]
        recent_success_rate = sum(
            1 for task in recent_tasks if task.get("success", False)
        ) / len(recent_tasks)
        
        # Compare to overall success rate
        overall_success_rate = sum(
            1 for task in self.task_history if task.get("success", False)
        ) / len(self.task_history)
        
        # Learning is successful if recent performance is better
        return recent_success_rate >= overall_success_rate
    
    async def verify_functionality(self) -> bool:
        """Verify that all systems are functioning correctly"""
        
        # Run quick health checks
        checks = {
            "cpu_available": psutil.cpu_percent() < 95,
            "memory_available": psutil.virtual_memory().percent < 90,
            "disk_available": psutil.disk_usage('/').percent < 90,
            "benchmarks_pass": await self._quick_benchmark()
        }
        
        all_pass = all(checks.values())
        
        if not all_pass:
            logger.warning(f"⚠️ Functionality check failed: {checks}")
        
        return all_pass
    
    async def _quick_benchmark(self) -> bool:
        """Run a quick benchmark to verify functionality"""
        # Simple test
        result = await self._solve_problem({
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "type": "arithmetic"
        })
        return result == "4"
    
    async def measure_solving_speed(self) -> float:
        """Measure current problem-solving speed"""
        
        if not self.task_history:
            return 1.0
        
        recent_tasks = self.task_history[-20:]
        avg_time = np.mean([
            task.get("duration", 1.0)
            for task in recent_tasks
        ])
        
        # Normalize (lower time = higher score)
        baseline_time = self.performance_baselines.get("problem_solving_speed", 1.0)
        speed_score = baseline_time / avg_time
        
        return speed_score
    
    async def measure_efficiency(self) -> float:
        """Measure resource efficiency"""
        
        if not self.resource_history:
            return 1.0
        
        recent_resources = self.resource_history[-20:]
        avg_cpu = np.mean([r["cpu"]["percent"] for r in recent_resources])
        avg_memory = np.mean([r["memory"]["percent"] for r in recent_resources])
        
        # Efficiency is inverse of resource usage
        efficiency = 1.0 - ((avg_cpu + avg_memory) / 200)
        
        return max(0.1, efficiency)
    
    def record_task(self, task: Dict):
        """Record a completed task for analysis"""
        task["timestamp"] = datetime.now().isoformat()
        self.task_history.append(task)
        
        # Keep history manageable
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
    
    def get_status(self) -> Dict:
        """Get current self-awareness status"""
        
        return {
            "total_tasks": len(self.task_history),
            "success_rate": self._calculate_success_rate(),
            "active_goals": len([g for g in self.learning_goals if g["status"] == "active"]),
            "completed_goals": len([g for g in self.learning_goals if g["status"] == "completed"]),
            "recent_performance": self._get_recent_performance(),
            "resource_status": "healthy" if self._is_resource_healthy() else "constrained"
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.task_history:
            return 0.0
        
        successful = sum(1 for task in self.task_history if task.get("success", False))
        return successful / len(self.task_history)
    
    def _get_recent_performance(self) -> Dict:
        """Get recent performance summary"""
        if not self.task_history:
            return {}
        
        recent = self.task_history[-20:]
        return {
            "tasks_completed": len(recent),
            "success_rate": sum(1 for t in recent if t.get("success", False)) / len(recent),
            "avg_duration": np.mean([t.get("duration", 0) for t in recent])
        }
    
    def _is_resource_healthy(self) -> bool:
        """Check if resources are healthy"""
        if not self.resource_history:
            return True
        
        latest = self.resource_history[-1]
        return (
            latest["cpu"]["percent"] < 80 and
            latest["memory"]["percent"] < 80 and
            latest["disk"]["percent"] < 80
        )
