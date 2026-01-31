"""
Meta-Learning Core - Self-Improvement Engine

Handles neural architecture search, AutoML, and self-modifying capabilities.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ImprovementResult:
    """Result of an improvement attempt"""
    success: bool
    performance_gain: float
    metrics: Dict[str, float]
    details: str
    timestamp: datetime


class MetaLearningCore:
    """
    Meta-learning core for autonomous self-improvement.
    
    Capabilities:
    - Neural Architecture Search (NAS)
    - Automated Machine Learning (AutoML)
    - Self-modifying code generation
    - Performance optimization
    - Learning strategy evolution
    """
    
    def __init__(self, config: Dict):
        """Initialize meta-learning core"""
        self.config = config
        self.improvement_history: List[ImprovementResult] = []
        self.successful_strategies: List[Dict] = []
        self.failed_strategies: List[Dict] = []
        self.current_learning_rate = config.get("initial_learning_rate", 0.001)
        
        # Initialize LLM for code generation
        self.llm = self._initialize_llm(config)
        
        logger.info("✅ Meta-Learning Core initialized")
    
    def _initialize_llm(self, config: Dict):
        """Initialize LLM for reasoning and code generation.

        For now we support OpenAI's chat completions API using the modern
        `openai` client (>=1.0). If no key is configured, meta-learning will
        fall back to a stub mode and simply return empty JSON.
        """
        provider = config.get("llm_provider", "openai")

        if provider == "openai":
            try:
                from openai import AsyncOpenAI  # type: ignore
            except Exception as e:  # pragma: no cover - import-time failure
                logger.error(f"Failed to import AsyncOpenAI client: {e}")
                return None

            api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OPENAI_API_KEY configured; MetaLearningCore will run in stub mode.")
                return None

            client = AsyncOpenAI(api_key=api_key)
            return {"provider": "openai", "client": client}

        # Other providers are not wired up yet; run in stub mode.
        logger.warning(f"LLM provider '{provider}' not implemented in stub MetaLearningCore; running without LLM.")
        return None
    
    async def design_solution(
        self,
        problem: str,
        current_state: Dict,
        constraints: Dict
    ) -> Dict:
        """
        Design a solution to improve a specific problem area.
        
        Args:
            problem: Description of the problem to solve
            current_state: Current state and metrics
            constraints: Constraints for the solution
            
        Returns:
            Improvement plan with implementation details
        """
        logger.info(f"🎯 Designing solution for: {problem}")
        
        # 1. Analyze similar past improvements
        similar_cases = self._find_similar_improvements(problem)
        
        # 2. Generate solution using LLM (if available)
        solution = await self._generate_solution_with_llm(
            problem=problem,
            current_state=current_state,
            constraints=constraints,
            similar_cases=similar_cases
        )
        
        # 3. Refine solution
        refined_solution = await self._refine_solution(solution)
        
        # 4. Estimate impact
        estimated_impact = await self._estimate_impact(refined_solution)
        
        improvement_plan = {
            "name": f"improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "problem": problem,
            "solution": refined_solution,
            "estimated_impact": estimated_impact,
            "implementation_type": self._classify_implementation_type(refined_solution),
            "impact": self._assess_impact_level(estimated_impact),
            "constraints": constraints,
            "created_at": datetime.now().isoformat()
        }
        
        return improvement_plan
    
    async def _generate_solution_with_llm(
        self,
        problem: str,
        current_state: Dict,
        constraints: Dict,
        similar_cases: List[Dict]
    ) -> Dict:
        """Generate solution using LLM reasoning"""
        
        # Build context from similar cases
        context = self._build_context_from_cases(similar_cases)
        
        prompt = f"""You are an autonomous AI system capable of self-improvement.

Problem to solve: {problem}

Current state:
{json.dumps(current_state, indent=2)}

Constraints:
{json.dumps(constraints, indent=2)}

Similar past improvements (learn from these):
{context}

Design a detailed solution to improve performance in this area. Consider:
1. What specific changes are needed (code, algorithm, architecture)?
2. What is the implementation approach?
3. How can this be tested safely?
4. What are potential risks?
5. What metrics will show improvement?

Provide a structured solution in JSON format with:
{{
    "approach": "description of the approach",
    "changes": ["list of specific changes"],
    "implementation_steps": ["step 1", "step 2", ...],
    "testing_strategy": "how to test this",
    "success_metrics": ["metric 1", "metric 2", ...],
    "risks": ["risk 1", "risk 2", ...],
    "rollback_plan": "how to revert if needed"
}}
"""
        
        # Call LLM via API client; in this stubbed setup we don't support
        # local models, so if no client is configured we just return an
        # empty JSON object.
        response = await self._call_api_llm(prompt)
        
        # Parse JSON response
        try:
            solution = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from response
            solution = self._extract_json_from_text(response)
        
        return solution
    
    async def _call_api_llm(self, prompt: str) -> str:
        """Call API-based LLM.

        In this stubbed environment we only support OpenAI via AsyncOpenAI.
        If no client is configured, we simply return an empty JSON object
        so the rest of the pipeline can continue.
        """
        if not self.llm:
            logger.warning("MetaLearningCore._call_api_llm called without an LLM client; returning empty JSON.")
            return "{}"

        try:
            provider = self.llm.get("provider")
            if provider == "openai":
                client = self.llm["client"]
                response = await client.chat.completions.create(
                    model=self.config.get("llm_model", "gpt-4-turbo-preview"),
                    messages=[
                        {"role": "system", "content": "You are an autonomous self-improving AI system."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
                return response.choices[0].message.content

            logger.warning(f"LLM provider '{provider}' not supported in _call_api_llm; returning empty JSON.")
            return "{}"

        except Exception as e:  # pragma: no cover - runtime API failures
            logger.error(f"Error calling API LLM: {e}")
            return "{}"
    
    async def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM"""
        model = self.llm["model"]
        tokenizer = self.llm["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=2000, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """Extract JSON object from text response"""
        try:
            # Find JSON in text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        # Return default structure
        return {
            "approach": "unable to parse",
            "changes": [],
            "implementation_steps": [],
            "testing_strategy": "manual review",
            "success_metrics": [],
            "risks": ["unknown"],
            "rollback_plan": "restore from backup"
        }
    
    def _find_similar_improvements(self, problem: str) -> List[Dict]:
        """Find similar past improvements for learning"""
        similar = []
        
        # Simple similarity based on keywords (can be enhanced with embeddings)
        problem_keywords = set(problem.lower().split())
        
        for improvement in self.improvement_history:
            if improvement.success:
                # Check keyword overlap
                imp_keywords = set(str(improvement.details).lower().split())
                overlap = len(problem_keywords & imp_keywords)
                
                if overlap > 2:
                    similar.append({
                        "problem": improvement.details,
                        "performance_gain": improvement.performance_gain,
                        "metrics": improvement.metrics
                    })
        
        return similar[:5]  # Return top 5
    
    def _build_context_from_cases(self, cases: List[Dict]) -> str:
        """Build context string from similar cases"""
        if not cases:
            return "No similar past improvements found."
        
        context_parts = []
        for i, case in enumerate(cases):
            context_parts.append(f"""
Case {i+1}:
- Problem: {case['problem']}
- Performance gain: {case['performance_gain']:.2%}
- Metrics: {case['metrics']}
""")
        
        return "\n".join(context_parts)
    
    async def _refine_solution(self, solution: Dict) -> Dict:
        """Refine and validate solution design"""
        
        # Add validation checks
        refined = solution.copy()
        
        # Ensure all required fields exist
        required_fields = [
            "approach", "changes", "implementation_steps",
            "testing_strategy", "success_metrics", "risks", "rollback_plan"
        ]
        
        for field in required_fields:
            if field not in refined:
                refined[field] = []
        
        # Add safety checks
        refined["safety_checks"] = [
            "Verify alignment before implementation",
            "Test in sandbox environment",
            "Monitor resource usage",
            "Create backup before changes"
        ]
        
        return refined
    
    async def _estimate_impact(self, solution: Dict) -> Dict:
        """Estimate the impact of implementing a solution"""
        
        # Analyze solution complexity
        num_changes = len(solution.get("changes", []))
        num_steps = len(solution.get("implementation_steps", []))
        num_risks = len(solution.get("risks", []))
        
        # Calculate estimated impact scores
        complexity_score = min(1.0, (num_changes + num_steps) / 20)
        risk_score = min(1.0, num_risks / 10)
        
        # Estimate based on similar past improvements
        similar_gains = [
            imp.performance_gain
            for imp in self.improvement_history
            if imp.success
        ]
        
        avg_gain = np.mean(similar_gains) if similar_gains else 0.1
        
        return {
            "estimated_performance_gain": avg_gain * (1 - risk_score),
            "complexity": complexity_score,
            "risk_level": risk_score,
            "confidence": 0.7 if similar_gains else 0.3
        }
    
    def _classify_implementation_type(self, solution: Dict) -> str:
        """Classify the type of implementation"""
        changes = str(solution.get("changes", [])).lower()
        
        if "architecture" in changes or "model" in changes:
            return "architectural"
        elif "algorithm" in changes or "optimization" in changes:
            return "algorithmic"
        elif "code" in changes or "refactor" in changes:
            return "code_improvement"
        elif "parameter" in changes or "hyperparameter" in changes:
            return "parameter_tuning"
        else:
            return "general"
    
    def _assess_impact_level(self, estimated_impact: Dict) -> str:
        """Assess the impact level of a change"""
        complexity = estimated_impact.get("complexity", 0)
        risk = estimated_impact.get("risk_level", 0)
        
        impact_score = (complexity + risk) / 2
        
        if impact_score > 0.7:
            return "critical"
        elif impact_score > 0.5:
            return "high"
        elif impact_score > 0.3:
            return "medium"
        else:
            return "low"
    
    async def learn_from_failure(self, plan: Dict, result: Any):
        """
        Learn from a failed improvement attempt.
        
        Args:
            plan: The improvement plan that failed
            result: Result details from the failure
        """
        logger.info(f"📚 Learning from failure: {plan['name']}")
        
        failure_record = {
            "plan": plan,
            "result": str(result),
            "timestamp": datetime.now().isoformat(),
            "lessons": await self._extract_lessons_from_failure(plan, result)
        }
        
        self.failed_strategies.append(failure_record)
        
        # Update improvement history
        improvement_result = ImprovementResult(
            success=False,
            performance_gain=-1.0,
            metrics={},
            details=f"Failed: {plan['problem']}",
            timestamp=datetime.now()
        )
        
        self.improvement_history.append(improvement_result)
        
        # Adjust learning strategy
        await self._adjust_learning_strategy(failure_record)
    
    async def _extract_lessons_from_failure(self, plan: Dict, result: Any) -> List[str]:
        """Extract lessons from a failure"""
        
        prompt = f"""Analyze this failed improvement attempt and extract lessons:

Plan:
{json.dumps(plan, indent=2)}

Result:
{str(result)}

What went wrong and what can be learned? Provide a list of specific lessons.
"""
        
        response = await self._call_api_llm(prompt)
        
        # Extract lessons (simplified - could be more sophisticated)
        lessons = [
            line.strip()
            for line in response.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
        
        return lessons[:5]  # Top 5 lessons
    
    async def _adjust_learning_strategy(self, failure_record: Dict):
        """Adjust learning strategy based on failures"""
        
        # If too many recent failures, reduce learning rate
        recent_failures = [
            imp for imp in self.improvement_history[-10:]
            if not imp.success
        ]
        
        if len(recent_failures) > 7:  # More than 70% failure rate
            self.current_learning_rate *= 0.5
            logger.warning(f"⚠️ High failure rate detected. Reducing learning rate to {self.current_learning_rate}")
        
        # If few failures, can be more aggressive
        elif len(recent_failures) < 2:
            self.current_learning_rate = min(
                self.current_learning_rate * 1.2,
                self.config.get("max_learning_rate", 0.01)
            )
            logger.info(f"✅ Low failure rate. Increasing learning rate to {self.current_learning_rate}")
    
    async def measure_learning_rate(self) -> float:
        """Measure current learning rate"""
        return self.current_learning_rate
    
    async def evolve_learning_strategy(self) -> Dict:
        """
        Evolve the meta-learning strategy itself.
        
        This allows the AI to improve how it learns.
        """
        logger.info("🧬 Evolving learning strategy...")
        
        # Analyze what strategies have worked best
        successful_patterns = self._analyze_successful_patterns()
        
        # Generate new strategies
        new_strategies = await self._generate_new_strategies(successful_patterns)
        
        # Test strategies
        best_strategy = await self._test_strategies(new_strategies)
        
        return best_strategy
    
    def _analyze_successful_patterns(self) -> Dict:
        """Analyze patterns in successful improvements"""
        
        successful = [imp for imp in self.improvement_history if imp.success]
        
        if not successful:
            return {}
        
        patterns = {
            "avg_performance_gain": np.mean([imp.performance_gain for imp in successful]),
            "best_performance_gain": max([imp.performance_gain for imp in successful]),
            "success_rate": len(successful) / len(self.improvement_history),
            "common_approaches": self._extract_common_approaches(successful)
        }
        
        return patterns
    
    def _extract_common_approaches(self, improvements: List[ImprovementResult]) -> List[str]:
        """Extract common approaches from successful improvements"""
        # Simplified - could use NLP to extract patterns
        return ["iterative_refinement", "modular_design", "thorough_testing"]
    
    async def _generate_new_strategies(self, patterns: Dict) -> List[Dict]:
        """Generate new learning strategies based on patterns"""
        
        # Use LLM to generate strategies
        prompt = f"""Based on these successful patterns:
{json.dumps(patterns, indent=2)}

Generate 3 new learning strategies that could improve the meta-learning process.
Each strategy should include:
- name
- description
- key_principles
- expected_benefit

Respond in JSON format.
"""
        
        response = await self._call_api_llm(prompt)
        
        try:
            strategies = json.loads(response)
            if isinstance(strategies, dict):
                strategies = [strategies]
        except:
            strategies = []
        
        return strategies
    
    async def _test_strategies(self, strategies: List[Dict]) -> Dict:
        """Test different strategies and return the best one"""
        
        # For now, return the first strategy (would implement actual testing)
        if strategies:
            return strategies[0]
        
        return {
            "name": "default",
            "description": "Default learning strategy",
            "key_principles": ["safety_first", "incremental_improvement"],
            "expected_benefit": 0.1
        }
    
    async def neural_architecture_search(
        self,
        task: str,
        constraints: Dict
    ) -> Dict:
        """
        Perform neural architecture search to design optimal architecture.
        
        Args:
            task: Task description (e.g., "image_classification")
            constraints: Constraints like max_params, latency, etc.
            
        Returns:
            Designed architecture specification
        """
        logger.info(f"🔍 Performing neural architecture search for: {task}")
        
        # Use LLM to design architecture
        prompt = f"""Design an optimal neural network architecture for: {task}

Constraints:
{json.dumps(constraints, indent=2)}

Provide architecture in JSON format with:
{{
    "name": "architecture_name",
    "layers": [
        {{"type": "layer_type", "params": {{}}}},
        ...
    ],
    "estimated_params": "number",
    "estimated_flops": "number",
    "expected_performance": "description"
}}
"""
        
        response = await self._call_api_llm(prompt)
        
        try:
            architecture = json.loads(response)
        except:
            architecture = self._extract_json_from_text(response)
        
        return architecture
    
    def record_success(self, plan: Dict, metrics: Dict, performance_gain: float):
        """Record a successful improvement"""
        
        result = ImprovementResult(
            success=True,
            performance_gain=performance_gain,
            metrics=metrics,
            details=plan['problem'],
            timestamp=datetime.now()
        )
        
        self.improvement_history.append(result)
        self.successful_strategies.append(plan)
        
        logger.info(f"✅ Recorded successful improvement: {performance_gain:.2%} gain")
    
    def get_statistics(self) -> Dict:
        """Get meta-learning statistics"""
        
        total = len(self.improvement_history)
        if total == 0:
            return {
                "total_attempts": 0,
                "success_rate": 0,
                "avg_performance_gain": 0
            }
        
        successful = [imp for imp in self.improvement_history if imp.success]
        
        return {
            "total_attempts": total,
            "successful_attempts": len(successful),
            "success_rate": len(successful) / total,
            "avg_performance_gain": np.mean([imp.performance_gain for imp in successful]) if successful else 0,
            "current_learning_rate": self.current_learning_rate,
            "best_improvement": max([imp.performance_gain for imp in successful]) if successful else 0
        }
