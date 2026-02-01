"""
Complete AIMO Competition Pipeline
Integrates all components: classification, symbolic bridge, enhanced MCTS
"""

import time
import gc
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json

from problem_classifier import StrategySelector, ProblemProfile
from symbolic_bridge import SymbolicBridge, StateEvaluator, MathState
from enhanced_mcts import EnhancedAIMO_MCTS, FastSolver


@dataclass
class SolutionResult:
    """Complete solution with metadata."""
    problem: str
    solution: str
    answer: Optional[str]
    confidence: float
    solve_time: float
    strategy_used: str
    metrics: Dict[str, Any]
    state: Optional[Dict] = None


class AIMOSolver:
    """
    Complete AIMO competition solver.
    
    Features:
    1. Problem classification and strategy selection
    2. Symbolic state tracking
    3. Adaptive MCTS or fast sampling based on problem type
    4. Multi-verifier pipeline
    """
    
    def __init__(
        self,
        generator,
        verifier=None,
        use_symbolic=True,
        use_adaptive_strategy=True,
        fast_mode=False
    ):
        """
        Args:
            generator: LLM generator (e.g., TransformersGenerator)
            verifier: Optional external verifier
            use_symbolic: Use symbolic bridge for state tracking
            use_adaptive_strategy: Adapt strategy based on problem type
            fast_mode: Use fast inference for competition speed
        """
        self.generator = generator
        self.verifier = verifier
        self.use_symbolic = use_symbolic
        self.use_adaptive_strategy = use_adaptive_strategy
        self.fast_mode = fast_mode
        
        # Initialize components
        self.strategy_selector = StrategySelector()
        self.bridge = SymbolicBridge() if use_symbolic else None
        self.evaluator = StateEvaluator() if use_symbolic else None
        
        # Statistics
        self.stats = {
            'problems_solved': 0,
            'problems_failed': 0,
            'total_time': 0.0,
            'by_type': {},
            'by_difficulty': {},
        }
    
    def solve(
        self,
        problem: str,
        max_time: Optional[float] = None,
        force_strategy: Optional[str] = None
    ) -> SolutionResult:
        """
        Solve a single problem.
        
        Args:
            problem: Problem text
            max_time: Optional timeout in seconds
            force_strategy: Force specific strategy ('fast', 'mcts', 'full_mcts')
            
        Returns:
            SolutionResult with solution and metadata
        """
        start_time = time.time()
        
        try:
            # === Phase 1: Problem Analysis ===
            if self.use_adaptive_strategy and not force_strategy:
                strategy_info = self.strategy_selector.select_strategy(problem)
                recommended_strategy = strategy_info['search_strategy']
            else:
                strategy_info = {'search_strategy': force_strategy or 'mcts'}
                recommended_strategy = strategy_info['search_strategy']
            
            # Apply timeout
            if max_time is None and 'timeout_seconds' in strategy_info:
                max_time = strategy_info['timeout_seconds']
            
            # === Phase 2: Solve ===
            if self.fast_mode or recommended_strategy == 'best_of_n':
                solution, metadata = self._solve_fast(problem, max_time)
                strategy_used = 'fast_sampling'
            elif recommended_strategy == 'full_mcts':
                solution, metadata = self._solve_full_mcts(problem, strategy_info, max_time)
                strategy_used = 'full_mcts'
            else:  # light_mcts
                solution, metadata = self._solve_light_mcts(problem, strategy_info, max_time)
                strategy_used = 'light_mcts'
            
            # === Phase 3: Extract Answer ===
            answer = self._extract_answer(solution)
            
            # === Phase 4: Confidence Estimation ===
            confidence = self._estimate_confidence(solution, metadata)
            
            solve_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(strategy_info, solve_time, success=True)
            
            result = SolutionResult(
                problem=problem,
                solution=solution,
                answer=answer,
                confidence=confidence,
                solve_time=solve_time,
                strategy_used=strategy_used,
                metrics=metadata.get('metrics', {}),
                state=metadata.get('state')
            )
            
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            self._update_stats({}, solve_time, success=False)
            
            return SolutionResult(
                problem=problem,
                solution=f"Error: {str(e)}",
                answer=None,
                confidence=0.0,
                solve_time=solve_time,
                strategy_used='error',
                metrics={'error': str(e)}
            )
    
    def _solve_fast(self, problem: str, timeout: Optional[float]) -> Tuple[str, Dict]:
        """Fast solve using best-of-N sampling."""
        n_samples = 10 if timeout and timeout < 60 else 20
        actual_timeout = timeout or 30
        
        fast_solver = FastSolver(
            self.generator,
            n_samples=n_samples,
            timeout=actual_timeout
        )
        
        return fast_solver.solve(problem)
    
    def _solve_light_mcts(
        self,
        problem: str,
        strategy_info: Dict,
        timeout: Optional[float]
    ) -> Tuple[str, Dict]:
        """Lightweight MCTS for medium problems."""
        mcts = EnhancedAIMO_MCTS(
            generator=self.generator,
            verifier=self.verifier,
            use_symbolic_bridge=self.use_symbolic,
            selection_strategy='uct'
        )
        
        iterations = min(strategy_info.get('mcts_iterations', 10), 10)
        max_depth = strategy_info.get('mcts_depth', 8)
        candidates = strategy_info.get('candidates_per_node', 2)
        
        return mcts.search(
            problem,
            iterations=iterations,
            max_depth=max_depth,
            candidates_per_node=candidates,
            early_stop_on_answer=True
        )
    
    def _solve_full_mcts(
        self,
        problem: str,
        strategy_info: Dict,
        timeout: Optional[float]
    ) -> Tuple[str, Dict]:
        """Full MCTS for hard problems."""
        mcts = EnhancedAIMO_MCTS(
            generator=self.generator,
            verifier=self.verifier,
            use_symbolic_bridge=self.use_symbolic,
            selection_strategy='puct'  # Use PUCT for better exploration
        )
        
        iterations = strategy_info.get('mcts_iterations', 20)
        max_depth = strategy_info.get('mcts_depth', 12)
        candidates = strategy_info.get('candidates_per_node', 4)
        
        return mcts.search(
            problem,
            iterations=iterations,
            max_depth=max_depth,
            candidates_per_node=candidates,
            early_stop_on_answer=False  # Don't early stop on hard problems
        )
    
    def _extract_answer(self, solution: str) -> Optional[str]:
        """Extract final answer from solution."""
        if self.bridge:
            states = self.bridge.extract_reasoning_chain(solution)
            answer = self.bridge.get_final_answer(states)
            if answer:
                return str(answer)
        
        # Fallback: regex extraction
        import re
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'final answer[:\s]*\$?([^\n\$]+)\$?',
            r'answer is[:\s]*\$?([^\n\$]+)\$?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _estimate_confidence(self, solution: str, metadata: Dict) -> float:
        """
        Estimate confidence in solution.
        
        Factors:
        - Has valid answer: +0.3
        - Symbolic verification passed: +0.3
        - Good state quality: +0.2
        - High visit count (for MCTS): +0.2
        """
        confidence = 0.0
        
        # Has answer
        if self._extract_answer(solution):
            confidence += 0.3
        
        # Check state validity
        if 'state' in metadata and metadata['state']:
            state_dict = metadata['state']
            if state_dict.get('status') == 'valid':
                confidence += 0.3
        
        # Check metrics
        metrics = metadata.get('metrics', {})
        
        # Low contradiction rate
        if metrics.get('contradiction_rate', 1.0) < 0.1:
            confidence += 0.2
        
        # Sufficient exploration
        if metrics.get('iterations', 0) >= 5:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _update_stats(self, strategy_info: Dict, solve_time: float, success: bool):
        """Update solving statistics."""
        if success:
            self.stats['problems_solved'] += 1
        else:
            self.stats['problems_failed'] += 1
        
        self.stats['total_time'] += solve_time
        
        # Track by type and difficulty
        ptype = strategy_info.get('problem_type', 'unknown')
        difficulty = strategy_info.get('difficulty', 'unknown')
        
        self.stats['by_type'][ptype] = self.stats['by_type'].get(ptype, 0) + 1
        self.stats['by_difficulty'][difficulty] = self.stats['by_difficulty'].get(difficulty, 0) + 1
    
    def get_stats(self) -> Dict:
        """Get solver statistics."""
        total = self.stats['problems_solved'] + self.stats['problems_failed']
        
        return {
            'total_problems': total,
            'solved': self.stats['problems_solved'],
            'failed': self.stats['problems_failed'],
            'accuracy': self.stats['problems_solved'] / total if total > 0 else 0.0,
            'total_time': round(self.stats['total_time'], 2),
            'avg_time': round(self.stats['total_time'] / total, 2) if total > 0 else 0.0,
            'by_type': self.stats['by_type'],
            'by_difficulty': self.stats['by_difficulty'],
        }
    
    def solve_batch(
        self,
        problems: List[str],
        max_time_per_problem: Optional[float] = None,
        save_results: Optional[str] = None
    ) -> List[SolutionResult]:
        """
        Solve multiple problems.
        
        Args:
            problems: List of problem texts
            max_time_per_problem: Timeout per problem
            save_results: Optional path to save results JSON
            
        Returns:
            List of SolutionResult objects
        """
        results = []
        
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Solving...")
            
            result = self.solve(problem, max_time=max_time_per_problem)
            results.append(result)
            
            print(f"  Answer: {result.answer}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Time: {result.solve_time:.2f}s")
            
            # Memory management
            if i % 5 == 0:
                gc.collect()
        
        # Save results
        if save_results:
            self._save_results(results, save_results)
        
        return results
    
    def _save_results(self, results: List[SolutionResult], filepath: str):
        """Save results to JSON file."""
        data = {
            'results': [
                {
                    'problem': r.problem[:200],  # Truncate for readability
                    'answer': r.answer,
                    'confidence': r.confidence,
                    'solve_time': r.solve_time,
                    'strategy': r.strategy_used,
                    'metrics': r.metrics,
                }
                for r in results
            ],
            'statistics': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


# ===== Competition Mode =====

class CompetitionSolver(AIMOSolver):
    """
    Optimized for AIMO competition constraints.
    - Fast inference
    - Aggressive timeouts
    - Result caching
    """
    
    def __init__(self, generator, verifier=None):
        super().__init__(
            generator=generator,
            verifier=verifier,
            use_symbolic=True,
            use_adaptive_strategy=True,
            fast_mode=True  # Always use fast mode in competition
        )
        
        self.cache = {}  # Cache results
    
    def solve(
        self,
        problem: str,
        max_time: Optional[float] = 30  # 30s default timeout
    ) -> SolutionResult:
        """Solve with aggressive timeout."""
        # Check cache
        cache_key = hash(problem)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Solve
        result = super().solve(problem, max_time=max_time, force_strategy='best_of_n')
        
        # Cache
        self.cache[cache_key] = result
        
        return result


if __name__ == "__main__":
    print("AIMO Competition Pipeline loaded.")
    print("\nComponents:")
    print("  ✓ Problem Classification")
    print("  ✓ Symbolic Bridge")
    print("  ✓ Enhanced MCTS")
    print("  ✓ Fast Solver")
    print("  ✓ Competition Mode")
    print("\nReady to solve mathematical olympiad problems!")
