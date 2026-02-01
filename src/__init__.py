"""
Complete AIMO Architecture - All Components
This file imports and exposes all components for easy access
"""

# Core Components
from src.symbolic_bridge import (
    SymbolicBridge,
    StateEvaluator,
    MathState,
    StateStatus
)

from src.enhanced_mcts import (
    EnhancedAIMO_MCTS,
    FastSolver,
    EnhancedMCTSMetrics
)

from src.problem_classifier import (
    ProblemClassifier,
    StrategySelector,
    ProblemType,
    DifficultyLevel
)

from src.aimo_pipeline import (
    AIMOSolver,
    CompetitionSolver,
    SolutionResult
)

__version__ = "1.0.0"

__all__ = [
    # Symbolic Bridge
    'SymbolicBridge',
    'StateEvaluator', 
    'MathState',
    'StateStatus',
    
    # MCTS
    'EnhancedAIMO_MCTS',
    'FastSolver',
    'EnhancedMCTSMetrics',
    
    # Classification
    'ProblemClassifier',
    'StrategySelector',
    'ProblemType',
    'DifficultyLevel',
    
    # Pipeline
    'AIMOSolver',
    'CompetitionSolver',
    'SolutionResult',
]


def print_architecture():
    """Print architecture overview."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        AIMO Neuro-Symbolic Architecture v1.0.0                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Components Loaded:
    
    ✓ Symbolic Bridge
      • SymbolicBridge: Text → Formal State
      • StateEvaluator: State Quality Scoring
      • MathState: Formal State Representation
      
    ✓ Enhanced MCTS
      • EnhancedAIMO_MCTS: MCTS with States
      • FastSolver: Best-of-N Sampling
      • EnhancedMCTSMetrics: Comprehensive Metrics
      
    ✓ Problem Classification
      • ProblemClassifier: Type & Difficulty
      • StrategySelector: Adaptive Strategy
      
    ✓ Complete Pipeline
      • AIMOSolver: Full-Featured Solver
      • CompetitionSolver: Optimized for Speed
    
    Quick Start:
    
    >>> from aimo_architecture import AIMOSolver
    >>> solver = AIMOSolver(generator=your_model)
    >>> result = solver.solve("Your problem here")
    >>> print(f"Answer: {result.answer}")
    
    """)


if __name__ == "__main__":
    print_architecture()
