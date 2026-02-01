"""
Problem Categorization and Strategy Selection
Routes problems to specialized solvers based on mathematical domain
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class ProblemType(Enum):
    """Mathematical problem categories."""
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    LINEAR_ALGEBRA = "linear_algebra"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DifficultyLevel(Enum):
    """Problem difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    OLYMPIAD = "olympiad"


@dataclass
class ProblemProfile:
    """Complete problem analysis."""
    text: str
    problem_type: ProblemType
    difficulty: DifficultyLevel
    requires_symbolic: bool
    requires_code: bool
    requires_geometry_engine: bool
    estimated_steps: int
    keywords: List[str]
    confidence: float  # 0-1, how confident we are in classification


class ProblemClassifier:
    """
    Classifies mathematical problems to route them to appropriate solvers.
    """
    
    def __init__(self):
        # Keyword patterns for each problem type
        self.patterns = {
            ProblemType.ALGEBRA: [
                r'\bsolve\b.*\bequation',
                r'\bpolynomial\b',
                r'\bquadratic\b',
                r'\blinear\b.*\bequation',
                r'\bfactor\b',
                r'\bexpand\b',
                r'\bsimplify\b',
                r'\broots?\b',
                r'x\^2|x\^3|x\^n',
            ],
            ProblemType.GEOMETRY: [
                r'\btriangle\b',
                r'\bcircle\b',
                r'\bangle\b',
                r'\barea\b',
                r'\bperimeter\b',
                r'\bparallel\b',
                r'\bperpendicular\b',
                r'\bcongruent\b',
                r'\bsimilar\b',
                r'\btheorem\b.*\b(pythagoras|pythagorean)',
            ],
            ProblemType.NUMBER_THEORY: [
                r'\bprime\b',
                r'\bdivisible\b',
                r'\bdivisor\b',
                r'\bgcd\b',
                r'\blcm\b',
                r'\bmodulo\b',
                r'\bcongruence\b',
                r'\binteger\b',
                r'\bfactorial\b',
                r'\bdiophantine\b',
            ],
            ProblemType.COMBINATORICS: [
                r'\bcombination\b',
                r'\bpermutation\b',
                r'\bchoose\b',
                r'\bways to\b',
                r'\bcount\b.*\bnumber\b',
                r'\barrange\b',
                r'\bselect\b',
                r'n choose k|\bC\(',
                r'\bbinomial\b',
            ],
            ProblemType.CALCULUS: [
                r'\bderivative\b',
                r'\bintegral\b',
                r'\blimit\b',
                r'\bcontinuous\b',
                r'\bdifferentiable\b',
                r'\\frac\{d\}',
                r'\\int',
                r'\\lim',
                r'\bmaximum\b.*\bfunction\b',
                r'\bminimum\b.*\bfunction\b',
            ],
            ProblemType.PROBABILITY: [
                r'\bprobability\b',
                r'\brandom\b',
                r'\bexpected value\b',
                r'\bvariance\b',
                r'\bdistribution\b',
                r'\bindependent\b.*\bevents?\b',
                r'\bcoin\b.*\bflip',
                r'\bdice\b',
                r'\blikelihood\b',
            ],
            ProblemType.LINEAR_ALGEBRA: [
                r'\bmatrix\b',
                r'\bvector\b',
                r'\beigenvalue\b',
                r'\beigenvector\b',
                r'\bdeterminant\b',
                r'\binverse\b.*\bmatrix\b',
                r'\bdot product\b',
                r'\bcross product\b',
                r'\blinear transformation\b',
            ]
        }
        
        # Difficulty indicators
        self.difficulty_keywords = {
            DifficultyLevel.EASY: [
                r'\bsimple\b', r'\bbasic\b', r'\belementary\b',
                r'single digit', r'one step'
            ],
            DifficultyLevel.MEDIUM: [
                r'\bsolve\b', r'\bfind\b', r'\bcalculate\b',
                r'multiple steps', r'\bstandard\b'
            ],
            DifficultyLevel.HARD: [
                r'\bprove\b', r'\bshow that\b', r'\bdemonstrate\b',
                r'\boptimize\b', r'\bmaximize\b.*\bsubject to\b'
            ],
            DifficultyLevel.OLYMPIAD: [
                r'\bIMO\b', r'\bUSAMO\b', r'\bAIME\b', r'\bPutnam\b',
                r'\bolympiad\b', r'\binternational\b.*\bcompetition\b'
            ]
        }
    
    def classify(self, problem_text: str) -> ProblemProfile:
        """
        Classify a problem and create a profile.
        
        Args:
            problem_text: The problem statement
            
        Returns:
            ProblemProfile with classification and recommendations
        """
        text = problem_text.lower()
        
        # Detect problem type
        type_scores = {ptype: 0 for ptype in ProblemType}
        
        for ptype, patterns in self.patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                type_scores[ptype] += matches
        
        # Get best match
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        if best_score == 0:
            best_type = ProblemType.UNKNOWN
            confidence = 0.0
        else:
            total_score = sum(type_scores.values())
            confidence = best_score / total_score if total_score > 0 else 0.5
        
        # Detect difficulty
        difficulty = self._classify_difficulty(text)
        
        # Analyze requirements
        requires_symbolic = self._requires_symbolic(text, best_type)
        requires_code = self._requires_code(text, best_type)
        requires_geometry_engine = best_type == ProblemType.GEOMETRY
        
        # Estimate steps
        estimated_steps = self._estimate_steps(text, best_type, difficulty)
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        return ProblemProfile(
            text=problem_text,
            problem_type=best_type,
            difficulty=difficulty,
            requires_symbolic=requires_symbolic,
            requires_code=requires_code,
            requires_geometry_engine=requires_geometry_engine,
            estimated_steps=estimated_steps,
            keywords=keywords,
            confidence=confidence
        )
    
    def _classify_difficulty(self, text: str) -> DifficultyLevel:
        """Classify problem difficulty."""
        for level in [DifficultyLevel.OLYMPIAD, DifficultyLevel.HARD, 
                      DifficultyLevel.MEDIUM, DifficultyLevel.EASY]:
            for pattern in self.difficulty_keywords.get(level, []):
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        # Default heuristics
        if len(text) > 500 or text.count('$') > 10:
            return DifficultyLevel.HARD
        elif len(text) > 200:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def _requires_symbolic(self, text: str, ptype: ProblemType) -> bool:
        """Check if problem requires symbolic computation."""
        symbolic_types = [
            ProblemType.ALGEBRA,
            ProblemType.CALCULUS,
            ProblemType.NUMBER_THEORY
        ]
        
        if ptype in symbolic_types:
            return True
        
        # Check for equations
        has_equations = bool(re.search(r'[=<>]', text))
        has_variables = bool(re.search(r'\b[a-zA-Z]\s*[=+\-*/]', text))
        
        return has_equations and has_variables
    
    def _requires_code(self, text: str, ptype: ProblemType) -> bool:
        """Check if problem benefits from code execution."""
        code_types = [
            ProblemType.COMBINATORICS,
            ProblemType.NUMBER_THEORY,
            ProblemType.PROBABILITY
        ]
        
        if ptype in code_types:
            return True
        
        # Check for computational patterns
        computational_keywords = [
            'compute', 'calculate', 'enumerate', 'count',
            'find all', 'list all', 'brute force'
        ]
        
        return any(kw in text.lower() for kw in computational_keywords)
    
    def _estimate_steps(self, text: str, ptype: ProblemType, difficulty: DifficultyLevel) -> int:
        """Estimate number of reasoning steps needed."""
        base_steps = {
            DifficultyLevel.EASY: 2,
            DifficultyLevel.MEDIUM: 4,
            DifficultyLevel.HARD: 6,
            DifficultyLevel.OLYMPIAD: 10
        }
        
        steps = base_steps.get(difficulty, 4)
        
        # Adjust for problem type
        if ptype == ProblemType.GEOMETRY:
            steps += 2  # Geometry often needs more steps
        elif ptype == ProblemType.COMBINATORICS:
            steps += 1
        
        # Adjust for length
        if len(text) > 500:
            steps += 2
        
        return min(steps, 15)  # Cap at 15 steps
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract mathematical keywords from text."""
        # Common math terms
        math_terms = [
            'equation', 'solve', 'find', 'prove', 'show',
            'triangle', 'circle', 'angle', 'prime', 'integer',
            'polynomial', 'function', 'derivative', 'integral',
            'probability', 'permutation', 'combination'
        ]
        
        found = []
        text_lower = text.lower()
        
        for term in math_terms:
            if term in text_lower:
                found.append(term)
        
        return found[:10]  # Return top 10


class StrategySelector:
    """
    Selects the best solving strategy based on problem profile.
    """
    
    def __init__(self):
        self.classifier = ProblemClassifier()
    
    def select_strategy(self, problem: str) -> Dict:
        """
        Analyze problem and recommend strategy.
        
        Returns:
            Dictionary with strategy recommendations
        """
        profile = self.classifier.classify(problem)
        
        strategy = {
            'problem_type': profile.problem_type.value,
            'difficulty': profile.difficulty.value,
            'confidence': profile.confidence,
            
            # MCTS parameters
            'mcts_iterations': self._recommend_iterations(profile),
            'mcts_depth': profile.estimated_steps + 2,
            'candidates_per_node': self._recommend_candidates(profile),
            
            # Verifier selection
            'use_symbolic_verifier': profile.requires_symbolic,
            'use_code_verifier': profile.requires_code,
            'use_geometry_engine': profile.requires_geometry_engine,
            
            # Search strategy
            'search_strategy': self._recommend_search_strategy(profile),
            'timeout_seconds': self._recommend_timeout(profile),
            
            # Profile details
            'profile': {
                'estimated_steps': profile.estimated_steps,
                'keywords': profile.keywords,
                'requires_symbolic': profile.requires_symbolic,
                'requires_code': profile.requires_code,
            }
        }
        
        return strategy
    
    def _recommend_iterations(self, profile: ProblemProfile) -> int:
        """Recommend number of MCTS iterations."""
        base = {
            DifficultyLevel.EASY: 5,
            DifficultyLevel.MEDIUM: 10,
            DifficultyLevel.HARD: 15,
            DifficultyLevel.OLYMPIAD: 20
        }
        return base.get(profile.difficulty, 10)
    
    def _recommend_candidates(self, profile: ProblemProfile) -> int:
        """Recommend candidates per MCTS node."""
        if profile.difficulty in [DifficultyLevel.HARD, DifficultyLevel.OLYMPIAD]:
            return 4
        else:
            return 3
    
    def _recommend_search_strategy(self, profile: ProblemProfile) -> str:
        """Recommend search strategy."""
        if profile.difficulty == DifficultyLevel.OLYMPIAD:
            return "full_mcts"  # Full MCTS for hard problems
        elif profile.difficulty == DifficultyLevel.EASY:
            return "best_of_n"  # Fast sampling for easy problems
        else:
            return "light_mcts"  # Light MCTS for medium problems
    
    def _recommend_timeout(self, profile: ProblemProfile) -> int:
        """Recommend timeout in seconds."""
        base = {
            DifficultyLevel.EASY: 30,
            DifficultyLevel.MEDIUM: 60,
            DifficultyLevel.HARD: 120,
            DifficultyLevel.OLYMPIAD: 300
        }
        return base.get(profile.difficulty, 60)


# ===== Test =====
if __name__ == "__main__":
    selector = StrategySelector()
    
    test_problems = [
        "Solve for x: 2x + 3 = 11",
        "In triangle ABC, angle A is 60 degrees and angle B is 45 degrees. Find angle C.",
        "Prove that there are infinitely many prime numbers.",
        "How many ways can you arrange 5 books on a shelf?",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'='*60}")
        print(f"Problem {i}: {problem}")
        print('='*60)
        
        strategy = selector.select_strategy(problem)
        
        print(f"Type: {strategy['problem_type']} (confidence: {strategy['confidence']:.2f})")
        print(f"Difficulty: {strategy['difficulty']}")
        print(f"Strategy: {strategy['search_strategy']}")
        print(f"MCTS: {strategy['mcts_iterations']} iterations, depth {strategy['mcts_depth']}")
        print(f"Verifiers: symbolic={strategy['use_symbolic_verifier']}, "
              f"code={strategy['use_code_verifier']}")
        print(f"Timeout: {strategy['timeout_seconds']}s")
