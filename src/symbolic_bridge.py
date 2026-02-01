"""
Symbolic Bridge - Core State Representation System
Transforms LLM textual reasoning into formal state objects with symbolic verification
"""

import re
import sympy as sp
from sympy import symbols, sympify, simplify, solve
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class StateStatus(Enum):
    VALID = "valid"
    CONTRADICTION = "contradiction"
    INCOMPLETE = "incomplete"
    INVALID_SYNTAX = "invalid_syntax"


@dataclass
class MathState:
    """Formal representation of mathematical problem state at a given step."""
    
    # Known variables and their values
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Active constraints/equations
    constraints: List[str] = field(default_factory=list)
    
    # Goal state (what we're solving for)
    goal: Optional[str] = None
    
    # Symbolic expressions
    expressions: Dict[str, sp.Expr] = field(default_factory=dict)
    
    # Dependencies between variables
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Step number in reasoning chain
    step: int = 0
    
    # Raw text of this step
    raw_text: str = ""
    
    # Status
    status: StateStatus = StateStatus.VALID
    status_message: str = ""
    
    def __post_init__(self):
        """Initialize symbolic environment."""
        self.sympy_namespace = {}
        
    def copy(self):
        """Create a deep copy of the state."""
        import copy
        return copy.deepcopy(self)
    
    def add_variable(self, name: str, value: Any = None, expr: Optional[sp.Expr] = None):
        """Add or update a variable in the state."""
        if value is not None:
            if name in self.variables and self.variables[name] != value:
                # Detect contradiction
                self.status = StateStatus.CONTRADICTION
                self.status_message = f"Variable '{name}' redefined: {self.variables[name]} → {value}"
                return False
            self.variables[name] = value
        
        if expr is not None:
            self.expressions[name] = expr
            # Track dependencies
            deps = set(str(s) for s in expr.free_symbols)
            self.dependencies[name] = deps
        
        return True
    
    def add_constraint(self, constraint: str):
        """Add a constraint/equation to the state."""
        self.constraints.append(constraint)
    
    def verify_consistency(self) -> Tuple[bool, str]:
        """Check if all constraints are mutually consistent."""
        try:
            # Check variable consistency
            for var, deps in self.dependencies.items():
                if var in self.variables:
                    # Evaluate expression with known values
                    expr = self.expressions.get(var)
                    if expr:
                        subs_dict = {sp.Symbol(k): v for k, v in self.variables.items() if k in deps}
                        if subs_dict:
                            result = expr.subs(subs_dict)
                            if result.is_number and result != self.variables[var]:
                                return False, f"Inconsistent value for {var}: computed={result}, stored={self.variables[var]}"
            
            # Check constraints
            for constraint in self.constraints:
                if '=' in constraint and '==' not in constraint:
                    try:
                        lhs, rhs = constraint.split('=', 1)
                        lhs_expr = sympify(lhs.strip())
                        rhs_expr = sympify(rhs.strip())
                        
                        # Substitute known variables
                        subs_dict = {sp.Symbol(k): v for k, v in self.variables.items()}
                        lhs_val = lhs_expr.subs(subs_dict)
                        rhs_val = rhs_expr.subs(subs_dict)
                        
                        diff = simplify(lhs_val - rhs_val)
                        if diff.is_number and abs(float(diff)) > 1e-10:
                            return False, f"Constraint violated: {constraint} (diff={diff})"
                    except Exception as e:
                        # Skip unparseable constraints
                        pass
            
            return True, "consistent"
            
        except Exception as e:
            return True, f"verification_error: {str(e)}"
    
    def solve_for(self, target: str) -> Optional[Any]:
        """Attempt to solve for a target variable using constraints."""
        try:
            # Collect all equations involving the target
            equations = []
            target_sym = sp.Symbol(target)
            
            for constraint in self.constraints:
                if '=' in constraint and target in constraint:
                    try:
                        lhs, rhs = constraint.split('=', 1)
                        equations.append(sympify(lhs.strip()) - sympify(rhs.strip()))
                    except:
                        pass
            
            if not equations:
                return None
            
            # Substitute known variables
            subs_dict = {sp.Symbol(k): v for k, v in self.variables.items() if k != target}
            substituted = [eq.subs(subs_dict) for eq in equations]
            
            # Solve
            solutions = solve(substituted, target_sym)
            if solutions:
                return solutions[0] if isinstance(solutions, list) else solutions
            
            return None
            
        except Exception:
            return None
    
    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            'step': self.step,
            'variables': {k: str(v) for k, v in self.variables.items()},
            'constraints': self.constraints,
            'goal': self.goal,
            'status': self.status.value,
            'status_message': self.status_message,
            'raw_text': self.raw_text[:200] if self.raw_text else ""
        }


class SymbolicBridge:
    """
    Transforms LLM text output into formal MathState objects.
    This is the core neuro-symbolic bridge.
    """
    
    def __init__(self):
        self.equation_pattern = re.compile(r'\$([^$]+)\$')
        self.assignment_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)')
        self.let_pattern = re.compile(r'[Ll]et\s+([a-zA-Z_]\w*)\s*=\s*([^,\n.]+)')
        self.given_pattern = re.compile(r'[Gg]iven\s+(.+?)(?:\.|,|$)')
        self.boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')
        
    def text_to_state(self, text: str, previous_state: Optional[MathState] = None) -> MathState:
        """
        Convert LLM text output to a formal MathState.
        
        Args:
            text: Raw LLM output text
            previous_state: Previous state to build upon
            
        Returns:
            MathState object representing the current reasoning step
        """
        state = previous_state.copy() if previous_state else MathState()
        state.raw_text = text
        state.step = previous_state.step + 1 if previous_state else 0
        
        # Extract equations
        equations = self.equation_pattern.findall(text)
        for eq in equations:
            if '=' in eq:
                state.add_constraint(eq.strip())
        
        # Extract variable assignments (both "let x = 5" and "x = 5")
        for pattern in [self.let_pattern, self.assignment_pattern]:
            for match in pattern.finditer(text):
                var_name = match.group(1).strip()
                value_str = match.group(2).strip()
                
                try:
                    # Try to parse as symbolic expression
                    expr = sympify(value_str)
                    state.add_variable(var_name, expr=expr)
                    
                    # If it's a number, store the value too
                    if expr.is_number:
                        state.add_variable(var_name, value=float(expr))
                except Exception:
                    # Store as string if not parseable
                    state.add_variable(var_name, value=value_str)
        
        # Extract goal from common patterns
        goal_patterns = [
            r'[Ff]ind\s+([a-zA-Z_]\w*)',
            r'[Ss]olve for\s+([a-zA-Z_]\w*)',
            r'[Ww]hat is\s+([a-zA-Z_]\w*)',
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, text)
            if match and not state.goal:
                state.goal = match.group(1)
        
        # Verify consistency
        is_consistent, message = state.verify_consistency()
        if not is_consistent:
            state.status = StateStatus.CONTRADICTION
            state.status_message = message
        
        return state
    
    def extract_reasoning_chain(self, full_text: str) -> List[MathState]:
        """
        Extract a sequence of states from a full solution.
        
        Returns:
            List of MathState objects, one per reasoning step
        """
        # Split by common step indicators
        step_patterns = [
            r'Step \d+:',
            r'\n\d+\.',
            r'\n\n',
        ]
        
        # Combine patterns
        split_pattern = '|'.join(f'(?={p})' for p in step_patterns)
        steps = re.split(split_pattern, full_text)
        
        states = []
        previous_state = None
        
        for i, step_text in enumerate(steps):
            if step_text.strip():
                state = self.text_to_state(step_text, previous_state)
                states.append(state)
                previous_state = state
        
        return states
    
    def get_final_answer(self, states: List[MathState]) -> Optional[Any]:
        """Extract the final answer from a reasoning chain."""
        # Check last state for boxed answer
        if states:
            last_text = states[-1].raw_text
            match = self.boxed_pattern.search(last_text)
            if match:
                try:
                    return sympify(match.group(1))
                except:
                    return match.group(1)
        
        # Check if goal variable was solved
        for state in reversed(states):
            if state.goal and state.goal in state.variables:
                return state.variables[state.goal]
        
        return None
    
    def verify_reasoning_chain(self, states: List[MathState]) -> Tuple[bool, str]:
        """
        Verify the entire reasoning chain for consistency.
        
        Returns:
            (is_valid, reason)
        """
        for i, state in enumerate(states):
            if state.status == StateStatus.CONTRADICTION:
                return False, f"Step {i+1}: {state.status_message}"
            
            is_consistent, message = state.verify_consistency()
            if not is_consistent:
                return False, f"Step {i+1}: {message}"
        
        return True, "valid"


class StateEvaluator:
    """
    Evaluates the quality of a MathState using symbolic verification.
    Used by MCTS to score nodes.
    """
    
    def __init__(self):
        self.bridge = SymbolicBridge()
    
    def score_state(self, state: MathState) -> float:
        """
        Score a state from 0.0 (worst) to 1.0 (best).
        
        Scoring factors:
        - Has answer: +0.5
        - Consistent: +0.3
        - Progress toward goal: +0.2
        """
        score = 0.0
        
        # Check for answer
        if self.bridge.boxed_pattern.search(state.raw_text):
            score += 0.5
        
        # Check consistency
        if state.status == StateStatus.VALID:
            is_consistent, _ = state.verify_consistency()
            if is_consistent:
                score += 0.3
        
        # Check progress
        if state.goal and state.goal in state.variables:
            score += 0.2
        elif state.variables:  # Has some variables defined
            score += 0.1
        
        return score
    
    def compare_states(self, state1: MathState, state2: MathState) -> int:
        """
        Compare two states.
        Returns: -1 if state1 < state2, 0 if equal, 1 if state1 > state2
        """
        score1 = self.score_state(state1)
        score2 = self.score_state(state2)
        
        if score1 < score2:
            return -1
        elif score1 > score2:
            return 1
        else:
            return 0


# ===== Quick Test =====
if __name__ == "__main__":
    bridge = SymbolicBridge()
    evaluator = StateEvaluator()
    
    # Test text
    test_text = """
    Problem: Find x if $2x + 3 = 11$
    
    Step 1: Let's subtract 3 from both sides: $2x = 8$
    Step 2: Divide by 2: $x = 4$
    
    Therefore, $\\boxed{4}$
    """
    
    states = bridge.extract_reasoning_chain(test_text)
    
    print(f"Extracted {len(states)} states")
    for i, state in enumerate(states):
        score = evaluator.score_state(state)
        print(f"\nState {i+1} (score={score:.2f}):")
        print(f"  Variables: {state.variables}")
        print(f"  Constraints: {state.constraints}")
        print(f"  Status: {state.status.value}")
    
    is_valid, reason = bridge.verify_reasoning_chain(states)
    print(f"\nChain valid: {is_valid} ({reason})")
    
    answer = bridge.get_final_answer(states)
    print(f"Final answer: {answer}")
