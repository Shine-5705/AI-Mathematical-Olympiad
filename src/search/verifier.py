"""Verifiers for pruning invalid MCTS branches."""
import re
import subprocess
import sys
from typing import Tuple, Dict, List, Optional
from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, text: str) -> Tuple[bool, str]:
        """Returns (is_valid, reason)."""
        pass


class NoOpVerifier(BaseVerifier):
    """Accepts everything — for ablation."""
    def verify(self, text: str) -> Tuple[bool, str]:
        return True, "no_verify"


class SymbolicVerifier(BaseVerifier):
    """Checks mathematical consistency using SymPy.

    Detects:
    - Single-equation contradictions (LHS ≠ RHS)
    - Cross-step variable contradictions (x=5 in step 1, x=3 in step 4)
    - Boxed answer contradictions
    """

    def __init__(self):
        import sympy
        self.sp = sympy

    def verify(self, text: str) -> Tuple[bool, str]:
        equations = re.findall(r'\$(.*?)\$', text)
        boxed = re.findall(r'\\boxed\{(.+?)\}', text)

        # Check individual equations
        for err in self._check_equations(equations):
            return False, err

        # Check boxed expressions
        for b in boxed:
            if "=" in b:
                for err in self._check_equations([b]):
                    return False, f"Boxed contradiction: {err}"

        # Check cross-equation variable consistency
        assignments = self._extract_assignments(equations + boxed)
        for err in self._check_variable_consistency(assignments):
            return False, err

        return True, "passed"

    def _check_equations(self, equations: list) -> list:
        errors = []
        for eq in equations:
            try:
                if "=" in eq and "==" not in eq and "\\neq" not in eq and "!=" not in eq:
                    parts = eq.split("=", 1)
                    if len(parts) != 2:
                        continue
                    lhs, rhs = parts[0].strip(), parts[1].strip()
                    if not lhs or not rhs:
                        continue
                    diff = self.sp.simplify(f"({lhs}) - ({rhs})")
                    if diff != 0 and diff.is_number:
                        errors.append(f"Contradiction: {lhs} ≠ {rhs} (diff={diff})")
            except Exception:
                continue
        return errors

    def _extract_assignments(self, equations: list) -> Dict[str, List]:
        """Extract variable assignments like 'x = 5' across all equations."""
        assignments: Dict[str, List] = {}
        var_pattern = re.compile(r'^([a-zA-Z_]\w*)\s*$')

        for eq in equations:
            try:
                if "=" not in eq or "==" in eq or "\\neq" in eq or "!=" in eq:
                    continue
                parts = eq.split("=", 1)
                if len(parts) != 2:
                    continue
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if not lhs or not rhs:
                    continue

                # Check if LHS is a simple variable name
                if var_pattern.match(lhs):
                    try:
                        value = self.sp.sympify(rhs)
                        if value.is_number:
                            assignments.setdefault(lhs, []).append(value)
                    except Exception:
                        continue

                # Also check reversed: "5 = x"
                if var_pattern.match(rhs):
                    try:
                        value = self.sp.sympify(lhs)
                        if value.is_number:
                            assignments.setdefault(rhs, []).append(value)
                    except Exception:
                        continue
            except Exception:
                continue
        return assignments

    def _check_variable_consistency(self, assignments: Dict[str, List]) -> list:
        """Flag variables assigned different numeric values."""
        errors = []
        for var, values in assignments.items():
            if len(values) < 2:
                continue
            first = values[0]
            for v in values[1:]:
                diff = self.sp.simplify(first - v)
                if diff != 0 and diff.is_number:
                    errors.append(
                        f"Variable '{var}' assigned conflicting values: {first} and {v}"
                    )
                    break
        return errors


class CodeExecutionVerifier(BaseVerifier):
    """Executes Python code blocks and checks for errors.

    Concatenates all code blocks into a single script so that
    variables from block 1 are available in block 2 (persistent state).
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def verify(self, text: str) -> Tuple[bool, str]:
        blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if not blocks:
            return True, "no_code"

        # Concatenate all blocks into one script (persistent state)
        combined = "\n".join(blocks)
        success, output = self._run(combined)
        if not success:
            return False, f"Code execution failed: {output}"
        return True, "code_passed"

    def _run(self, code: str) -> Tuple[bool, str]:
        """Execute code in subprocess with timeout."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                return False, result.stderr.strip().split('\n')[-1]
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)


class AnswerVerifier(BaseVerifier):
    """Substitutes the final answer back into the problem constraints.

    If the problem contains equations and the solution has \\boxed{answer},
    substitute the answer into the equations to verify correctness.
    """

    def __init__(self):
        import sympy
        self.sp = sympy

    def verify(self, text: str) -> Tuple[bool, str]:
        answer_match = re.search(r'\\boxed\{(.+?)\}', text)
        if not answer_match:
            return True, "no_answer_yet"

        answer_str = answer_match.group(1)
        try:
            answer_val = self.sp.sympify(answer_str)
        except Exception:
            return True, "answer_not_parseable"

        # Find equations in the problem (first line or first paragraph)
        lines = text.split("\n")
        problem_text = lines[0] if lines else text

        # Extract equations from the problem statement
        equations = re.findall(r'\$(.*?)\$', problem_text)
        if not equations:
            return True, "no_equations_in_problem"

        for eq in equations:
            try:
                if "=" not in eq or "==" in eq or "\\neq" in eq:
                    continue
                parts = eq.split("=", 1)
                if len(parts) != 2:
                    continue
                lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
                if not lhs_str or not rhs_str:
                    continue

                lhs = self.sp.sympify(lhs_str)
                rhs = self.sp.sympify(rhs_str)

                # Find free variables in the equation
                free_vars = lhs.free_symbols | rhs.free_symbols
                if len(free_vars) != 1:
                    continue

                var = free_vars.pop()
                substituted = self.sp.simplify(
                    (lhs - rhs).subs(var, answer_val)
                )
                if substituted != 0 and substituted.is_number:
                    return False, (
                        f"Back-substitution failed: {var}={answer_val} "
                        f"does not satisfy {lhs_str}={rhs_str} (residual={substituted})"
                    )
            except Exception:
                continue

        return True, "back_substitution_passed"


class CompositeVerifier(BaseVerifier):
    """Runs multiple verifiers — fails if any fails."""

    def __init__(self, verifiers: list[BaseVerifier]):
        self.verifiers = verifiers

    def verify(self, text: str) -> Tuple[bool, str]:
        for v in self.verifiers:
            ok, reason = v.verify(text)
            if not ok:
                return False, f"{v.__class__.__name__}: {reason}"
        return True, "all_passed"


def create_verifier(mode: str = "both") -> BaseVerifier:
    """Create verifier by mode.

    Modes:
        none     - accept everything (ablation baseline)
        symbolic - SymPy equation + variable tracking
        code     - code execution with persistent state
        both     - symbolic + code + answer back-substitution
    """
    if mode == "none":
        return NoOpVerifier()
    elif mode == "symbolic":
        return SymbolicVerifier()
    elif mode == "code":
        return CodeExecutionVerifier()
    elif mode == "both":
        return CompositeVerifier([
            SymbolicVerifier(),
            CodeExecutionVerifier(),
            AnswerVerifier(),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: none, symbolic, code, both")
