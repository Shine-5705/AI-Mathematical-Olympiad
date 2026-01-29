"""Verifiers for pruning invalid MCTS branches."""
import re
import subprocess
import sys
from typing import Tuple
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

    Handles:
    - $...$ LaTeX equations
    - \\boxed{} answers with contradictions
    - Inline equations like "x = 5" in reasoning
    """

    def __init__(self):
        import sympy
        self.sp = sympy

    def verify(self, text: str) -> Tuple[bool, str]:
        # Check $...$ equations
        equations = re.findall(r'\$(.*?)\$', text)
        for eq in self._check_equations(equations):
            return False, eq

        # Check \boxed{} for numeric contradictions
        boxed = re.findall(r'\\boxed\{(.+?)\}', text)
        for b in boxed:
            # If boxed contains an equation, verify it
            if "=" in b:
                for err in self._check_equations([b]):
                    return False, f"Boxed contradiction: {err}"

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


class CodeExecutionVerifier(BaseVerifier):
    """Executes Python code blocks and checks for errors.

    TIR solutions contain ```python ... ``` blocks.
    If execution fails, the branch is pruned.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def verify(self, text: str) -> Tuple[bool, str]:
        blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if not blocks:
            return True, "no_code"

        for i, code in enumerate(blocks):
            success, output = self._run(code)
            if not success:
                return False, f"Code block {i} failed: {output}"
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


def create_verifier(mode: str = "symbolic") -> BaseVerifier:
    modes = {
        "none": NoOpVerifier,
        "symbolic": SymbolicVerifier,
        "code": CodeExecutionVerifier,
        "both": lambda: CompositeVerifier([SymbolicVerifier(), CodeExecutionVerifier()]),
    }
    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(modes.keys())}")

    factory = modes[mode]
    return factory() if callable(factory) else factory
