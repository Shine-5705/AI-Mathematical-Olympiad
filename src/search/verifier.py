import re
from typing import Tuple
from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """Abstract base for solution verifiers."""

    @abstractmethod
    def verify(self, text: str) -> Tuple[bool, str]:
        """Verify text. Returns (is_valid, reason)."""
        pass


class NoOpVerifier(BaseVerifier):
    """Verifier that accepts everything (for ablation studies)."""

    def verify(self, text: str) -> Tuple[bool, str]:
        return True, "no_verify"


class SymbolicVerifier(BaseVerifier):
    """Verifies mathematical consistency using SymPy."""

    def __init__(self):
        try:
            import sympy
            self.sp = sympy
        except ImportError:
            raise ImportError("sympy not installed. Run: pip install sympy")

    def verify(self, text: str) -> Tuple[bool, str]:
        """Check for mathematical contradictions in LaTeX equations."""
        equations = re.findall(r'\$(.*?)\$', text)
        for eq in equations:
            try:
                if "=" in eq and "==" not in eq:
                    parts = eq.split("=", 1)
                    if len(parts) != 2:
                        continue
                    lhs, rhs = parts
                    diff = self.sp.simplify(f"({lhs}) - ({rhs})")
                    if diff != 0 and diff.is_number:
                        return False, f"Contradiction: {lhs} != {rhs}"
            except Exception:
                continue
        return True, "passed"


class CodeExecutionVerifier(BaseVerifier):
    """Verifies by executing Python code blocks."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def verify(self, text: str) -> Tuple[bool, str]:
        """Execute Python code blocks and verify they run without error."""
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if not code_blocks:
            return True, "no_code"

        for i, code in enumerate(code_blocks):
            success, result = self._execute_code(code)
            if not success:
                return False, f"Code block {i} failed: {result}"
        return True, "code_passed"

    def _execute_code(self, code: str) -> Tuple[bool, str]:
        """Safely execute code with timeout."""
        import multiprocessing
        import io
        import sys

        def run_code(code: str, output_queue):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                exec_globals = {"__builtins__": __builtins__}
                exec(code, exec_globals)
                output = sys.stdout.getvalue()
                output_queue.put((True, output))
            except Exception as e:
                output_queue.put((False, str(e)))
            finally:
                sys.stdout = old_stdout

        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_code, args=(code, output_queue))
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return False, "timeout"

        if output_queue.empty():
            return False, "no_output"

        return output_queue.get()


class CompositeVerifier(BaseVerifier):
    """Combines multiple verifiers."""

    def __init__(self, verifiers: list[BaseVerifier]):
        self.verifiers = verifiers

    def verify(self, text: str) -> Tuple[bool, str]:
        """Run all verifiers. Fails if any verifier fails."""
        for verifier in self.verifiers:
            is_valid, reason = verifier.verify(text)
            if not is_valid:
                return False, f"{verifier.__class__.__name__}: {reason}"
        return True, "all_passed"


def create_verifier(mode: str = "symbolic") -> BaseVerifier:
    """Factory function to create verifier."""
    if mode == "none":
        return NoOpVerifier()
    elif mode == "symbolic":
        return SymbolicVerifier()
    elif mode == "code":
        return CodeExecutionVerifier()
    elif mode == "both":
        return CompositeVerifier([SymbolicVerifier(), CodeExecutionVerifier()])
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: none, symbolic, code, both")
