import time
from dataclasses import dataclass
from typing import Optional
import re
from collections import Counter

from src.search.generator import BaseGenerator, create_generator


@dataclass
class BaselineMetrics:
    """Tracks metrics for baseline methods."""
    total_tokens: int = 0
    solve_time: float = 0.0
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "total_tokens": self.total_tokens,
            "solve_time": round(self.solve_time, 2),
        }


class DirectPrompting:
    """Baseline: Single-shot direct prompting without search."""

    def __init__(self, generator: BaseGenerator = None, backend: str = "auto", **kwargs):
        self.generator = generator or create_generator(backend, **kwargs)
        self.metrics = BaselineMetrics(method="direct")

    def solve(self, problem: str) -> str:
        """Generate a single solution without search."""
        self.metrics = BaselineMetrics(method="direct")
        start_time = time.time()

        prompt = f"""Solve this math problem. Give your final answer as: Final Answer: [answer]

Problem: {problem}

Solution:"""

        candidates = self.generator.get_candidates(prompt, n=1)
        solution = candidates[0] if candidates else ""

        self.metrics.total_tokens = len(solution.split())
        self.metrics.solve_time = time.time() - start_time

        return prompt + "\n" + solution

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()


class ChainOfThought:
    """Baseline: Chain-of-thought prompting (linear reasoning)."""

    def __init__(self, generator: BaseGenerator = None, backend: str = "auto", **kwargs):
        self.generator = generator or create_generator(backend, **kwargs)
        self.metrics = BaselineMetrics(method="cot")

    def solve(self, problem: str, max_steps: int = 5) -> str:
        """Generate solution using chain-of-thought prompting."""
        self.metrics = BaselineMetrics(method="cot")
        start_time = time.time()

        prompt = f"""Solve this math problem step by step. Think carefully about each step.
After solving, write your final answer as: Final Answer: [answer]

Problem: {problem}

Let me solve this step by step:"""

        solution = prompt
        for _ in range(max_steps):
            candidates = self.generator.get_candidates(solution, n=1)
            if not candidates:
                break

            continuation = candidates[0]
            solution += "\n" + continuation
            self.metrics.total_tokens += len(continuation.split())

            if "final answer:" in solution.lower():
                break

        self.metrics.solve_time = time.time() - start_time
        return solution

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()


class SelfConsistency:
    """Baseline: Self-consistency (majority voting over multiple samples)."""

    def __init__(self, generator: BaseGenerator = None, backend: str = "auto", **kwargs):
        self.generator = generator or create_generator(backend, **kwargs)
        self.metrics = BaselineMetrics(method="self_consistency")

    def solve(self, problem: str, n_samples: int = 5) -> str:
        """Generate multiple solutions and return majority answer."""
        self.metrics = BaselineMetrics(method="self_consistency")
        start_time = time.time()

        prompt = f"""Solve this math problem step by step.
Write your final answer as: Final Answer: [answer]

Problem: {problem}

Solution:"""

        candidates = self.generator.get_candidates(prompt, n=n_samples)

        answers = []
        for candidate in candidates:
            self.metrics.total_tokens += len(candidate.split())
            answer = self._extract_answer(candidate)
            if answer:
                answers.append(answer)

        self.metrics.solve_time = time.time() - start_time

        if not answers:
            return prompt + "\n" + (candidates[0] if candidates else "")

        most_common = Counter(answers).most_common(1)[0][0]

        for candidate in candidates:
            if most_common in candidate:
                return prompt + "\n" + candidate

        return prompt + "\n" + candidates[0]

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from solution text."""
        match = re.search(r'final answer[:\s]*([^\n]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()
