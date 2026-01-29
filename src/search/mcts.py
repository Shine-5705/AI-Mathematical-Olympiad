"""MCTS for TIR-format mathematical reasoning."""
import math
import re
import time
from typing import List, Optional
from dataclasses import dataclass

from src.search.generator import BaseGenerator, create_generator
from src.search.verifier import BaseVerifier, create_verifier


@dataclass
class MCTSMetrics:
    total_nodes: int = 0
    pruned_nodes: int = 0
    total_tokens: int = 0
    search_time: float = 0.0
    iterations_completed: int = 0

    @property
    def prune_rate(self) -> float:
        return self.pruned_nodes / self.total_nodes if self.total_nodes else 0.0

    def to_dict(self) -> dict:
        return {
            "total_nodes": self.total_nodes,
            "pruned_nodes": self.pruned_nodes,
            "prune_rate": round(self.prune_rate, 4),
            "total_tokens": self.total_tokens,
            "search_time": round(self.search_time, 2),
            "iterations": self.iterations_completed,
        }


class MCTSNode:
    def __init__(self, state: str, parent: Optional["MCTSNode"] = None):
        self.state = state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0

    def uct(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    @property
    def depth(self) -> int:
        d, node = 0, self
        while node.parent:
            d += 1
            node = node.parent
        return d


class AIMO_MCTS:
    """MCTS with symbolic verification for TIR math reasoning."""

    def __init__(
        self,
        generator: BaseGenerator = None,
        verifier: BaseVerifier = None,
        backend: str = "auto",
        verify_mode: str = "symbolic",
        **kwargs
    ):
        self.generator = generator or create_generator(backend, **kwargs)
        self.verifier = verifier or create_verifier(verify_mode)
        self.metrics = MCTSMetrics()

    def search(
        self,
        problem: str,
        iterations: int = 10,
        max_depth: int = 10,
        candidates_per_node: int = 3,
    ) -> str:
        """Run MCTS search for a verified solution."""
        self.metrics = MCTSMetrics()
        start = time.time()

        root = MCTSNode(state=problem)

        for i in range(iterations):
            # Select
            node = root
            while node.children:
                node = max(node.children, key=lambda c: c.uct())

            # Expand
            if node.depth < max_depth:
                candidates = self.generator.get_candidates(node.state, n=candidates_per_node)
                for candidate in candidates:
                    self.metrics.total_nodes += 1
                    self.metrics.total_tokens += len(candidate.split())

                    is_valid, _ = self.verifier.verify(candidate)
                    if is_valid:
                        node.children.append(
                            MCTSNode(state=node.state + "\n" + candidate, parent=node)
                        )
                    else:
                        self.metrics.pruned_nodes += 1

            # Simulate
            reward = self._score(node.state)

            # Backpropagate
            current = node
            while current:
                current.visits += 1
                current.value += reward
                current = current.parent

            self.metrics.iterations_completed = i + 1
            if self._has_answer(node.state):
                break

        self.metrics.search_time = time.time() - start

        # Best path = most visited
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node.state

    def _score(self, state: str) -> float:
        """Reward based on TIR solution progress."""
        if self._has_answer(state):
            return 1.0
        if "```python" in state and "```output" in state:
            return 0.5
        if "```python" in state:
            return 0.3
        return 0.1

    def _has_answer(self, text: str) -> bool:
        """Check for \\boxed{} or Final Answer."""
        return bool(re.search(r'\\boxed\{.+?\}', text)) or "final answer" in text.lower()

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()
