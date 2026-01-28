import math
import time
from typing import List, Optional
from dataclasses import dataclass, field

from src.search.generator import BaseGenerator, create_generator
from src.search.verifier import BaseVerifier, create_verifier


@dataclass
class MCTSMetrics:
    """Tracks MCTS search metrics for experiments."""
    total_nodes: int = 0
    pruned_nodes: int = 0
    total_tokens: int = 0
    search_time: float = 0.0
    iterations_completed: int = 0

    @property
    def prune_rate(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return self.pruned_nodes / self.total_nodes

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
    """Node in the MCTS tree representing a solution state."""

    def __init__(self, state: str, parent: Optional["MCTSNode"] = None):
        self.state = state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0

    def uct(self, c: float = 1.41) -> float:
        """Calculate Upper Confidence Bound for Trees score."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    @property
    def depth(self) -> int:
        """Return depth of this node in the tree."""
        d = 0
        node = self
        while node.parent:
            d += 1
            node = node.parent
        return d


class AIMO_MCTS:
    """MCTS orchestrator combining neural generation with symbolic verification."""

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
        """Run MCTS search to find the best verified solution."""
        self.metrics = MCTSMetrics()
        start_time = time.time()

        root = MCTSNode(state=self._format_prompt(problem))

        for i in range(iterations):
            node = self._select(root)

            if node.depth < max_depth:
                self._expand(node, candidates_per_node)

            reward = self._simulate(node)
            self._backpropagate(node, reward)
            self.metrics.iterations_completed = i + 1

            if self._is_solution_complete(node):
                break

        self.metrics.search_time = time.time() - start_time
        return self._get_best_path(root)

    def _format_prompt(self, problem: str) -> str:
        """Format problem with system prompt for math solving."""
        return f"""Solve this math problem step by step. Show your reasoning clearly.
After solving, write your final answer as: Final Answer: [answer]

Problem: {problem}

Solution:"""

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select the most promising node using UCT."""
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.uct())
        return node

    def _expand(self, node: MCTSNode, n_candidates: int) -> None:
        """Generate and verify candidate expansions."""
        candidates = self.generator.get_candidates(node.state, n=n_candidates)

        for candidate in candidates:
            self.metrics.total_nodes += 1
            self.metrics.total_tokens += len(candidate.split())

            is_valid, reason = self.verifier.verify(candidate)
            if is_valid:
                child = MCTSNode(
                    state=node.state + "\n" + candidate,
                    parent=node
                )
                node.children.append(child)
            else:
                self.metrics.pruned_nodes += 1

    def _simulate(self, node: MCTSNode) -> float:
        """Calculate reward based on solution quality."""
        state = node.state.lower()

        if "final answer:" in state:
            return 1.0

        if any(kw in state for kw in ["therefore", "thus", "hence", "so we have"]):
            return 0.3

        return 0.1

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward back up the tree."""
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _is_solution_complete(self, node: MCTSNode) -> bool:
        """Check if current node contains a complete solution."""
        return "final answer:" in node.state.lower()

    def _get_best_path(self, root: MCTSNode) -> str:
        """Return the state from the most visited leaf node."""
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node.state

    def get_metrics(self) -> dict:
        """Return search metrics for analysis."""
        return self.metrics.to_dict()
