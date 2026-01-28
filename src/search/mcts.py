import math
from typing import List, Optional

from src.search.generator import NuminaGenerator
from src.search.verifier import SymbolicVerifier


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


class AIMO_MCTS:
    """MCTS orchestrator combining neural generation with symbolic verification."""

    def __init__(self):
        self.generator = NuminaGenerator()
        self.verifier = SymbolicVerifier()

    def search(self, problem: str, iterations: int = 10) -> str:
        """Run MCTS search to find the best verified solution."""
        root = MCTSNode(state=problem)

        for _ in range(iterations):
            node = self._select(root)
            self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        return self._get_best_path(root)

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select the most promising node using UCT."""
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.uct())
        return node

    def _expand(self, node: MCTSNode) -> None:
        """Generate and verify candidate expansions."""
        candidates = self.generator.get_candidates(node.state)
        for candidate in candidates:
            if self.verifier.verify(candidate):
                child = MCTSNode(
                    state=node.state + "\n" + candidate,
                    parent=node
                )
                node.children.append(child)

    def _simulate(self, node: MCTSNode) -> float:
        """Calculate reward based on solution completeness."""
        return 0.5 if "Final Answer:" in node.state else 0.1

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward back up the tree."""
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _get_best_path(self, root: MCTSNode) -> str:
        """Return the state from the most visited leaf node."""
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node.state
