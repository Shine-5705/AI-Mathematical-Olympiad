"""MCTS for TIR-format mathematical reasoning."""
import math
import re
import time
from typing import List, Optional, Dict
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
    answer_candidates: int = 0
    unique_answers: int = 0

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
            "answer_candidates": self.answer_candidates,
            "unique_answers": self.unique_answers,
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
    """MCTS with symbolic verification and weighted majority voting."""

    def __init__(
        self,
        generator: BaseGenerator = None,
        verifier: BaseVerifier = None,
        backend: str = "auto",
        verify_mode: str = "both",
        uct_constant: float = 1.41,
        **kwargs
    ):
        self.generator = generator or create_generator(backend, **kwargs)
        self.verifier = verifier or create_verifier(verify_mode)
        self.uct_constant = uct_constant
        self.metrics = MCTSMetrics()
        self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the generator's tokenizer if available."""
        if self._tokenizer is None:
            tok = getattr(self.generator, 'tokenizer', None)
            if tok and hasattr(tok, 'encode'):
                self._tokenizer = tok
            else:
                self._tokenizer = False

        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return len(text.split())

    def search(
        self,
        problem: str,
        iterations: int = 10,
        max_depth: int = 10,
        candidates_per_node: int = 3,
    ) -> str:
        """Run MCTS search with weighted majority voting."""
        self.metrics = MCTSMetrics()
        start = time.time()

        root = MCTSNode(state=problem)

        for i in range(iterations):
            # Select
            node = root
            while node.children:
                node = max(node.children, key=lambda c: c.uct(self.uct_constant))

            # Expand + Verify
            if node.depth < max_depth:
                candidates = self.generator.get_candidates(node.state, n=candidates_per_node)
                for candidate in candidates:
                    self.metrics.total_nodes += 1
                    self.metrics.total_tokens += self._count_tokens(candidate)

                    full_state = node.state + "\n" + candidate
                    is_valid, _ = self.verifier.verify(full_state)
                    if is_valid:
                        node.children.append(MCTSNode(state=full_state, parent=node))
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

        self.metrics.search_time = time.time() - start

        return self._select_answer(root)

    def _collect_answers(self, node: MCTSNode) -> List[Dict]:
        """Recursively collect all \\boxed{} answers from leaf nodes with visit weights."""
        results = []
        if not node.children:
            answer = self._extract_boxed(node.state)
            if answer:
                results.append({
                    "answer": answer,
                    "state": node.state,
                    "visits": node.visits,
                })
            return results

        for child in node.children:
            results.extend(self._collect_answers(child))
        return results

    def _select_answer(self, root: MCTSNode) -> str:
        """Select final answer via weighted majority voting over all leaf answers."""
        candidates = self._collect_answers(root)

        if not candidates:
            # Fallback: follow most-visited path
            node = root
            while node.children:
                node = max(node.children, key=lambda c: c.visits)
            return node.state

        self.metrics.answer_candidates = len(candidates)

        # Accumulate visit-weighted votes per normalized answer
        answer_weights: Dict[str, float] = {}
        best_state: Dict[str, tuple] = {}  # normalized -> (visits, state)

        for c in candidates:
            norm = self._normalize_answer(c["answer"])
            answer_weights[norm] = answer_weights.get(norm, 0.0) + c["visits"]
            prev = best_state.get(norm)
            if prev is None or c["visits"] > prev[0]:
                best_state[norm] = (c["visits"], c["state"])

        self.metrics.unique_answers = len(answer_weights)

        winner = max(answer_weights, key=answer_weights.get)
        return best_state[winner][1]

    def _extract_boxed(self, text: str) -> Optional[str]:
        match = re.search(r'\\boxed\{(.+?)\}', text)
        return match.group(1) if match else None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize for voting comparison."""
        answer = re.sub(r'\s+', '', answer.strip())
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return f"{num:.6f}".rstrip('0').rstrip('.')
        except ValueError:
            return answer.lower()

    def _score(self, state: str) -> float:
        """TIR-aware reward based on solution progress."""
        if self._has_answer(state):
            return 1.0
        if "```python" in state and "```output" in state:
            return 0.5
        if "```python" in state:
            return 0.3
        return 0.1

    def _has_answer(self, text: str) -> bool:
        return bool(re.search(r'\\boxed\{.+?\}', text)) or "final answer" in text.lower()

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()
