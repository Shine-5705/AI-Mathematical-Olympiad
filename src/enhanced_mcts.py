"""
Enhanced MCTS with State Tracking and Symbolic Verification
Integrates symbolic bridge for formal state representation
"""

import math
import time
import gc
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from symbolic_bridge import MathState, SymbolicBridge, StateEvaluator, StateStatus


@dataclass
class EnhancedMCTSMetrics:
    """Extended metrics with state tracking."""
    total_nodes: int = 0
    pruned_nodes: int = 0
    state_contradictions: int = 0
    state_valid: int = 0
    total_tokens: int = 0
    search_time: float = 0.0
    iterations_completed: int = 0
    answer_candidates: int = 0
    unique_answers: int = 0
    max_depth_reached: int = 0
    symbolic_checks: int = 0
    
    @property
    def prune_rate(self):
        return self.pruned_nodes / self.total_nodes if self.total_nodes else 0.0
    
    @property
    def contradiction_rate(self):
        return self.state_contradictions / self.symbolic_checks if self.symbolic_checks else 0.0
    
    def to_dict(self):
        return {
            "total_nodes": self.total_nodes,
            "pruned_nodes": self.pruned_nodes,
            "state_contradictions": self.state_contradictions,
            "state_valid": self.state_valid,
            "prune_rate": round(self.prune_rate, 4),
            "contradiction_rate": round(self.contradiction_rate, 4),
            "total_tokens": self.total_tokens,
            "search_time": round(self.search_time, 2),
            "iterations": self.iterations_completed,
            "answer_candidates": self.answer_candidates,
            "unique_answers": self.unique_answers,
            "max_depth": self.max_depth_reached,
            "symbolic_checks": self.symbolic_checks,
        }


class EnhancedMCTSNode:
    """MCTS node with formal state representation."""
    
    def __init__(self, text: str, state: MathState, parent=None):
        self.text = text  # Raw text
        self.state = state  # Formal symbolic state
        self.parent = parent
        self.children: List['EnhancedMCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0  # Can be set by policy network
        
    def uct(self, c=1.41, use_prior=False):
        """Upper Confidence Bound with optional prior."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        if use_prior:
            exploration *= self.prior
        
        return exploitation + exploration
    
    def puct(self, c=1.41):
        """PUCT formula used in AlphaGo/AlphaZero."""
        if self.parent is None:
            return 0.0
        
        u = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        q = self.value / self.visits if self.visits > 0 else 0
        return q + u
    
    @property
    def depth(self):
        d, node = 0, self
        while node.parent:
            d += 1
            node = node.parent
        return d
    
    def get_path_states(self) -> List[MathState]:
        """Get all states from root to this node."""
        states = []
        node = self
        while node:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))


class EnhancedAIMO_MCTS:
    """
    Enhanced MCTS with symbolic state tracking.
    
    Key improvements over basic MCTS:
    1. Formal state representation at each node
    2. Symbolic verification catches contradictions early
    3. State-aware scoring
    4. Better pruning decisions
    """
    
    def __init__(
        self,
        generator,
        verifier=None,
        uct_constant=1.41,
        use_symbolic_bridge=True,
        selection_strategy="uct"  # "uct" or "puct"
    ):
        self.generator = generator
        self.verifier = verifier
        self.uct_constant = uct_constant
        self.use_symbolic_bridge = use_symbolic_bridge
        self.selection_strategy = selection_strategy
        
        if use_symbolic_bridge:
            self.bridge = SymbolicBridge()
            self.evaluator = StateEvaluator()
        else:
            self.bridge = None
            self.evaluator = None
        
        self.metrics = EnhancedMCTSMetrics()
        self._tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            tok = getattr(self.generator, 'tokenizer', None)
            self._tokenizer = tok if tok and hasattr(tok, 'encode') else False
        
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except:
                pass
        return len(text.split())
    
    def _text_to_state(self, text: str, previous_state: Optional[MathState] = None) -> MathState:
        """Convert text to state using symbolic bridge."""
        if self.bridge:
            return self.bridge.text_to_state(text, previous_state)
        else:
            # Fallback: create minimal state
            state = MathState(raw_text=text)
            if previous_state:
                state.step = previous_state.step + 1
            return state
    
    def _score_state(self, state: MathState) -> float:
        """Score a state for MCTS value."""
        if self.evaluator:
            return self.evaluator.score_state(state)
        else:
            # Fallback: simple text-based scoring
            text = state.raw_text.lower()
            score = 0.0
            if "boxed" in text or "final answer" in text:
                score += 0.5
            if "therefore" in text or "thus" in text:
                score += 0.2
            if any(op in text for op in ["=", "+", "-", "*", "/"]):
                score += 0.1
            return min(score, 1.0)
    
    def search(
        self,
        problem: str,
        iterations: int = 10,
        max_depth: int = 10,
        candidates_per_node: int = 3,
        early_stop_on_answer: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run MCTS search with symbolic verification.
        
        Returns:
            (best_solution_text, metadata)
        """
        self.metrics = EnhancedMCTSMetrics()
        start = time.time()
        
        # Initialize root with problem as initial state
        root_state = self._text_to_state(problem, None)
        root_state.raw_text = problem
        root = EnhancedMCTSNode(text=problem, state=root_state)
        
        best_answer_node = None
        
        for i in range(iterations):
            # === Selection ===
            node = root
            while node.children and node.depth < max_depth:
                if self.selection_strategy == "puct":
                    node = max(node.children, key=lambda c: c.puct(self.uct_constant))
                else:
                    node = max(node.children, key=lambda c: c.uct(self.uct_constant))
            
            # === Expansion ===
            if node.depth < max_depth:
                candidates = self.generator.get_candidates(node.text, n=candidates_per_node)
                
                for candidate_text in candidates:
                    self.metrics.total_nodes += 1
                    self.metrics.total_tokens += self._count_tokens(candidate_text)
                    
                    # Create full text (problem + previous steps + new step)
                    full_text = node.text + "\n" + candidate_text
                    
                    # === Symbolic Bridge ===
                    if self.use_symbolic_bridge:
                        new_state = self._text_to_state(candidate_text, node.state)
                        self.metrics.symbolic_checks += 1
                        
                        # Check state validity
                        if new_state.status == StateStatus.CONTRADICTION:
                            self.metrics.pruned_nodes += 1
                            self.metrics.state_contradictions += 1
                            continue  # Prune this branch
                        
                        self.metrics.state_valid += 1
                    else:
                        new_state = MathState(raw_text=candidate_text)
                    
                    # === Traditional Verifier ===
                    if self.verifier:
                        is_valid, _ = self.verifier.verify(full_text)
                        if not is_valid:
                            self.metrics.pruned_nodes += 1
                            continue
                    
                    # Create new node
                    child = EnhancedMCTSNode(text=full_text, state=new_state, parent=node)
                    node.children.append(child)
                    
                    # Track max depth
                    self.metrics.max_depth_reached = max(self.metrics.max_depth_reached, child.depth)
                    
                    # Check for answer
                    if self._has_answer(candidate_text):
                        if best_answer_node is None or child.depth < best_answer_node.depth:
                            best_answer_node = child
                
                del candidates
            
            # === Simulation/Evaluation ===
            # Use the leaf node for evaluation
            leaf = node.children[-1] if node.children else node
            reward = self._score_state(leaf.state)
            
            # === Backpropagation ===
            current = leaf
            while current:
                current.visits += 1
                current.value += reward
                current = current.parent
            
            self.metrics.iterations_completed = i + 1
            
            # Early stopping
            if early_stop_on_answer and best_answer_node and best_answer_node.visits >= 3:
                break
            
            # Memory management
            if (i + 1) % 5 == 0:
                gc.collect()
        
        self.metrics.search_time = time.time() - start
        
        # === Answer Selection ===
        if best_answer_node:
            final_text = best_answer_node.text
            final_state = best_answer_node.state
        else:
            final_text, final_state = self._select_best_leaf(root)
        
        metadata = {
            'metrics': self.metrics.to_dict(),
            'state': final_state.to_dict() if final_state else None,
            'depth': final_state.step if final_state else 0
        }
        
        # Cleanup
        self._clear_tree(root)
        del root
        gc.collect()
        
        return final_text, metadata
    
    def _select_best_leaf(self, root: EnhancedMCTSNode) -> Tuple[str, MathState]:
        """Select best leaf node using visit count and state quality."""
        def get_leaves(node):
            if not node.children:
                return [node]
            leaves = []
            for child in node.children:
                leaves.extend(get_leaves(child))
            return leaves
        
        leaves = get_leaves(root)
        if not leaves:
            return root.text, root.state
        
        # Score leaves
        scored = []
        for leaf in leaves:
            state_score = self._score_state(leaf.state)
            visit_score = leaf.visits / max(root.visits, 1)
            combined_score = 0.6 * state_score + 0.4 * visit_score
            scored.append((combined_score, leaf))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        best_leaf = scored[0][1]
        
        return best_leaf.text, best_leaf.state
    
    def _clear_tree(self, node: EnhancedMCTSNode):
        """Recursively clear tree."""
        for child in node.children:
            self._clear_tree(child)
        node.children.clear()
        node.text = None
        node.state = None
        node.parent = None
    
    def _has_answer(self, text: str) -> bool:
        """Check if text contains an answer."""
        return bool(re.search(r'\\boxed\{.+?\}', text)) or "final answer" in text.lower()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get search metrics."""
        return self.metrics.to_dict()


# ===== Fast Inference Mode =====

class FastSolver:
    """
    Optimized for competition speed.
    Uses best-of-N sampling with symbolic verification instead of full MCTS.
    """
    
    def __init__(self, generator, n_samples=10, timeout=30):
        self.generator = generator
        self.n_samples = n_samples
        self.timeout = timeout
        self.bridge = SymbolicBridge()
        self.evaluator = StateEvaluator()
    
    def solve(self, problem: str) -> Tuple[str, Dict]:
        """
        Fast solve using best-of-N sampling.
        
        Returns:
            (solution_text, metadata)
        """
        start = time.time()
        candidates = []
        
        # Generate N candidates
        for i in range(self.n_samples):
            if time.time() - start > self.timeout:
                break
            
            try:
                candidate = self.generator.get_candidates(problem, n=1)[0]
                
                # Verify with symbolic bridge
                states = self.bridge.extract_reasoning_chain(candidate)
                is_valid, reason = self.bridge.verify_reasoning_chain(states)
                
                if is_valid and states:
                    final_state = states[-1]
                    score = self.evaluator.score_state(final_state)
                    candidates.append((score, candidate, final_state))
                
                del candidate, states
                gc.collect()
                
            except Exception as e:
                continue
        
        solve_time = time.time() - start
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_solution, best_state = candidates[0]
            
            metadata = {
                'solve_time': solve_time,
                'candidates_generated': len(candidates),
                'best_score': best_score,
                'state': best_state.to_dict()
            }
            
            return best_solution, metadata
        else:
            # Fallback: return raw generation
            fallback = self.generator.get_candidates(problem, n=1)[0]
            return fallback, {'solve_time': solve_time, 'fallback': True}


if __name__ == "__main__":
    print("Enhanced MCTS with Symbolic Bridge loaded.")
    print("\nFeatures:")
    print("  ✓ Formal state representation at each node")
    print("  ✓ Symbolic contradiction detection")
    print("  ✓ State-aware scoring")
    print("  ✓ Fast inference mode for competition")
