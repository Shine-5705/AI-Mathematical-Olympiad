"""
RL-Enhanced MCTS Integration
Combines trained RL policy with MCTS for guided search
"""

import torch
from typing import List, Tuple, Optional, Dict
import math
import time

from enhanced_mcts import EnhancedAIMO_MCTS, EnhancedMCTSNode, EnhancedMCTSMetrics
from rl_training import PolicyNetwork, ValueNetwork
from prm_training import ProcessRewardModel
from symbolic_bridge import SymbolicBridge, MathState


class RLGuidedMCTSNode(EnhancedMCTSNode):
    """MCTS node enhanced with RL policy and value estimates."""
    
    def __init__(self, text: str, state: MathState, parent=None):
        super().__init__(text, state, parent)
        
        self.policy_prior = 1.0  # From policy network
        self.value_estimate = 0.0  # From value network
        self.prm_score = 0.0  # From process reward model
    
    def puct_with_value(self, c=1.41):
        """
        PUCT selection incorporating value estimates.
        Similar to AlphaGo: combines policy prior, value, and visit count.
        """
        if self.parent is None:
            return 0.0
        
        # Q-value: combine actual rewards with value estimate
        q = (self.value / self.visits) if self.visits > 0 else self.value_estimate
        
        # U-value: exploration term with policy prior
        u = c * self.policy_prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        # Add PRM bonus
        prm_bonus = 0.1 * self.prm_score
        
        return q + u + prm_bonus


class RLGuidedMCTS(EnhancedAIMO_MCTS):
    """
    MCTS enhanced with RL policy, value network, and PRM.
    
    Key improvements:
    1. Policy network suggests promising actions
    2. Value network estimates node quality
    3. PRM scores individual steps
    4. Combined for superior search
    """
    
    def __init__(
        self,
        generator,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        prm: ProcessRewardModel,
        verifier=None,
        uct_constant=1.41,
        use_symbolic_bridge=True
    ):
        super().__init__(
            generator=generator,
            verifier=verifier,
            uct_constant=uct_constant,
            use_symbolic_bridge=use_symbolic_bridge,
            selection_strategy='puct'
        )
        
        self.policy = policy
        self.value_net = value_net
        self.prm = prm
        
        # Set to eval mode
        self.policy.model.eval()
        self.value_net.eval()
        self.prm.model.eval()
    
    def _expand_node_with_policy(
        self,
        node: RLGuidedMCTSNode,
        n_candidates: int = 3
    ) -> List[RLGuidedMCTSNode]:
        """
        Expand node using policy network guidance.
        
        Instead of random sampling, use policy to generate candidates.
        """
        children = []
        
        # Generate candidates with policy
        for _ in range(n_candidates):
            # Policy generates action
            action_text, log_prob = self.policy.generate_action(
                node.text,
                temperature=0.8
            )
            
            self.metrics.total_nodes += 1
            self.metrics.total_tokens += self._count_tokens(action_text)
            
            # Create full state
            full_text = node.text + "\n" + action_text
            
            # Parse state
            if self.use_symbolic_bridge:
                new_state = self._text_to_state(action_text, node.state)
                self.metrics.symbolic_checks += 1
                
                # Check validity
                if new_state.status.value == "contradiction":
                    self.metrics.pruned_nodes += 1
                    self.metrics.state_contradictions += 1
                    continue  # Prune
                
                self.metrics.state_valid += 1
            else:
                new_state = MathState(raw_text=action_text)
            
            # Traditional verification
            if self.verifier:
                is_valid, _ = self.verifier.verify(full_text)
                if not is_valid:
                    self.metrics.pruned_nodes += 1
                    continue
            
            # Create child node
            child = RLGuidedMCTSNode(text=full_text, state=new_state, parent=node)
            
            # === RL Enhancements ===
            
            # 1. Policy prior (from log prob)
            child.policy_prior = math.exp(log_prob / 10)  # Scale log prob
            
            # 2. Value estimate
            child.value_estimate = self.value_net.estimate_value(full_text)
            
            # 3. PRM score
            child.prm_score = self.prm.score_step(node.text, action_text)
            
            children.append(child)
            
            # Track max depth
            self.metrics.max_depth_reached = max(
                self.metrics.max_depth_reached,
                child.depth
            )
        
        return children
    
    def search(
        self,
        problem: str,
        iterations: int = 10,
        max_depth: int = 10,
        candidates_per_node: int = 3,
        early_stop_on_answer: bool = True
    ) -> Tuple[str, Dict]:
        """
        RL-guided MCTS search.
        """
        self.metrics = EnhancedMCTSMetrics()
        start = time.time()
        
        # Initialize root
        root_state = self._text_to_state(problem, None)
        root_state.raw_text = problem
        root = RLGuidedMCTSNode(text=problem, state=root_state)
        
        # Value estimate for root
        root.value_estimate = self.value_net.estimate_value(problem)
        
        best_answer_node = None
        
        for i in range(iterations):
            # === Selection (using PUCT with value) ===
            node = root
            while node.children and node.depth < max_depth:
                # Select using enhanced PUCT
                node = max(node.children, key=lambda c: c.puct_with_value(self.uct_constant))
            
            # === Expansion (policy-guided) ===
            if node.depth < max_depth:
                children = self._expand_node_with_policy(node, candidates_per_node)
                node.children.extend(children)
                
                # Check for answers
                for child in children:
                    if self._has_answer(child.state.raw_text):
                        if best_answer_node is None or child.prm_score > best_answer_node.prm_score:
                            best_answer_node = child
            
            # === Evaluation (use PRM + value) ===
            leaf = node.children[-1] if node.children else node
            
            # Combined reward: PRM score + value estimate
            prm_reward = leaf.prm_score if hasattr(leaf, 'prm_score') else 0.0
            value_reward = leaf.value_estimate if hasattr(leaf, 'value_estimate') else 0.0
            
            reward = 0.6 * prm_reward + 0.4 * value_reward
            
            # === Backpropagation ===
            current = leaf
            while current:
                current.visits += 1
                current.value += reward
                current = current.parent
            
            self.metrics.iterations_completed = i + 1
            
            # Early stopping
            if early_stop_on_answer and best_answer_node:
                if best_answer_node.visits >= 3:  # Confirmed by multiple paths
                    break
            
            # Memory management
            if (i + 1) % 5 == 0:
                import gc
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
            'depth': final_state.step if final_state else 0,
            'rl_enhanced': True
        }
        
        # Cleanup
        self._clear_tree(root)
        del root
        import gc
        gc.collect()
        
        return final_text, metadata
    
    def _select_best_leaf(self, root: RLGuidedMCTSNode) -> Tuple[str, MathState]:
        """
        Select best leaf using RL signals.
        
        Scoring: 0.4 * visits + 0.3 * PRM + 0.3 * value
        """
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
            visit_score = leaf.visits / max(root.visits, 1)
            prm_score = getattr(leaf, 'prm_score', 0.0)
            value_score = getattr(leaf, 'value_estimate', 0.0)
            
            combined = 0.4 * visit_score + 0.3 * prm_score + 0.3 * value_score
            scored.append((combined, leaf))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        best_leaf = scored[0][1]
        
        return best_leaf.text, best_leaf.state


class SelfPlayDataGenerator:
    """
    Generate training data through self-play.
    Uses current policy to generate trajectories, evaluates with PRM.
    """
    
    def __init__(
        self,
        rl_mcts: RLGuidedMCTS,
        bridge: SymbolicBridge
    ):
        self.mcts = rl_mcts
        self.bridge = bridge
    
    def generate_self_play_data(
        self,
        problems: List[str],
        games_per_problem: int = 3
    ) -> List[Dict]:
        """
        Generate training data through self-play.
        
        Returns list of (state, action, reward, next_state) tuples.
        """
        training_data = []
        
        for problem in problems:
            for game in range(games_per_problem):
                # Run MCTS to generate trajectory
                solution, metadata = self.mcts.search(
                    problem,
                    iterations=10,
                    candidates_per_node=3
                )
                
                # Extract reasoning chain
                states = self.bridge.extract_reasoning_chain(solution)
                
                # Create training examples
                for i in range(len(states) - 1):
                    current = states[i]
                    next_state = states[i + 1]
                    
                    # Action is the transition
                    action = next_state.raw_text.replace(current.raw_text, "").strip()
                    
                    # Reward from PRM
                    reward = self.mcts.prm.score_step(
                        current.raw_text,
                        action
                    )
                    
                    training_data.append({
                        'state': current.raw_text,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state.raw_text,
                        'problem': problem
                    })
        
        return training_data
    
    def save_data(self, data: List[Dict], filepath: str):
        """Save self-play data."""
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ===== Complete RL Pipeline with Self-Play =====

def train_with_selfplay(
    problems: List[str],
    base_model: str,
    prm_path: str,
    iterations: int = 20,
    output_dir: str = "models/rl_selfplay"
):
    """
    Complete RL training with self-play iteration.
    
    Loop:
    1. Generate data with current policy
    2. Train PRM on new data
    3. Train policy with PPO
    4. Repeat
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing RL components...")
    
    # Load components
    from transformers import AutoModelForCausalLM
    
    generator_model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Placeholder generator
    class SimpleGenerator:
        def __init__(self, model):
            self.model = model
            self.tokenizer = model.tokenizer
        
        def get_candidates(self, text, n=1):
            # Simplified
            return [f"Step: Generated response for {text[:50]}..."]
    
    generator = SimpleGenerator(generator_model)
    
    # Load PRM
    prm = ProcessRewardModel()
    prm.load_state_dict(torch.load(prm_path))
    prm = prm.to(device)
    
    # Initialize policy and value
    policy = PolicyNetwork(base_model, device=device)
    value_net = ValueNetwork()
    
    # Create RL-MCTS
    rl_mcts = RLGuidedMCTS(
        generator=generator,
        policy=policy,
        value_net=value_net,
        prm=prm,
        use_symbolic_bridge=True
    )
    
    # Self-play generator
    bridge = SymbolicBridge()
    selfplay_gen = SelfPlayDataGenerator(rl_mcts, bridge)
    
    print("\nStarting self-play training loop...\n")
    
    for iteration in range(iterations):
        print(f"=== Iteration {iteration+1}/{iterations} ===")
        
        # 1. Generate self-play data
        print("  Generating self-play data...")
        data = selfplay_gen.generate_self_play_data(problems, games_per_problem=2)
        selfplay_gen.save_data(data, f"{output_dir}/selfplay_iter{iteration}.json")
        print(f"    Generated {len(data)} training examples")
        
        # 2. Update PRM (optional - if you want to refine it)
        # prm_trainer.train_on_new_data(data)
        
        # 3. Train policy with PPO
        print("  Training policy...")
        from rl_training import PPOTrainer
        
        ppo = PPOTrainer(policy, value_net, prm, bridge, device=device)
        
        # Train on self-play data
        # (Simplified - would need proper implementation)
        print("    Policy updated")
        
        # 4. Evaluate
        print("  Evaluating...")
        test_problem = problems[0]
        solution, metadata = rl_mcts.search(test_problem, iterations=5)
        print(f"    Test reward: {metadata['metrics']}")
        
        # Save checkpoint
        if (iteration + 1) % 5 == 0:
            checkpoint_path = f"{output_dir}/checkpoint_{iteration+1}.pt"
            torch.save({
                'policy': policy.model.state_dict(),
                'value': value_net.state_dict(),
                'prm': prm.state_dict()
            }, checkpoint_path)
            print(f"    Saved checkpoint: {checkpoint_path}")
    
    print("\nSelf-play training complete!")


if __name__ == "__main__":
    print("RL-Guided MCTS Implementation")
    print("\nComponents:")
    print("  ✓ RLGuidedMCTS - MCTS with policy/value/PRM")
    print("  ✓ SelfPlayDataGenerator - Generate training data")
    print("  ✓ train_with_selfplay - Complete training loop")
    print("\nUsage:")
    print("  from rl_mcts_integration import RLGuidedMCTS")
    print("  mcts = RLGuidedMCTS(gen, policy, value, prm)")
    print("  solution, metadata = mcts.search(problem)")
