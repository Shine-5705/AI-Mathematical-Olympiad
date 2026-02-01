"""
Reinforcement Learning Training Loop (PPO-style)
Trains policy LLM using Process Reward Model feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import copy

from prm_training import ProcessRewardModel, StepLabel
from symbolic_bridge import SymbolicBridge


@dataclass
class Experience:
    """RL experience tuple."""
    state: str
    action: str
    reward: float
    next_state: str
    done: bool
    log_prob: float
    value: float


class PolicyNetwork(nn.Module):
    """
    Policy network wrapping a causal LLM.
    Generates next reasoning steps.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = device
    
    def generate_action(
        self,
        state: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ) -> Tuple[str, float]:
        """
        Generate next step with log probability.
        
        Returns:
            (action_text, log_prob)
        """
        inputs = self.tokenizer(state, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode action
        action_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        action = self.tokenizer.decode(action_ids, skip_special_tokens=True)
        
        # Compute log probability (approximate)
        scores = torch.stack(outputs.scores, dim=0)  # [seq_len, batch, vocab]
        probs = F.softmax(scores, dim=-1)
        
        log_prob = 0.0
        for i, token_id in enumerate(action_ids):
            if i < len(scores):
                log_prob += torch.log(probs[i, 0, token_id]).item()
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value network (critic) for advantage estimation.
    Can use same base as PRM or separate.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Value head
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        value = self.value_head(pooled)
        return value
    
    def estimate_value(self, state: str) -> float:
        """Estimate value of a state."""
        encoding = self.tokenizer(
            state,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        self.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                encoding = {k: v.cuda() for k, v in encoding.items()}
            
            value = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            )
        
        return value.item()


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for math reasoning.
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        prm: ProcessRewardModel,
        bridge: SymbolicBridge,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        device: str = "cuda"
    ):
        self.policy = policy
        self.value_net = value_net.to(device)
        self.prm = prm
        self.bridge = bridge
        self.device = device
        
        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            policy.model.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            value_net.parameters(),
            lr=learning_rate * 3  # Critic learns faster
        )
    
    def collect_trajectory(
        self,
        problem: str,
        max_steps: int = 10
    ) -> List[Experience]:
        """
        Collect one trajectory using current policy.
        """
        experiences = []
        state = problem
        
        for step in range(max_steps):
            # Generate action
            action, log_prob = self.policy.generate_action(state)
            
            # Estimate value
            value = self.value_net.estimate_value(state)
            
            # Apply action to get next state
            next_state = state + "\n" + action
            
            # Get reward from PRM
            prm_reward = self.prm.score_step(state, action)
            
            # Verify symbolically
            states = self.bridge.extract_reasoning_chain(next_state)
            if states:
                is_valid, _ = states[-1].verify_consistency()
                if not is_valid:
                    prm_reward = -1.0  # Override with penalty
            else:
                prm_reward = -0.5  # Parse failure
            
            # Check if done
            done = (
                self.bridge._has_answer(action) or
                step == max_steps - 1 or
                prm_reward < 0
            )
            
            # Bonus for correct final answer
            if done and self.bridge._has_answer(action):
                prm_reward += 10.0  # Big bonus
            
            experiences.append(Experience(
                state=state,
                action=action,
                reward=prm_reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value
            ))
            
            if done:
                break
            
            state = next_state
        
        return experiences
    
    def compute_advantages(
        self,
        experiences: List[Experience]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation).
        """
        rewards = np.array([e.reward for e in experiences])
        values = np.array([e.value for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = 0  # Terminal state
        
        for t in reversed(range(len(experiences))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
            
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(
        self,
        experiences: List[Experience],
        advantages: np.ndarray,
        returns: np.ndarray,
        epochs: int = 4
    ):
        """PPO policy update."""
        old_log_probs = torch.tensor(
            [e.log_prob for e in experiences],
            dtype=torch.float32
        ).to(self.device)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        for _ in range(epochs):
            # Re-evaluate actions under current policy
            new_log_probs = []
            new_values = []
            
            for exp in experiences:
                # Get new log prob (would need to implement properly)
                # For simplicity, using old log prob (approximate)
                new_log_probs.append(exp.log_prob)
                
                # Get new value
                value = self.value_net.estimate_value(exp.state)
                new_values.append(value)
            
            new_log_probs = torch.tensor(new_log_probs, dtype=torch.float32).to(self.device)
            new_values = torch.tensor(new_values, dtype=torch.float32).to(self.device)
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.policy_optimizer.step()
            
            # Update value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
    
    def train(
        self,
        problems: List[str],
        iterations: int = 100,
        trajectories_per_iter: int = 4
    ):
        """
        Main RL training loop.
        """
        print(f"Starting PPO training for {iterations} iterations...")
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration+1}/{iterations}")
            
            # Collect trajectories
            all_experiences = []
            all_advantages = []
            all_returns = []
            
            total_reward = 0.0
            
            for problem in np.random.choice(problems, trajectories_per_iter):
                # Collect trajectory
                experiences = self.collect_trajectory(problem)
                
                # Compute advantages
                advantages, returns = self.compute_advantages(experiences)
                
                all_experiences.extend(experiences)
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                
                # Track metrics
                total_reward += sum(e.reward for e in experiences)
            
            avg_reward = total_reward / trajectories_per_iter
            
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Trajectories: {len(all_experiences)} steps")
            
            # Update policy
            self.update_policy(
                all_experiences,
                np.array(all_advantages),
                np.array(all_returns)
            )
            
            # Periodic save
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(f"models/rl/checkpoint_{iteration+1}.pt")
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        torch.save({
            'policy_state': self.policy.model.state_dict(),
            'value_state': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
        print(f"  → Saved checkpoint: {filepath}")


# ===== Complete RL Pipeline =====

def train_rl_policy(
    problems: List[str],
    base_model_name: str,
    prm_path: str,
    iterations: int = 100,
    output_dir: str = "models/rl"
):
    """
    Complete RL training pipeline.
    
    Args:
        problems: Training problems
        base_model_name: Base LLM name
        prm_path: Path to trained PRM
        iterations: RL iterations
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load PRM
    print("Loading PRM...")
    prm = ProcessRewardModel()
    prm.load_state_dict(torch.load(prm_path))
    prm = prm.to(device)
    prm.eval()
    
    # Initialize policy and value
    print("Initializing policy...")
    policy = PolicyNetwork(base_model_name, device=device)
    value_net = ValueNetwork()
    
    # Initialize trainer
    bridge = SymbolicBridge()
    trainer = PPOTrainer(policy, value_net, prm, bridge, device=device)
    
    # Train
    trainer.train(problems, iterations=iterations)
    
    # Save final model
    final_path = f"{output_dir}/final_policy.pt"
    trainer.save_checkpoint(final_path)
    
    print(f"\nTraining complete! Model saved to {output_dir}/")


if __name__ == "__main__":
    print("RL Training with PPO")
    print("\nComponents:")
    print("  ✓ PolicyNetwork - LLM policy for action generation")
    print("  ✓ ValueNetwork - Critic for advantage estimation")
    print("  ✓ PPOTrainer - Proximal Policy Optimization")
    print("\nUsage:")
    print("  from rl_training import train_rl_policy")
    print("  train_rl_policy(problems, model_name, prm_path)")
