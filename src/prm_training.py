"""
Process Reward Model (PRM) Implementation
Scores individual reasoning steps for RL training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json

from symbolic_bridge import SymbolicBridge, MathState


@dataclass
class StepLabel:
    """Label for a single reasoning step."""
    state_text: str
    action_text: str  # The step taken
    reward: float  # 0.0 to 1.0 (or -1.0 for invalid)
    is_correct: bool
    verification_result: str
    next_state: Optional[str] = None


class StepDataset(Dataset):
    """Dataset of reasoning steps with rewards."""
    
    def __init__(self, steps: List[StepLabel], tokenizer, max_length=512):
        self.steps = steps
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, idx):
        step = self.steps[idx]
        
        # Concatenate state + action
        text = f"State: {step.state_text}\nAction: {step.action_text}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'reward': torch.tensor(step.reward, dtype=torch.float32),
            'is_correct': torch.tensor(1.0 if step.is_correct else 0.0, dtype=torch.float32)
        }


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model (PRM) that scores individual reasoning steps.
    
    Based on "Let's Verify Step by Step" (OpenAI, 2023)
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # Regression: predict reward score
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Sigmoid to constrain to [0, 1]
        return torch.sigmoid(outputs.logits)
    
    def score_step(self, state_text: str, action_text: str) -> float:
        """Score a single step."""
        text = f"State: {state_text}\nAction: {action_text}"
        
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        self.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                encoding = {k: v.cuda() for k, v in encoding.items()}
            
            score = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            )
        
        return score.item()


class PRMTrainer:
    """Trainer for Process Reward Model."""
    
    def __init__(
        self,
        model: ProcessRewardModel,
        symbolic_bridge: SymbolicBridge,
        learning_rate: float = 2e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.bridge = symbolic_bridge
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Loss: MSE for reward regression + BCE for correctness
        self.reward_loss_fn = nn.MSELoss()
        self.correct_loss_fn = nn.BCELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_reward_loss = 0.0
        total_correct_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            rewards = batch['reward'].to(self.device)
            is_correct = batch['is_correct'].to(self.device)
            
            # Forward pass
            predicted_rewards = self.model(input_ids, attention_mask).squeeze()
            
            # Compute losses
            reward_loss = self.reward_loss_fn(predicted_rewards, rewards)
            correct_loss = self.correct_loss_fn(predicted_rewards, is_correct)
            
            # Combined loss
            loss = reward_loss + 0.5 * correct_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_reward_loss += reward_loss.item()
            total_correct_loss += correct_loss.item()
            n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'reward_loss': total_reward_loss / n_batches,
            'correct_loss': total_correct_loss / n_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                rewards = batch['reward'].to(self.device)
                is_correct = batch['is_correct'].to(self.device)
                
                predicted_rewards = self.model(input_ids, attention_mask).squeeze()
                
                loss = self.reward_loss_fn(predicted_rewards, rewards)
                total_loss += loss.item()
                
                # Accuracy: threshold at 0.5
                predictions = (predicted_rewards > 0.5).float()
                correct_predictions += (predictions == is_correct).sum().item()
                total_predictions += is_correct.size(0)
        
        return {
            'eval_loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions
        }


class DataGenerator:
    """
    Generate training data for PRM by running LLM + symbolic verification.
    """
    
    def __init__(self, generator, bridge: SymbolicBridge):
        self.generator = generator
        self.bridge = bridge
    
    def generate_trajectory(
        self,
        problem: str,
        max_steps: int = 10
    ) -> List[StepLabel]:
        """
        Generate a complete solution trajectory with step-level rewards.
        """
        trajectory = []
        current_state_text = problem
        
        for step_num in range(max_steps):
            # Generate next step
            candidates = self.generator.get_candidates(current_state_text, n=1)
            if not candidates:
                break
            
            action_text = candidates[0]
            
            # Parse state
            full_text = current_state_text + "\n" + action_text
            states = self.bridge.extract_reasoning_chain(full_text)
            
            if not states:
                # Failed to parse
                trajectory.append(StepLabel(
                    state_text=current_state_text,
                    action_text=action_text,
                    reward=-1.0,
                    is_correct=False,
                    verification_result="parse_failed"
                ))
                break
            
            current_state = states[-1]
            
            # Verify step
            is_valid, reason = current_state.verify_consistency()
            
            # Compute reward
            if not is_valid:
                reward = -1.0
                is_correct = False
            elif self.bridge._has_answer(action_text):
                # Reached answer - big bonus
                reward = 1.0
                is_correct = True
            else:
                # Valid intermediate step
                reward = 0.5
                is_correct = True
            
            trajectory.append(StepLabel(
                state_text=current_state_text,
                action_text=action_text,
                reward=reward,
                is_correct=is_correct,
                verification_result=reason,
                next_state=full_text
            ))
            
            # Update state
            current_state_text = full_text
            
            # Stop if answer reached
            if self.bridge._has_answer(action_text):
                break
        
        return trajectory
    
    def generate_dataset(
        self,
        problems: List[str],
        trajectories_per_problem: int = 3
    ) -> List[StepLabel]:
        """Generate full dataset from problems."""
        all_steps = []
        
        for i, problem in enumerate(problems):
            print(f"Generating trajectories for problem {i+1}/{len(problems)}...")
            
            for _ in range(trajectories_per_problem):
                trajectory = self.generate_trajectory(problem)
                all_steps.extend(trajectory)
        
        print(f"Generated {len(all_steps)} step labels")
        return all_steps
    
    def save_dataset(self, steps: List[StepLabel], filepath: str):
        """Save dataset to JSON."""
        data = [
            {
                'state': s.state_text,
                'action': s.action_text,
                'reward': s.reward,
                'is_correct': s.is_correct,
                'verification': s.verification_result
            }
            for s in steps
        ]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filepath: str) -> List[StepLabel]:
        """Load dataset from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [
            StepLabel(
                state_text=d['state'],
                action_text=d['action'],
                reward=d['reward'],
                is_correct=d['is_correct'],
                verification_result=d['verification']
            )
            for d in data
        ]


# ===== Training Pipeline =====

def train_prm(
    problems: List[str],
    generator,
    epochs: int = 5,
    batch_size: int = 16,
    output_dir: str = "models/prm"
):
    """
    Complete PRM training pipeline.
    
    Args:
        problems: List of training problems
        generator: LLM generator
        epochs: Training epochs
        batch_size: Batch size
        output_dir: Where to save model
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    bridge = SymbolicBridge()
    data_gen = DataGenerator(generator, bridge)
    
    # Generate training data
    print("Generating training data...")
    steps = data_gen.generate_dataset(problems, trajectories_per_problem=3)
    
    # Save dataset
    data_gen.save_dataset(steps, f"{output_dir}/training_data.json")
    
    # Split train/val
    split_idx = int(0.9 * len(steps))
    train_steps = steps[:split_idx]
    val_steps = steps[split_idx:]
    
    # Create model
    prm = ProcessRewardModel()
    
    # Create datasets
    train_dataset = StepDataset(train_steps, prm.tokenizer)
    val_dataset = StepDataset(val_steps, prm.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train
    trainer = PRMTrainer(prm, bridge)
    
    print(f"\nTraining PRM on {len(train_steps)} steps...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        print(f"  Val Loss: {val_metrics['eval_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.3f}")
        
        # Save best model
        if val_metrics['eval_loss'] < best_loss:
            best_loss = val_metrics['eval_loss']
            torch.save(prm.state_dict(), f"{output_dir}/best_prm.pt")
            print("  → Saved best model")
    
    print(f"\nTraining complete. Model saved to {output_dir}/")


if __name__ == "__main__":
    print("Process Reward Model (PRM) Implementation")
    print("\nComponents:")
    print("  ✓ ProcessRewardModel - Scores individual steps")
    print("  ✓ DataGenerator - Generates labeled trajectories")
    print("  ✓ PRMTrainer - Trains the reward model")
    print("\nUsage:")
    print("  from prm_training import train_prm")
    print("  train_prm(problems, generator)")
