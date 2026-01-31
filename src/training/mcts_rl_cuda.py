"""MCTS-guided Rejection Sampling + RL Fine-Tuning (CUDA)."""
import json
import logging
import math
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCTSRLTrainerCUDA:
    """MCTS-guided rejection sampling then RL fine-tuning on NVIDIA GPUs."""

    def __init__(
        self,
        sft_model_path: str = "models/sft/lora_adapter",
        base_model_id: str = "AI-MO/NuminaMath-7B-TIR",
        output_dir: str = "models/mcts_rl",
    ):
        self.sft_model_path = Path(sft_model_path)
        self.base_model_id = base_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_verified_solutions(
        self,
        parquet_path: str,
        n_candidates: int = 5,
        mcts_iterations: int = 8,
        limit: int = None,
    ) -> list[dict]:
        """Generate solutions with MCTS, keep verified-correct ones."""
        from src.search.mcts import AIMO_MCTS
        from src.search.verifier import CompositeVerifier, SymbolicVerifier, CodeExecutionVerifier
        from src.search.generator import TransformersGenerator
        from src.evaluation.metrics import evaluate_answer
        from src.data.pipeline import format_tir_chat

        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)

        # Load SFT model for generation
        model_path = str(self.sft_model_path) if self.sft_model_path.exists() else self.base_model_id
        logger.info(f"Loading generator: {model_path}")
        generator = TransformersGenerator(model_id=model_path)

        verifier = CompositeVerifier([SymbolicVerifier(), CodeExecutionVerifier()])
        engine = AIMO_MCTS(generator=generator, verifier=verifier)

        verified = []
        stats = {"total": 0, "correct": 0}

        for idx, row in df.iterrows():
            problem = row.get("problem", "")
            expected = str(row.get("expected_answer", ""))
            stats["total"] += 1

            logger.info(f"Problem {idx + 1}/{len(df)}")

            correct_solutions = []
            for _ in range(n_candidates):
                solution = engine.search(problem, iterations=mcts_iterations)
                metrics = engine.get_metrics()
                result = evaluate_answer(solution, expected)

                if result["correct"]:
                    correct_solutions.append({
                        "solution": solution,
                        "prune_rate": metrics.get("prune_rate", 0),
                    })

            if correct_solutions:
                stats["correct"] += 1
                best = max(correct_solutions, key=lambda s: s["prune_rate"])
                verified.append({
                    "text": format_tir_chat(problem, best["solution"]),
                    "expected_answer": expected,
                })
                logger.info(f"  VERIFIED ({len(correct_solutions)}/{n_candidates})")
            else:
                logger.info(f"  FAILED (0/{n_candidates})")

        logger.info(f"Verified: {stats['correct']}/{stats['total']}")

        gen_path = self.output_dir / "verified_solutions.json"
        with open(gen_path, "w") as f:
            json.dump(verified, f, indent=2)

        return verified

    def train(
        self,
        parquet_path: str,
        epochs: int = 2,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        n_candidates: int = 5,
        mcts_iterations: int = 8,
        limit: int = None,
    ) -> dict:
        """Full pipeline: generate verified data → QLoRA fine-tune."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
        from datasets import Dataset
        from src.evaluation.plots import plot_training_loss

        # Step 1: Generate or load verified solutions
        gen_path = self.output_dir / "verified_solutions.json"
        if gen_path.exists():
            logger.info(f"Loading cached verified solutions from {gen_path}")
            with open(gen_path) as f:
                verified = json.load(f)
        else:
            verified = self.generate_verified_solutions(
                parquet_path, n_candidates, mcts_iterations, limit
            )

        if len(verified) < 3:
            logger.error(f"Only {len(verified)} verified solutions — need more data.")
            return {"error": "insufficient_data", "count": len(verified)}

        # Split
        split = max(1, int(len(verified) * 0.9))
        train_data = verified[:split]
        eval_data = verified[split:]
        logger.info(f"MCTS-RL Train: {len(train_data)} | Eval: {len(eval_data)}")

        # Step 2: Load base model
        logger.info(f"Loading {self.base_model_id} (4-bit)...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        # Merge SFT adapter if available
        if self.sft_model_path.exists():
            logger.info(f"Loading SFT adapter from {self.sft_model_path}")
            model = PeftModel.from_pretrained(model, str(self.sft_model_path))
            model = model.merge_and_unload()
            model = prepare_model_for_kbit_training(model)

        # New LoRA for RL stage (lower rank)
        model = get_peft_model(model, LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Step 3: Tokenize
        def tokenize(example):
            tokens = tokenizer(example["text"], truncation=True, max_length=2048, padding="max_length")
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        train_ds = Dataset.from_list(train_data).map(tokenize, remove_columns=["text", "expected_answer"])
        eval_ds = Dataset.from_list(eval_data).map(tokenize, remove_columns=["text", "expected_answer"])

        steps_per_epoch = max(1, len(train_ds) // (batch_size * 4))
        eval_steps = max(1, steps_per_epoch // 2)

        # Step 4: Train on verified solutions
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                fp16=True,
                logging_steps=5,
                eval_strategy="steps",
                eval_steps=eval_steps,
                save_steps=eval_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="none",
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        logger.info("Starting MCTS-RL training (CUDA)...")
        result = trainer.train()

        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]
        log_history = trainer.state.log_history
        train_losses = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
        eval_losses = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]

        metrics = {
            "mode": "mcts_rl",
            "backend": "cuda",
            "base_model": self.base_model_id,
            "sft_adapter": str(self.sft_model_path),
            "verified_solutions": len(verified),
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds),
            "epochs": epochs,
            "trainable_params": trainable_params,
            "trainable_pct": round(trainable_params / total_params * 100, 2),
            "final_train_loss": round(result.training_loss, 4),
            "final_eval_loss": round(eval_loss, 4),
            "eval_perplexity": round(math.exp(eval_loss), 2) if eval_loss < 10 else float('inf'),
            "train_loss_history": train_losses,
            "eval_loss_history": eval_losses,
        }

        save_path = self.output_dir / "lora_adapter"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        plot_training_loss(train_losses, eval_losses, str(self.output_dir / "loss_curve.png"))

        self._print_summary(metrics)
        return metrics

    def _print_summary(self, m):
        print("\n" + "=" * 60)
        print("MCTS-RL TRAINING RESULTS (CUDA)")
        print("=" * 60)
        print(f"  Base Model:    {m['base_model']}")
        print(f"  SFT Adapter:   {m['sft_adapter']}")
        print(f"  Verified Data: {m['verified_solutions']}")
        print(f"  Train/Eval:    {m['train_examples']} / {m['eval_examples']}")
        print(f"  Trainable:     {m['trainable_params']:,} ({m['trainable_pct']}%)")
        print("-" * 60)
        print(f"  Train Loss:    {m['final_train_loss']}")
        print(f"  Eval Loss:     {m['final_eval_loss']}")
        print(f"  Perplexity:    {m['eval_perplexity']}")
        print("=" * 60)
        print(f"  Saved to:      {self.output_dir / 'lora_adapter'}")
        print(f"  Loss curve:    {self.output_dir / 'loss_curve.png'}")
        print("=" * 60)
