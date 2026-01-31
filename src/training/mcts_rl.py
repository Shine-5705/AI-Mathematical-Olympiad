"""
MCTS-guided Rejection Sampling + RL Fine-Tuning (MLX).

1. Load SFT-trained model
2. Generate solution paths using MCTS
3. Verify with symbolic + code execution
4. Keep only verified-correct solutions
5. Fine-tune on verified data
"""
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCTSRLTrainer:
    """MCTS-guided rejection sampling then RL fine-tuning on MLX."""

    def __init__(
        self,
        sft_adapter_path: str = "models/sft/lora_adapter",
        base_model_id: str = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
        output_dir: str = "models/mcts_rl",
    ):
        self.sft_adapter_path = Path(sft_adapter_path)
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
        from mlx_lm import load, generate
        from src.search.mcts import AIMO_MCTS
        from src.search.verifier import CompositeVerifier, SymbolicVerifier, CodeExecutionVerifier
        from src.search.generator import MLXGenerator
        from src.evaluation.metrics import evaluate_answer
        from src.data.pipeline import format_tir_chat

        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)

        # Load SFT model with adapter
        logger.info(f"Loading {self.base_model_id} with SFT adapter...")
        generator = MLXGenerator(model_id=self.base_model_id)

        # Load adapter weights if available
        if self.sft_adapter_path.exists():
            from mlx_lm.tuner.utils import apply_lora_layers
            import mlx.core as mx
            adapter_weights = list(Path(self.sft_adapter_path).glob("adapters*.safetensors"))
            if adapter_weights:
                logger.info(f"Loading adapter from {self.sft_adapter_path}")

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
            for c in range(n_candidates):
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
        learning_rate: float = 5e-6,
        n_candidates: int = 5,
        mcts_iterations: int = 8,
        limit: int = None,
    ) -> dict:
        """Full pipeline: generate verified data → fine-tune."""
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
        from mlx_lm.tuner.trainer import evaluate as mlx_evaluate
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.datasets import load_dataset as load_mlx_dataset

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
            logger.error(f"Only {len(verified)} verified solutions — need more data or iterations.")
            return {"error": "insufficient_data", "count": len(verified)}

        # Step 2: Write JSONL for MLX
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        split = int(len(verified) * 0.85)
        valid_split = split + (len(verified) - split) // 2

        for name, data in [
            ("train", verified[:split]),
            ("valid", verified[split:valid_split]),
            ("test", verified[valid_split:]),
        ]:
            path = data_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for r in data:
                    f.write(json.dumps({"text": r["text"]}) + "\n")
            logger.info(f"{name}: {len(data)} examples")

        # Step 3: Load model with SFT adapter
        logger.info(f"Loading {self.base_model_id}...")
        model, tokenizer = load(self.base_model_id)

        # Apply fresh LoRA (lower rank for RL stage)
        lora_config = {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "scale": 16.0 / 8.0,
        }
        linear_to_lora_layers(model, num_lora_layers=8, config=lora_config)

        # Load SFT adapter weights into base if available
        if self.sft_adapter_path.exists():
            adapter_file = self.sft_adapter_path / "adapters.safetensors"
            if adapter_file.exists():
                logger.info("Merging SFT adapter weights")
                from mlx.utils import tree_unflatten
                import mlx.nn as nn
                weights = mx.load(str(adapter_file))
                model.load_weights(list(weights.items()), strict=False)

        trainable = sum(p.size for n, p in model.trainable_parameters().items())
        total = sum(p.size for n, p in model.parameters().items())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        # Step 4: Load datasets
        train_ds, valid_ds, test_ds = load_mlx_dataset(
            data=str(data_dir),
            tokenizer=tokenizer,
        )

        adapter_path = str(self.output_dir / "lora_adapter")

        training_args = TrainingArgs(
            batch_size=batch_size,
            iters=len(train_ds) * epochs,
            val_batches=min(25, len(valid_ds)),
            steps_per_report=10,
            steps_per_eval=50,
            save_every=100,
            adapter_path=adapter_path,
            learning_rate=learning_rate,
            max_seq_length=2048,
        )

        # Step 5: Train on verified solutions
        logger.info("Starting MCTS-RL training...")
        model.train()
        mlx_train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            val_dataset=valid_ds,
        )

        model.eval()
        test_loss = mlx_evaluate(
            model=model,
            dataset=test_ds,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_batches=len(test_ds),
            max_seq_length=2048,
        )

        metrics = {
            "mode": "mcts_rl",
            "base_model": self.base_model_id,
            "sft_adapter": str(self.sft_adapter_path),
            "verified_solutions": len(verified),
            "trainable_params": trainable,
            "trainable_pct": round(trainable / total * 100, 2),
            "test_loss": round(test_loss, 4),
            "lora_rank": lora_config["rank"],
            "epochs": epochs,
        }

        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._print_summary(metrics)
        return metrics

    def _print_summary(self, m):
        print("\n" + "=" * 60)
        print("MCTS-RL TRAINING RESULTS (MLX)")
        print("=" * 60)
        print(f"  Base Model:    {m['base_model']}")
        print(f"  SFT Adapter:   {m['sft_adapter']}")
        print(f"  Verified Data: {m['verified_solutions']}")
        print(f"  Trainable:     {m['trainable_params']:,} ({m['trainable_pct']}%)")
        print("-" * 60)
        print(f"  Test Loss:     {m['test_loss']}")
        print("=" * 60)
        print(f"  Adapter:       {self.output_dir / 'lora_adapter'}")
        print("=" * 60)
