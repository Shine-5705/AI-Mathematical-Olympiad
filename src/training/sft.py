"""SFT training using MLX on Apple Silicon."""
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainer:
    """LoRA SFT using MLX — runs on Mac M-series natively."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
        output_dir: str = "models/sft",
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, parquet_path: str, limit: int = None):
        """Convert parquet to JSONL files for MLX training."""
        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)

        records = []
        for _, row in df.iterrows():
            tir_text = row.get("tir_text")
            if not tir_text:
                problem = row.get("problem", "")
                solution = row.get("solution", "")
                if not problem or not solution:
                    continue
                from src.data.pipeline import format_tir_chat
                tir_text = format_tir_chat(problem, solution)
            records.append({"text": tir_text})

        split = int(len(records) * 0.9)
        valid_split = split + (len(records) - split) // 2

        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [
            ("train", records[:split]),
            ("valid", records[split:valid_split]),
            ("test", records[valid_split:]),
        ]:
            path = data_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"{name}: {len(data)} examples → {path}")

        return str(data_dir)

    def train(
        self,
        parquet_path: str,
        epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        limit: int = None,
    ) -> dict:
        """Run LoRA fine-tuning with MLX."""
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.datasets import load_dataset as load_mlx_dataset
        from src.evaluation.plots import plot_training_loss

        data_dir = self.prepare_data(parquet_path, limit)

        logger.info(f"Loading {self.model_id}...")
        model, tokenizer = load(self.model_id)

        # Apply LoRA
        lora_config = {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "scale": 32.0 / 16.0,
        }
        linear_to_lora_layers(
            model,
            num_lora_layers=16,
            config=lora_config,
        )

        trainable = sum(p.size for n, p in model.trainable_parameters().items())
        total = sum(p.size for n, p in model.parameters().items())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        # Load datasets
        train_ds, valid_ds, test_ds = load_mlx_dataset(
            data=data_dir,
            tokenizer=tokenizer,
        )

        # Callback to collect losses
        train_losses = []
        eval_losses = []

        adapter_path = str(self.output_dir / "lora_adapter")

        training_args = TrainingArguments(
            batch_size=batch_size,
            iters=len(train_ds) * epochs,
            val_batches=25,
            steps_per_report=10,
            steps_per_eval=100,
            save_every=200,
            adapter_path=adapter_path,
            learning_rate=learning_rate,
            max_seq_length=2048,
        )

        # Train
        logger.info("Starting SFT training...")
        model.train()
        mlx_train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            val_dataset=valid_ds,
        )

        # Evaluate on test set
        model.eval()
        from mlx_lm.tuner.trainer import evaluate as mlx_evaluate
        test_loss = mlx_evaluate(
            model=model,
            dataset=test_ds,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_batches=len(test_ds),
            max_seq_length=2048,
        )

        # Read training log if available
        log_path = Path(adapter_path) / "adapter_config.json"
        metrics = {
            "mode": "sft",
            "model": self.model_id,
            "trainable_params": trainable,
            "trainable_pct": round(trainable / total * 100, 2),
            "total_params": total,
            "test_loss": round(test_loss, 4),
            "lora_rank": lora_config["rank"],
            "lora_alpha": lora_config["alpha"],
            "epochs": epochs,
            "learning_rate": learning_rate,
        }

        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._print_summary(metrics)
        return metrics

    def _print_summary(self, m):
        print("\n" + "=" * 60)
        print("SFT TRAINING RESULTS (MLX)")
        print("=" * 60)
        print(f"  Model:         {m['model']}")
        print(f"  Trainable:     {m['trainable_params']:,} ({m['trainable_pct']}%)")
        print(f"  LoRA rank:     {m['lora_rank']}")
        print("-" * 60)
        print(f"  Test Loss:     {m['test_loss']}")
        print("=" * 60)
        print(f"  Adapter:       {self.output_dir / 'lora_adapter'}")
        print(f"  Metrics:       {self.output_dir / 'training_metrics.json'}")
        print("=" * 60)
