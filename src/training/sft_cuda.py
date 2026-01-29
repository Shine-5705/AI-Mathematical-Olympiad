"""SFT training using PyTorch + QLoRA on NVIDIA GPUs."""
import json
import logging
import math
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainerCUDA:
    """QLoRA SFT for NuminaMath on NVIDIA GPUs (RTX 3060+)."""

    def __init__(
        self,
        model_id: str = "AI-MO/NuminaMath-7B-TIR",
        output_dir: str = "models/sft",
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, parquet_path: str, limit: int = None):
        """Load TIR data and split train/eval."""
        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)

        dataset = []
        for _, row in df.iterrows():
            tir_text = row.get("tir_text")
            if not tir_text:
                problem = row.get("problem", "")
                solution = row.get("solution", "")
                if not problem or not solution:
                    continue
                from src.data.pipeline import format_tir_chat
                tir_text = format_tir_chat(problem, solution)

            dataset.append({
                "text": tir_text,
                "expected_answer": str(row.get("expected_answer", "")),
            })

        split = int(len(dataset) * 0.9)
        logger.info(f"Train: {split} | Eval: {len(dataset) - split}")
        return dataset[:split], dataset[split:]

    def train(
        self,
        parquet_path: str,
        epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        limit: int = None,
    ) -> dict:
        """Run QLoRA SFT."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from src.evaluation.plots import plot_training_loss

        train_data, eval_data = self.prepare_dataset(parquet_path, limit)

        logger.info(f"Loading {self.model_id} (4-bit)...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

        def tokenize(example):
            tokens = tokenizer(example["text"], truncation=True, max_length=2048, padding="max_length")
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        train_ds = Dataset.from_list(train_data).map(tokenize, remove_columns=["text", "expected_answer"])
        eval_ds = Dataset.from_list(eval_data).map(tokenize, remove_columns=["text", "expected_answer"])

        steps_per_epoch = max(1, len(train_ds) // (batch_size * 4))
        eval_steps = max(1, steps_per_epoch // 2)

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

        logger.info("Starting SFT training (CUDA)...")
        result = trainer.train()

        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]
        log_history = trainer.state.log_history
        train_losses = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
        eval_losses = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]

        metrics = {
            "mode": "sft",
            "model": self.model_id,
            "backend": "cuda",
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
        print("SFT TRAINING RESULTS (CUDA)")
        print("=" * 60)
        print(f"  Model:         {m['model']}")
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
