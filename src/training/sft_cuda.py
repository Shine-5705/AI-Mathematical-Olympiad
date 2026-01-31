"""SFT training using PyTorch + QLoRA on NVIDIA GPUs.

Memory-safe: reads parquet row-by-row during training via a streaming
PyTorch IterableDataset. Only the current batch is ever in RAM.
"""
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _row_to_text(row: dict) -> str | None:
    """Convert a parquet row to TIR chat text."""
    if tir_text := row.get("tir_text"):
        return tir_text
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    if not problem or not solution:
        return None
    return (
        f"<|system|>\n<|end|>\n"
        f"<|user|>\n{problem}<|end|>\n"
        f"<|assistant|>\n{solution}<|end|>"
    )


class ParquetTIRDataset(IterableDataset):
    """Streams rows from a parquet file, tokenizes on-the-fly.
    Only one row-group is in memory at a time."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 256,
                 start_row: int = 0, end_row: int | None = None):
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_row = start_row
        self.end_row = end_row
        # Count total rows for this split
        pf = pq.ParquetFile(parquet_path)
        total = pf.metadata.num_rows
        self.end_row = min(end_row, total) if end_row else total
        self.length = self.end_row - self.start_row

    def __len__(self):
        return self.length

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        global_row = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["problem", "solution", "expected_answer"])
            for row in table.to_pylist():
                if global_row >= self.end_row:
                    return
                if global_row >= self.start_row:
                    text = _row_to_text(row)
                    if text:
                        tokens = self.tokenizer(
                            text, truncation=True,
                            max_length=self.max_length,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].squeeze(0)
                        attention_mask = tokens["attention_mask"].squeeze(0)
                        yield {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": input_ids.clone(),
                        }
                global_row += 1
            del table


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

    def train(
        self,
        parquet_path: str,
        epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        limit: int = None,
    ) -> dict:
        """Run QLoRA SFT with streaming data — constant memory usage."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_scheduler,
        )
        from peft import LoraConfig, get_peft_model
        from src.evaluation.plots import plot_training_loss

        # --- Tokenizer ---
        logger.info(f"Loading {self.model_id} (4-bit)...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Figure out row counts ---
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        if limit:
            total_rows = min(total_rows, limit)
        split_row = int(total_rows * 0.9)
        logger.info(f"Train: {split_row:,} | Eval: {total_rows - split_row:,} (streaming from parquet)")

        train_ds = ParquetTIRDataset(parquet_path, tokenizer, start_row=0, end_row=split_row)
        eval_ds = ParquetTIRDataset(parquet_path, tokenizer, start_row=split_row, end_row=total_rows)

        train_loader = DataLoader(train_ds, batch_size=batch_size)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size)

        # --- Model ---
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
            trust_remote_code=True,
        )
        # Freeze base model, then apply LoRA (which marks adapters as trainable).
        for param in model.parameters():
            param.requires_grad = False
        model = get_peft_model(model, LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ))

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

        # --- Optimizer on CPU to save GPU RAM ---
        grad_accum_steps = 4
        trainable_param_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_param_list,
            lr=learning_rate, weight_decay=0.01,
        )
        total_steps = (len(train_ds) // (batch_size * grad_accum_steps)) * epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        # --- Training loop ---
        logger.info("Starting SFT training (CUDA, streaming)...")
        device = next(model.parameters()).device
        train_losses = []
        eval_losses = []
        global_step = 0
        log_interval = 5
        eval_interval = max(1, (len(train_ds) // (batch_size * grad_accum_steps)) // 2)
        best_eval_loss = float("inf")
        running_loss = 0.0

        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.empty_cache()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum_steps

                scaler.scale(loss).backward()
                running_loss += loss.item()
                del outputs, loss
                torch.cuda.empty_cache()

                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % log_interval == 0:
                        avg_loss = running_loss
                        train_losses.append((global_step, round(avg_loss, 4)))
                        logger.info(f"Epoch {epoch+1}/{epochs} step {global_step}: loss={avg_loss:.4f}")
                        running_loss = 0.0

                    if global_step % eval_interval == 0:
                        eval_loss = self._evaluate(model, eval_loader, device)
                        eval_losses.append((global_step, round(eval_loss, 4)))
                        logger.info(f"  eval_loss={eval_loss:.4f}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_path = self.output_dir / "lora_adapter"
                            model.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            logger.info(f"  Saved best model (eval_loss={eval_loss:.4f})")
                        model.train()

        # Final eval
        final_eval_loss = self._evaluate(model, eval_loader, device)
        eval_losses.append((global_step, round(final_eval_loss, 4)))

        # Save final if it's the best
        if final_eval_loss <= best_eval_loss:
            save_path = self.output_dir / "lora_adapter"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        final_train_loss = train_losses[-1][1] if train_losses else 0.0

        metrics = {
            "mode": "sft",
            "model": self.model_id,
            "backend": "cuda",
            "train_examples": split_row,
            "eval_examples": total_rows - split_row,
            "epochs": epochs,
            "trainable_params": trainable_params,
            "trainable_pct": round(trainable_params / total_params * 100, 2),
            "final_train_loss": final_train_loss,
            "final_eval_loss": round(final_eval_loss, 4),
            "eval_perplexity": round(math.exp(final_eval_loss), 2) if final_eval_loss < 10 else float('inf'),
            "train_loss_history": train_losses,
            "eval_loss_history": eval_losses,
        }

        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        plot_training_loss(train_losses, eval_losses, str(self.output_dir / "loss_curve.png"))
        self._print_summary(metrics)
        return metrics

    @torch.no_grad()
    def _evaluate(self, model, eval_loader, device) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / max(count, 1)

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
