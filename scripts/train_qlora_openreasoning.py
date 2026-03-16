#!/usr/bin/env python3
"""
Train a QLoRA adapter on OpenMathReasoning for Qwen2.5-Math-7B-Instruct.

Example (Kaggle, internet ON):
python scripts/train_qlora_openreasoning.py \
  --base-model Qwen/Qwen2.5-Math-7B-Instruct \
  --output-dir /kaggle/working/orm-7b-lora \
  --max-samples 60000
"""

import argparse
import random

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=1536)
    return p.parse_args()


def build_examples(max_samples: int) -> list[dict]:
    stream = load_dataset("nvidia/OpenMathReasoning", split="train", streaming=True)
    examples = []
    for row in stream:
        if row.get("problem_type") != "has_answer_extracted":
            continue
        if row.get("generated_solution") is None or row.get("expected_answer") is None:
            continue
        mode = row.get("inference_mode")
        if mode not in {"tir", "genselect", "cot"}:
            continue
        examples.append(
            {
                "problem": row["problem"],
                "solution": row["generated_solution"],
                "expected_answer": str(row["expected_answer"]),
                "mode": mode,
            }
        )
        if len(examples) >= max_samples:
            break
    return examples


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = build_examples(args.max_samples)
    random.shuffle(raw)

    def to_text(example: dict) -> dict:
        system = (
            "Solve the math problem. Use tool-integrated reasoning when useful. "
            "Put the final integer answer in \\boxed{}."
        )
        assistant = (
            f"{example['solution']}\n\n"
            f"Final answer: \\boxed{{{example['expected_answer']}}}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(raw).map(
        to_text,
        remove_columns=["problem", "solution", "expected_answer", "mode"],
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        peft_config=peft_config,
        args=train_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
