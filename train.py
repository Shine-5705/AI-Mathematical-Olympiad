"""
Fine-tune NuminaMath on OpenMathReasoning.

Usage:
    python train.py --data data/processed/math_reasoning_tir.parquet --epochs 3
    python train.py --limit 100 --epochs 1   # Quick test run
"""
import argparse
from src.training import FineTuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/math_reasoning_tir.parquet")
    parser.add_argument("--model", default="AI-MO/NuminaMath-7B-TIR")
    parser.add_argument("--output", default="models/numina-openmath")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    trainer = FineTuner(model_id=args.model, output_dir=args.output)
    metrics = trainer.train(
        parquet_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
