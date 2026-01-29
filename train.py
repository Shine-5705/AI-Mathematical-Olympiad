"""
Train NuminaMath on OpenMathReasoning.

Two modes:
  sft      - Standard supervised fine-tuning (what Numina did)
  mcts-rl  - MCTS-guided rejection sampling + RL (our contribution)

Usage:
    python train.py --mode sft --epochs 3
    python train.py --mode mcts-rl --sft-model models/sft/lora_adapter
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train math reasoning model")
    parser.add_argument("--mode", choices=["sft", "mcts-rl"], required=True)
    parser.add_argument("--data", default="data/processed/math_reasoning_tir.parquet")
    parser.add_argument("--model", default="AI-MO/NuminaMath-7B-TIR")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=None)

    # MCTS-RL specific
    parser.add_argument("--sft-model", default="models/sft/lora_adapter")
    parser.add_argument("--mcts-iterations", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "sft":
        from src.training import SFTTrainer

        trainer = SFTTrainer(model_id=args.model, output_dir="models/sft")
        trainer.train(
            parquet_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            limit=args.limit,
        )

    elif args.mode == "mcts-rl":
        from src.training import MCTSRLTrainer

        trainer = MCTSRLTrainer(
            sft_model_path=args.sft_model,
            base_model_id=args.model,
            output_dir="models/mcts_rl",
        )
        trainer.train(
            parquet_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            n_candidates=args.n_candidates,
            mcts_iterations=args.mcts_iterations,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
