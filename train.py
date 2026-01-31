"""
Train math reasoning model.

Mac (MLX):    Qwen2.5-Math-7B-Instruct-4bit
Linux (CUDA): NuminaMath-7B-TIR with QLoRA

Usage:
    python train.py --mode sft --epochs 3
    python train.py --mode mcts-rl --sft-model models/sft/lora_adapter
"""
import argparse
import platform
from pathlib import Path


def detect_platform() -> dict:
    """Detect hardware and return default model + backend."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "model": "AI-MO/NuminaMath-7B-TIR",
                "backend": "cuda",
            }
    except ImportError:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Prefer local converted NuminaMath if available
        local_numina = Path("models/mlx/NuminaMath-7B-TIR-4bit")
        if local_numina.exists():
            return {
                "model": str(local_numina),
                "backend": "mlx",
            }
        return {
            "model": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
            "backend": "mlx",
        }

    return {
        "model": "AI-MO/NuminaMath-7B-TIR",
        "backend": "cpu",
    }


def main():
    defaults = detect_platform()

    parser = argparse.ArgumentParser(description="Train math reasoning model")
    parser.add_argument("--mode", choices=["sft", "mcts-rl"], required=True)
    parser.add_argument("--data", default="data/processed/math_reasoning_tir.parquet")
    parser.add_argument("--model", default=None, help=f"Default: {defaults['model']}")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=None)

    # MCTS-RL specific
    parser.add_argument("--sft-model", default="models/sft/lora_adapter")
    parser.add_argument("--mcts-iterations", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=5)

    args = parser.parse_args()
    model_id = args.model or defaults["model"]
    backend = defaults["backend"]

    print(f"Platform: {backend}")
    print(f"Model:    {model_id}")
    print(f"Mode:     {args.mode}")
    print()

    if args.mode == "sft":
        if backend == "mlx":
            from src.training.sft import SFTTrainer
            trainer = SFTTrainer(model_id=model_id, output_dir="models/sft")
        else:
            from src.training.sft_cuda import SFTTrainerCUDA
            trainer = SFTTrainerCUDA(model_id=model_id, output_dir="models/sft")

        trainer.train(
            parquet_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            limit=args.limit,
        )

    elif args.mode == "mcts-rl":
        if backend == "mlx":
            from src.training.mcts_rl import MCTSRLTrainer
            trainer = MCTSRLTrainer(
                sft_adapter_path=args.sft_model,
                base_model_id=model_id,
                output_dir="models/mcts_rl",
            )
        else:
            from src.training.mcts_rl_cuda import MCTSRLTrainerCUDA
            trainer = MCTSRLTrainerCUDA(
                sft_model_path=args.sft_model,
                base_model_id=model_id,
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
