"""
Compare SFT model vs MCTS-RL model.

Usage:
    python run_experiment.py --limit 10
    python run_experiment.py --sft-model models/sft/lora_adapter --mcts-model models/mcts_rl/lora_adapter --limit 20
"""
import argparse

from src.search import AIMO_MCTS, create_generator, create_verifier
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Compare SFT vs MCTS-RL models")
    parser.add_argument("--dataset", default="data/processed/math_reasoning_tir.parquet")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--backend", default="auto", choices=["transformers", "mlx", "vllm", "auto"])
    parser.add_argument("--base-model", default="AI-MO/NuminaMath-7B-TIR")
    parser.add_argument("--sft-model", default="models/sft/lora_adapter")
    parser.add_argument("--mcts-model", default="models/mcts_rl/lora_adapter")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--name", default=None)
    parser.add_argument("--methods", nargs="+",
                        default=["base", "sft", "mcts_rl"],
                        choices=["base", "sft", "mcts_rl"])
    args = parser.parse_args()

    print(f"Backend:  {args.backend}")
    print(f"Dataset:  {args.dataset}")
    print(f"Problems: {args.limit}")
    print(f"Methods:  {args.methods}")
    print()

    methods = {}

    # Baseline: original NuminaMath without fine-tuning
    if "base" in args.methods:
        methods["NuminaMath (base)"] = (
            AIMO_MCTS(backend=args.backend, verify_mode="symbolic", model_id=args.base_model),
            "search",
        )

    # SFT: fine-tuned on OpenMathReasoning
    if "sft" in args.methods:
        methods["SFT"] = (
            AIMO_MCTS(backend=args.backend, verify_mode="symbolic", model_id=args.sft_model),
            "search",
        )

    # MCTS-RL: fine-tuned with verified MCTS solutions
    if "mcts_rl" in args.methods:
        methods["MCTS-RL (ours)"] = (
            AIMO_MCTS(backend=args.backend, verify_mode="both", model_id=args.mcts_model),
            "search",
        )

    runner = ExperimentRunner()
    runner.run_experiment(
        methods=methods,
        dataset_path=args.dataset,
        limit=args.limit,
        experiment_name=args.name,
    )


if __name__ == "__main__":
    main()
