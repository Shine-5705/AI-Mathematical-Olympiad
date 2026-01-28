"""Run experiments comparing MCTS with baselines."""
import argparse

from src.search import AIMO_MCTS, DirectPrompting, ChainOfThought, SelfConsistency
from src.evaluation import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run AIMO experiments")
    parser.add_argument("--dataset", type=str, default="data/processed/math_reasoning_tir.parquet")
    parser.add_argument("--limit", type=int, default=10, help="Number of problems")
    parser.add_argument("--backend", type=str, default="auto", choices=["transformers", "mlx", "vllm", "auto"])
    parser.add_argument("--model", type=str, default=None, help="Model ID")
    parser.add_argument("--iterations", type=int, default=5, help="MCTS iterations")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["direct", "cot", "mcts"],
                        choices=["direct", "cot", "self_consistency", "mcts", "mcts_no_verify"])
    args = parser.parse_args()

    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Problems: {args.limit}")
    print(f"Methods: {args.methods}")

    kwargs = {"model_id": args.model} if args.model else {}

    available_methods = {
        "direct": lambda: (DirectPrompting(backend=args.backend, **kwargs), "solve"),
        "cot": lambda: (ChainOfThought(backend=args.backend, **kwargs), "solve"),
        "self_consistency": lambda: (SelfConsistency(backend=args.backend, **kwargs), "solve"),
        "mcts": lambda: (AIMO_MCTS(backend=args.backend, verify_mode="symbolic", **kwargs), "search"),
        "mcts_no_verify": lambda: (AIMO_MCTS(backend=args.backend, verify_mode="none", **kwargs), "search"),
    }

    methods = {name: available_methods[name]() for name in args.methods}

    runner = ExperimentRunner()
    runner.run_experiment(
        methods=methods,
        dataset_path=args.dataset,
        limit=args.limit,
        experiment_name=args.name,
    )


if __name__ == "__main__":
    main()
