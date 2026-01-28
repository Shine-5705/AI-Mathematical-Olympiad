import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from src.evaluation.metrics import evaluate_answer, compute_aggregate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs experiments comparing different solving methods."""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def load_dataset(self, path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load evaluation dataset."""
        df = pd.read_parquet(path)
        if limit:
            df = df.head(limit)
        logger.info(f"Loaded {len(df)} problems from {path}")
        return df

    def run_method(
        self,
        method_name: str,
        solver,
        problems: pd.DataFrame,
        solve_fn: str = "solve",
    ) -> List[Dict[str, Any]]:
        """Run a solving method on all problems."""
        results = []

        for idx, row in problems.iterrows():
            problem = row.get("instruction") or row.get("problem")
            expected = str(row.get("target") or row.get("expected_answer", ""))

            logger.info(f"[{method_name}] Problem {idx + 1}/{len(problems)}")

            try:
                solve_method = getattr(solver, solve_fn, None) or getattr(solver, "search")
                solution = solve_method(problem)
                metrics = solver.get_metrics()

                eval_result = evaluate_answer(solution, expected)

                result = {
                    "method": method_name,
                    "problem_idx": idx,
                    "problem": problem[:200] + "..." if len(problem) > 200 else problem,
                    "expected": expected,
                    "solution": solution,
                    "metrics": metrics,
                    **eval_result,
                }
                results.append(result)

                status = "✓" if eval_result["correct"] else "✗"
                logger.info(f"  {status} Expected: {expected}, Got: {eval_result['predicted_raw']}")

            except Exception as e:
                logger.error(f"  Error: {e}")
                results.append({
                    "method": method_name,
                    "problem_idx": idx,
                    "error": str(e),
                    "correct": False,
                })

        return results

    def run_experiment(
        self,
        methods: Dict[str, Any],
        dataset_path: str,
        limit: Optional[int] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full experiment comparing multiple methods."""
        problems = self.load_dataset(dataset_path, limit)
        experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        all_results = {}
        for method_name, (solver, solve_fn) in methods.items():
            logger.info(f"\n{'='*50}\nRunning {method_name}\n{'='*50}")
            results = self.run_method(method_name, solver, problems, solve_fn)
            all_results[method_name] = {
                "results": results,
                "aggregate": compute_aggregate_metrics(results),
            }

        self._save_results(experiment_name, all_results)
        self._print_summary(all_results)

        return all_results

    def _save_results(self, name: str, results: Dict[str, Any]) -> None:
        """Save experiment results to JSON."""
        output_path = self.output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary table of results."""
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"{'Method':<20} {'Accuracy':<12} {'Avg Tokens':<12} {'Avg Time':<12} {'Prune Rate':<12}")
        print("-" * 70)

        for method, data in results.items():
            agg = data["aggregate"]
            print(f"{method:<20} {agg['accuracy']:<12.2%} {agg['avg_tokens']:<12.0f} {agg['avg_time']:<12.2f}s {agg.get('prune_rate', 0):<12.2%}")

        print("=" * 70)
