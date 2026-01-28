"""Experiment runner that produces real metrics and plots."""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from src.evaluation.metrics import evaluate_answer, compute_aggregate_metrics
from src.evaluation.plots import generate_all_experiment_plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run experiments and produce result tables + plots."""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load evaluation dataset."""
        df = pd.read_parquet(path)
        if limit:
            df = df.head(limit)
        logger.info(f"Loaded {len(df)} problems")
        return df

    def run_method(
        self,
        method_name: str,
        solver,
        problems: pd.DataFrame,
        solve_fn: str = "solve",
    ) -> List[Dict[str, Any]]:
        """Run a method on all problems and collect results."""
        results = []

        for idx, row in problems.iterrows():
            problem = row.get("instruction") or row.get("problem")
            expected = str(row.get("target") or row.get("expected_answer", ""))

            logger.info(f"[{method_name}] Problem {idx + 1}/{len(problems)}")

            try:
                fn = getattr(solver, solve_fn, None) or getattr(solver, "search")
                solution = fn(problem)
                metrics = solver.get_metrics()
                eval_result = evaluate_answer(solution, expected)

                result = {
                    "method": method_name,
                    "problem_idx": int(idx),
                    "expected": expected,
                    "predicted": eval_result["predicted_raw"],
                    "correct": eval_result["correct"],
                    "metrics": metrics,
                }
                results.append(result)

                mark = "CORRECT" if eval_result["correct"] else "WRONG"
                logger.info(f"  {mark} | Expected: {expected} | Got: {eval_result['predicted_raw']}")

            except Exception as e:
                logger.error(f"  Error: {e}")
                results.append({
                    "method": method_name,
                    "problem_idx": int(idx),
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
        """Run full experiment and output results."""
        problems = self.load_dataset(dataset_path, limit)
        name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        all_results = {}
        for method_name, (solver, solve_fn) in methods.items():
            logger.info(f"\n{'='*50}\n{method_name}\n{'='*50}")
            results = self.run_method(method_name, solver, problems, solve_fn)
            aggregate = compute_aggregate_metrics(results)
            all_results[method_name] = {
                "results": results,
                "aggregate": aggregate,
            }

        # Save raw results
        results_path = self.output_dir / f"{name}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save summary CSV
        summary = self._build_summary_table(all_results)
        csv_path = self.output_dir / f"{name}_summary.csv"
        summary.to_csv(csv_path, index=False)

        # Generate all plots
        generate_all_experiment_plots(all_results, str(self.output_dir), name)

        # Print results
        self._print_results(all_results, name)

        return all_results

    def _build_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Build summary DataFrame."""
        rows = []
        for method, data in results.items():
            agg = data["aggregate"]
            rows.append({
                "Method": method,
                "Accuracy": agg.get("accuracy", 0),
                "Correct": agg.get("n_correct", 0),
                "Total": agg.get("n_total", 0),
                "Avg Tokens": agg.get("avg_tokens", 0),
                "Avg Time (s)": agg.get("avg_time", 0),
                "Total Time (s)": agg.get("total_time", 0),
                "Prune Rate": agg.get("prune_rate", 0),
            })
        return pd.DataFrame(rows)

    def _print_results(self, results: Dict[str, Any], name: str):
        """Print final results table."""
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS")
        print("=" * 80)
        print(f"{'Method':<20} {'Accuracy':<12} {'Correct':<10} {'Avg Tokens':<12} {'Avg Time':<12} {'Prune Rate':<12}")
        print("-" * 80)

        for method, data in results.items():
            agg = data["aggregate"]
            acc = agg.get("accuracy", 0)
            correct = agg.get("n_correct", 0)
            total = agg.get("n_total", 0)
            tokens = agg.get("avg_tokens", 0)
            time_ = agg.get("avg_time", 0)
            prune = agg.get("prune_rate", 0)
            print(f"{method:<20} {acc:<12.1%} {correct}/{total:<7} {tokens:<12.0f} {time_:<12.2f}s {prune:<12.1%}")

        print("=" * 80)
        base = self.output_dir / name
        print(f"  Results:    {base}.json")
        print(f"  Summary:    {base}_summary.csv")
        print(f"  Comparison: {base}_comparison.png")
        print(f"  Heatmap:    {base}_heatmap.png")
        print(f"  Efficiency: {base}_efficiency.png")
        print("=" * 80)
