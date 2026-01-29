"""ETL pipeline: OpenMathReasoning → NuminaMath TIR format."""
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NuminaMath TIR chat template
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve problems step by step. "
    "Use Python code blocks to verify calculations. "
    "Write your final answer as: \\boxed{answer}"
)


def format_tir_chat(problem: str, solution: str) -> str:
    """Format problem+solution into NuminaMath TIR chat template."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{problem}\n"
        f"<|assistant|>\n{solution}"
    )


class OpenMathETL:
    """ETL: OpenMathReasoning → NuminaMath TIR Parquet."""

    def __init__(
        self,
        dataset_name: str = "nvidia/OpenMathReasoning",
        output_dir: str = "data/processed",
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "math_reasoning_tir.parquet"

    def extract(self) -> Dataset:
        """Download TIR split from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name} (tir split)...")
        return load_dataset(self.dataset_name, split='tir')

    def transform(self, ds: Dataset) -> pd.DataFrame:
        """Transform to NuminaMath TIR format."""
        logger.info("Transforming to TIR format...")

        def tir_mapper(example):
            problem = example["problem"]
            solution = example["generated_solution"]
            return {
                "problem": problem,
                "solution": solution,
                "expected_answer": example["expected_answer"],
                "difficulty": example.get("problem_source", "unknown"),
                "tir_text": format_tir_chat(problem, solution),
            }

        processed = ds.map(tir_mapper, remove_columns=ds.column_names)
        return processed.to_pandas()

    def load(self, df: pd.DataFrame) -> None:
        """Save to Parquet."""
        logger.info(f"Saving {len(df)} examples to {self.output_path}...")
        df.to_parquet(self.output_path, compression='snappy', index=False)
        logger.info("Done.")

    def run_pipeline(self) -> None:
        """Run full ETL."""
        raw = self.extract()
        df = self.transform(raw)
        self.load(df)
        logger.info(f"Output: {self.output_path} ({len(df)} rows)")


if __name__ == "__main__":
    OpenMathETL().run_pipeline()
