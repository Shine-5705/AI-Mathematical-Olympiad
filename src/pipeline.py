import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenMathETL:
    """
    ETL Pipeline for NVIDIA OpenMathReasoning to SC-TIR Parquet format.
    """
    def __init__(self, dataset_name: str = "nvidia/OpenMathReasoning", output_dir: str = "data/processed"):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "math_reasoning_tir.parquet"

    def extract(self) -> Dataset:
        """ Step 1: Download/Extract from HuggingFace """
        logger.info(f"Extracting {self.dataset_name} from HiggingFace...")
        # we target the 'tir' (Tool-Integrated Reasoning) split specifically
        try:
            return load_dataset(self.dataset_name, split='tir')
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def transform(self, ds: Dataset) -> pd.DataFrame:
        """ Step 2: Transform to SC-TIR Format & Schema Clean-up """
        logger.info("Transforming dataset to SC-TIR schema...")

        def sc_tir_mapper(example):
            return {
                "instruction": example["problem"],
                "response": exmaple["generated_solution"],
                "target": example["expected_answer"],
                "difficulty": example.get("problem_source", "unknown")
            }

        # Apply mapping and convert to Pandas for Parquet optimization
        processed_ds = ds.map(sc_tir_mapper, remove_columns=ds.column_names)
        return processed_ds.to_pandas()

    def load(self, df: pd.DataFrame):
        """ Step 3: Load/Save to Parquet """
        logger.info(f"Loading data info {self.output_path}...")
        df.to_parquet(self.output_path, compression='snappy', index=False)
        logger.info("ETL Process Complete")

    def run_pipeline(self):
        """ Orchestrates the full ETL process """
        raw_data = self.extract()
        transformed_df = self.transform(raw_data)
        self.load(transformed_df)

if __name__ == "__main__":
    # Example of running the pipeline directly
    etl = OpenMathETL()
    etl.run_pipeline()
