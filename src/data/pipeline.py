"""ETL pipeline: OpenMathReasoning → NuminaMath TIR messages format.

Uses HuggingFace datasets in streaming mode so raw data is never saved to disk.
Rows are transformed on-the-fly and written directly to the output parquet file.
Only the final output file occupies disk space.
"""
import logging
import gc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 5_000

SCHEMA = pa.schema([
    ("problem", pa.string()),
    ("solution", pa.string()),
    ("expected_answer", pa.string()),
    ("difficulty", pa.string()),
    ("messages", pa.list_(pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ]))),
])


def build_tir_messages(problem: str, solution: str) -> list[dict]:
    """Build chat messages in Numina's format for TIR."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]


def _transform_row(example: dict) -> dict:
    problem = example["problem"]
    solution = example["generated_solution"]
    return {
        "problem": problem,
        "solution": solution,
        "expected_answer": example["expected_answer"],
        "difficulty": example.get("problem_source", "unknown"),
        "messages": build_tir_messages(problem, solution),
    }


class OpenMathETL:
    """Stream OpenMathReasoning TIR split from HuggingFace, transform, and write
    directly to a single parquet file. No intermediate files on disk.

    Resumable: if interrupted, re-running skips already-written rows by counting
    rows in the existing partial output file.
    """

    def __init__(
        self,
        dataset_name: str = "nvidia/OpenMathReasoning",
        output_dir: str = "data/processed",
        batch_size: int = BATCH_SIZE,
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "math_reasoning_tir.parquet"
        self.batch_size = batch_size

    def run_pipeline(self) -> None:
        """Stream dataset → transform → write parquet. One step, minimal disk usage."""

        # Check if already complete
        rows_done = 0
        if self.output_path.exists():
            try:
                rows_done = pq.read_metadata(self.output_path).num_rows
                logger.info(f"Existing output has {rows_done:,} rows")
            except Exception:
                logger.warning("Existing output is corrupt, starting fresh")
                self.output_path.unlink()
                rows_done = 0

        logger.info(f"Streaming {self.dataset_name} (tir split)...")
        ds = load_dataset(
            self.dataset_name,
            split="tir",
            streaming=True,
        )

        # If resuming, we append to a new file (can't append to parquet in-place)
        # so we write fresh, skipping rows_done rows from the stream.
        if rows_done > 0:
            # We need to start fresh but skip already-processed rows
            tmp_path = self.output_path.with_suffix(".tmp.parquet")
            logger.info(f"Resuming from row {rows_done:,} — writing to temp file first")
        else:
            tmp_path = self.output_path

        writer = None
        batch = []
        rows_written = 0
        rows_skipped = 0

        try:
            for example in tqdm(ds, desc="Streaming", unit="row"):
                # Skip already-processed rows when resuming
                if rows_skipped < rows_done:
                    rows_skipped += 1
                    continue

                batch.append(_transform_row(example))

                if len(batch) >= self.batch_size:
                    table = pa.Table.from_pylist(batch, schema=SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, SCHEMA, compression='snappy')
                    writer.write_table(table)
                    rows_written += len(batch)
                    batch.clear()
                    del table
                    gc.collect()
                    logger.info(f"Written {rows_written:,} rows")

            # Write remaining rows
            if batch:
                table = pa.Table.from_pylist(batch, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression='snappy')
                writer.write_table(table)
                rows_written += len(batch)
                batch.clear()
                del table
                gc.collect()

        finally:
            if writer is not None:
                writer.close()

        logger.info(f"Done: {rows_written:,} rows → {tmp_path}")

        # If we were resuming, replace original with new file
        if tmp_path != self.output_path:
            tmp_path.replace(self.output_path)
            logger.info(f"Replaced output file with resumed version")

        # Verify
        final_count = pq.read_metadata(self.output_path).num_rows
        logger.info(f"Verified: {final_count:,} rows in {self.output_path}")


if __name__ == "__main__":
    OpenMathETL().run_pipeline()
