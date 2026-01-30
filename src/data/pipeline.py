"""ETL pipeline: OpenMathReasoning → NuminaMath TIR messages format.

Supports resumable downloads via streaming + chunked parquet writes.
If interrupted, re-running picks up from the last saved chunk.
"""
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 10_000


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
    """ETL: OpenMathReasoning → TIR messages Parquet.

    Downloads via streaming in chunks so it can resume after interruption.
    """

    def __init__(
        self,
        dataset_name: str = "nvidia/OpenMathReasoning",
        output_dir: str = "data/processed",
        chunk_size: int = CHUNK_SIZE,
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "math_reasoning_tir.parquet"
        self.chunks_dir = self.output_dir / "chunks"
        self.chunk_size = chunk_size

    def _count_existing_rows(self) -> int:
        """Count rows already downloaded in chunk files."""
        if not self.chunks_dir.exists():
            return 0
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        total = 0
        for f in chunk_files:
            try:
                total += len(pd.read_parquet(f))
            except Exception:
                f.unlink()
        return total

    def _get_next_chunk_idx(self) -> int:
        """Get the next chunk index to write."""
        if not self.chunks_dir.exists():
            return 0
        existing = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        if not existing:
            return 0
        last = existing[-1].stem  # "chunk_0042"
        return int(last.split("_")[1]) + 1

    def extract_and_transform(self) -> None:
        """Stream dataset and save in resumable chunks."""
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        skip_rows = self._count_existing_rows()
        chunk_idx = self._get_next_chunk_idx()

        if skip_rows > 0:
            logger.info(f"Resuming download — {skip_rows:,} rows already saved ({chunk_idx} chunks)")

        logger.info(f"Streaming {self.dataset_name} (tir split)...")
        ds = load_dataset(self.dataset_name, split="tir", streaming=True)

        buffer = []
        total_saved = skip_rows

        for i, example in enumerate(ds):
            # Skip rows we already have
            if i < skip_rows:
                if i % 100_000 == 0 and i > 0:
                    logger.info(f"Skipping... {i:,}/{skip_rows:,}")
                continue

            buffer.append(_transform_row(example))

            if len(buffer) >= self.chunk_size:
                chunk_path = self.chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
                pd.DataFrame(buffer).to_parquet(chunk_path, compression='snappy', index=False)
                total_saved += len(buffer)
                logger.info(f"Saved chunk {chunk_idx} — {total_saved:,} rows total")
                buffer.clear()
                chunk_idx += 1

        # Save remaining
        if buffer:
            chunk_path = self.chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
            pd.DataFrame(buffer).to_parquet(chunk_path, compression='snappy', index=False)
            total_saved += len(buffer)
            logger.info(f"Saved final chunk {chunk_idx} — {total_saved:,} rows total")

        logger.info(f"Download complete: {total_saved:,} rows in {chunk_idx + 1} chunks")

    def merge_chunks(self) -> None:
        """Merge all chunks into a single parquet file."""
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            logger.error("No chunks found. Run extract_and_transform first.")
            return

        logger.info(f"Merging {len(chunk_files)} chunks into {self.output_path}...")
        dfs = [pd.read_parquet(f) for f in chunk_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_parquet(self.output_path, compression='snappy', index=False)
        logger.info(f"Merged: {len(merged):,} rows → {self.output_path}")

    def run_pipeline(self) -> None:
        """Run full ETL with resume support."""
        # Skip if final output already exists
        if self.output_path.exists():
            existing = pd.read_parquet(self.output_path)
            logger.info(f"Output already exists: {self.output_path} ({len(existing):,} rows)")
            logger.info("Delete it to re-download, or use extract_and_transform() to add more chunks.")
            return

        self.extract_and_transform()
        self.merge_chunks()
        logger.info(f"Pipeline complete: {self.output_path}")


if __name__ == "__main__":
    OpenMathETL().run_pipeline()
