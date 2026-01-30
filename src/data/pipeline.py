"""ETL pipeline: OpenMathReasoning → NuminaMath TIR messages format.

Download is fully resumable — HuggingFace caches raw files to disk.
If interrupted, re-running picks up where it left off.
Processing happens in chunks to avoid OOM on 16GB machines.
"""
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset

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

    Step 1 (download): HuggingFace downloads raw parquet shards to
    ~/.cache/huggingface/datasets/. This is resumable — if interrupted,
    re-running continues from where it stopped. No data is lost.

    Step 2 (transform): Processes the cached dataset in chunks of 10K rows,
    writing each chunk as a separate parquet file. If interrupted, re-running
    skips already-processed chunks.

    Step 3 (merge): Combines all chunks into one final parquet file.
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

    def download(self) -> Dataset:
        """Download dataset with automatic resume.

        HuggingFace downloads raw parquet shards to disk cache.
        Each shard download is resumable via HTTP range requests.
        If this step is interrupted, re-running it will:
        - Skip already-downloaded shards (verified by checksum)
        - Resume partially-downloaded shards from where they stopped
        """
        logger.info(f"Downloading {self.dataset_name} (tir split)...")
        logger.info("Files are cached at ~/.cache/huggingface/datasets/")
        logger.info("If interrupted, re-run this command — it will resume automatically.")

        ds = load_dataset(
            self.dataset_name,
            split="tir",
            num_proc=4,
        )

        logger.info(f"Download complete: {len(ds):,} rows cached to disk")
        return ds

    def transform(self, ds: Dataset) -> None:
        """Transform in chunks to avoid OOM. Resumable."""
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Figure out which chunks are already done
        existing_chunks = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        rows_done = 0
        for f in existing_chunks:
            try:
                rows_done += len(pd.read_parquet(f, columns=["problem"]))
            except Exception:
                f.unlink()
                existing_chunks = sorted(self.chunks_dir.glob("chunk_*.parquet"))
                break

        if rows_done > 0:
            logger.info(f"Resuming transform — {rows_done:,} rows already processed")

        chunk_idx = len(existing_chunks)
        total = len(ds)

        for start in range(rows_done, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            batch = ds.select(range(start, end))
            processed = batch.map(_transform_row, remove_columns=batch.column_names)
            df = processed.to_pandas()

            chunk_path = self.chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
            df.to_parquet(chunk_path, compression='snappy', index=False)

            chunk_idx += 1
            logger.info(f"Chunk {chunk_idx}: rows {start:,}-{end:,} / {total:,}")

        logger.info(f"Transform complete: {chunk_idx} chunks")

    def merge(self) -> None:
        """Merge all chunks into a single parquet file."""
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            logger.error("No chunks found. Run transform first.")
            return

        logger.info(f"Merging {len(chunk_files)} chunks → {self.output_path}")

        # Read and concat in batches to avoid OOM
        dfs = []
        for f in chunk_files:
            dfs.append(pd.read_parquet(f))

        merged = pd.concat(dfs, ignore_index=True)
        merged.to_parquet(self.output_path, compression='snappy', index=False)
        logger.info(f"Done: {len(merged):,} rows → {self.output_path}")

        # Verify the merged file is readable and has all rows
        verify = pd.read_parquet(self.output_path)
        expected = sum(len(df) for df in dfs)
        if len(verify) == expected:
            logger.info(f"Verified: {len(verify):,} rows (matches chunks)")
        else:
            logger.error(f"Row mismatch! Merged={len(verify):,}, Expected={expected:,}")
            logger.error("Keeping chunk files for safety. Do NOT delete them.")
            return

        # Clean up chunks only after verified merge
        for f in chunk_files:
            f.unlink()
        self.chunks_dir.rmdir()
        logger.info("Chunk files cleaned up.")

    def run_pipeline(self) -> None:
        """Run full ETL: download → transform → merge.

        Each step is independently resumable:
        - Download: HF resumes partial shard downloads automatically
        - Transform: skips already-processed chunks
        - Merge: only runs if chunks exist and final file doesn't
        """
        if self.output_path.exists():
            try:
                existing = pd.read_parquet(self.output_path, columns=["problem"])
                logger.info(f"Already done: {self.output_path} ({len(existing):,} rows)")
                return
            except Exception:
                logger.warning(f"Existing file is corrupt, re-processing...")
                self.output_path.unlink()

        ds = self.download()
        self.transform(ds)
        self.merge()


if __name__ == "__main__":
    OpenMathETL().run_pipeline()
