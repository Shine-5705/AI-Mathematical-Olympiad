# Deductive-State MCTS for Mathematical Reasoning

MCTS-based math solver that uses symbolic verification to prune invalid reasoning paths. Unlike standard Chain-of-Thought, this system explores multiple solution branches and kills mathematically inconsistent ones early.

## Setup

```bash
# Clone and enter directory
cd AI-Mathematical-Olympiad

# Mac (Apple Silicon)
pip install -r requirements-mac.txt

# Linux/Windows (NVIDIA GPU)
pip install -r requirements-cuda.txt

# CPU only
pip install -r requirements.txt
```

## Quick Start

### 1. Download Dataset

```bash
python -m src.data.pipeline
```

This downloads NVIDIA OpenMathReasoning dataset and saves to `data/processed/math_reasoning_tir.parquet`.

### 2. Run Single Problem

```bash
python main.py
```

### 3. Run Experiments

```bash
# Basic run (10 problems, auto-detect backend)
python run_experiment.py --limit 10

# Compare all methods
python run_experiment.py --methods direct cot mcts mcts_no_verify --limit 20

# Use specific backend
python run_experiment.py --backend vllm --limit 10      # NVIDIA GPU
python run_experiment.py --backend mlx --limit 10       # Mac (quantized)
python run_experiment.py --backend transformers --limit 10  # Any platform

# Use specific model
python run_experiment.py --backend transformers --model AI-MO/NuminaMath-7B-TIR --limit 10
```

Results save to `experiments/` as JSON.

## Project Structure

```
├── main.py                 # Single problem test
├── run_experiment.py       # Full experiment runner
├── src/
│   ├── data/
│   │   └── pipeline.py     # Dataset ETL
│   ├── evaluation/
│   │   ├── metrics.py      # Answer extraction & comparison
│   │   └── runner.py       # Experiment orchestration
│   └── search/
│       ├── baselines.py    # Direct, CoT, Self-Consistency
│       ├── generator.py    # LLM backends (transformers/mlx/vllm)
│       ├── mcts.py         # MCTS algorithm
│       └── verifier.py     # Symbolic & code verification
├── data/processed/         # Processed datasets
└── experiments/            # Experiment results
```

## Methods

| Method | Description |
|--------|-------------|
| `direct` | Single-shot prompting |
| `cot` | Chain-of-thought (linear) |
| `self_consistency` | Majority voting over samples |
| `mcts` | MCTS + symbolic verification |
| `mcts_no_verify` | MCTS without verification (ablation) |

## Backends

| Backend | Platform | Default Model | RAM |
|---------|----------|---------------|-----|
| `transformers` | Any | NuminaMath-7B-TIR | ~16GB |
| `mlx` | Mac M-series | Qwen2.5-Math-7B-4bit | ~8GB |
| `vllm` | NVIDIA GPU | NuminaMath-7B-TIR | ~16GB |

## Metrics Collected

- **Accuracy**: % correct answers
- **Tokens**: Total tokens generated
- **Time**: Solve time per problem
- **Prune Rate**: % branches killed by verifier (MCTS only)

## Python API

```python
from src.search import AIMO_MCTS

engine = AIMO_MCTS(backend="transformers", verify_mode="symbolic")
solution = engine.search("Find all integers n such that n^2 + 1 divides n^3 + 1", iterations=10)

print(solution)
print(engine.get_metrics())
```
