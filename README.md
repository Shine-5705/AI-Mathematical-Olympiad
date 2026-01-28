# Deductive-State MCTS for Mathematical Reasoning

Fine-tune NuminaMath (2024) on OpenMathReasoning (2026) + MCTS search with symbolic verification.

## Setup

```bash
# Mac
pip install -r requirements-mac.txt

# Linux/Windows with NVIDIA GPU
pip install -r requirements-cuda.txt
```

## Training

### 1. Download Dataset

```bash
python -m src.data.pipeline
```

### 2. Fine-tune NuminaMath on OpenMathReasoning

```bash
# Full training
python train.py --data data/processed/math_reasoning_tir.parquet --epochs 3

# Quick test (100 examples)
python train.py --data data/processed/math_reasoning_tir.parquet --limit 100 --epochs 1
```

Model saves to `models/numina-openmath/lora_adapter`.

### 3. Run Experiments

```bash
# With your trained model
python run_experiment.py --model models/numina-openmath/lora_adapter --limit 10

# Compare methods
python run_experiment.py --methods direct cot mcts --limit 20
```

## What This Does

```
NuminaMath-7B-TIR (2024)     →  Strong math reasoning base
        +
OpenMathReasoning (2026)    →  Latest competition-style problems
        +
MCTS + Symbolic Verification →  Search algorithm that prunes bad reasoning
        =
Your trained model that outperforms either alone
```

## Project Structure

```
├── train.py              # Training script
├── run_experiment.py     # Evaluation script
├── main.py               # Quick test
├── models/               # Trained models
├── src/
│   ├── data/pipeline.py  # Dataset ETL
│   ├── training/         # Fine-tuning code
│   ├── evaluation/       # Metrics & runner
│   └── search/           # MCTS, generators, verifiers
└── experiments/          # Results
```

## Commands

| Task | Command |
|------|---------|
| Download data | `python -m src.data.pipeline` |
| Train model | `python train.py --epochs 3` |
| Run experiments | `python run_experiment.py --limit 10` |
| Quick test | `python main.py` |
