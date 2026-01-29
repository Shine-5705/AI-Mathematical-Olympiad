# Deductive-State MCTS for Mathematical Reasoning

Fine-tune NuminaMath (2024) on OpenMathReasoning (2026), then improve it with MCTS-guided rejection sampling.

## Setup

```bash
# Mac
pip install -r requirements-mac.txt

# Linux/Windows with NVIDIA GPU
pip install -r requirements-cuda.txt
```

## Full Pipeline

### Step 1: Download Data

```bash
python -m src.data.pipeline
```

### Step 2: SFT Training (baseline — what Numina did)

```bash
python train.py --mode sft --epochs 3

# Quick test
python train.py --mode sft --limit 100 --epochs 1
```

Output: `models/sft/lora_adapter`

### Step 3: MCTS-RL Training (our contribution)

```bash
python train.py --mode mcts-rl --sft-model models/sft/lora_adapter --epochs 2

# Quick test
python train.py --mode mcts-rl --limit 50 --epochs 1
```

This does:
1. Generates multiple solution paths per problem using MCTS
2. Verifies each with SymPy + code execution
3. Keeps only correct, verified solutions
4. Fine-tunes the SFT model on these verified solutions

Output: `models/mcts_rl/lora_adapter`

### Step 4: Compare Results

```bash
python run_experiment.py --limit 20
```

Compares three models head-to-head:
- `NuminaMath (base)` — original model, no training
- `SFT` — fine-tuned on OpenMathReasoning
- `MCTS-RL (ours)` — fine-tuned with verified MCTS solutions

## What Gets Produced

| File | Content |
|------|---------|
| `models/sft/loss_curve.png` | SFT training loss plot |
| `models/mcts_rl/loss_curve.png` | MCTS-RL training loss plot |
| `experiments/*_comparison.png` | Accuracy bar chart |
| `experiments/*_heatmap.png` | Per-problem correct/wrong grid |
| `experiments/*_efficiency.png` | Accuracy vs compute scatter |
| `experiments/*_summary.csv` | Results table for paper |

## Project Structure

```
├── train.py                    # --mode sft | --mode mcts-rl
├── run_experiment.py           # Compare base vs SFT vs MCTS-RL
├── src/
│   ├── data/pipeline.py        # OpenMathReasoning → TIR format
│   ├── training/
│   │   ├── sft.py              # Standard supervised fine-tuning
│   │   └── mcts_rl.py          # MCTS-guided rejection sampling + RL
│   ├── search/
│   │   ├── mcts.py             # MCTS algorithm
│   │   ├── generator.py        # LLM backends (transformers/mlx/vllm)
│   │   └── verifier.py         # Symbolic + code verification
│   └── evaluation/
│       ├── metrics.py          # Answer extraction & comparison
│       ├── runner.py           # Experiment orchestration
│       └── plots.py            # Publication-quality charts
├── models/
│   ├── sft/                    # SFT model output
│   └── mcts_rl/                # MCTS-RL model output
└── experiments/                # Experiment results
```
