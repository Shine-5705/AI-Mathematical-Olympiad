# Deductive-State MCTS for Mathematical Reasoning

Fine-tune NuminaMath on OpenMathReasoning, then improve it with MCTS-guided rejection sampling.

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

### Step 2 (Mac only): Convert NuminaMath to MLX

```bash
# Convert + quantize to 4-bit (fits in 16GB RAM)
python convert_model.py

# Or specify a different model
python convert_model.py --model Qwen/Qwen2.5-Math-7B-Instruct --bits 4
```

Output: `models/mlx/NuminaMath-7B-TIR-4bit`

On Linux this step is not needed — QLoRA handles quantization at load time.

### Step 3: SFT Training (baseline — what Numina did)

```bash
python train.py --mode sft --epochs 3

# Quick test
python train.py --mode sft --limit 100 --epochs 1
```

Auto-detects platform:
- Mac → uses converted MLX model from Step 2 (or falls back to Qwen2.5-Math)
- Linux → uses NuminaMath-7B-TIR with QLoRA

Output: `models/sft/lora_adapter`

### Step 4: MCTS-RL Training (our contribution)

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

### Step 5: Compare Results

```bash
python run_experiment.py --limit 20
```

Compares three models head-to-head:
- `NuminaMath (base)` — original model, no training
- `SFT` — fine-tuned on OpenMathReasoning
- `MCTS-RL (ours)` — fine-tuned with verified MCTS solutions

## Platform Summary

| Platform | Model | Quantization | RAM/VRAM |
|----------|-------|-------------|----------|
| Mac M-series | NuminaMath-7B (converted) | MLX 4-bit | ~5GB |
| Mac M-series | Qwen2.5-Math-7B (fallback) | MLX 4-bit | ~5GB |
| Linux NVIDIA | NuminaMath-7B-TIR | QLoRA 4-bit | ~6GB |

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
├── convert_model.py            # HF → MLX conversion (Mac)
├── train.py                    # --mode sft | --mode mcts-rl
├── run_experiment.py           # Compare base vs SFT vs MCTS-RL
├── src/
│   ├── data/pipeline.py        # OpenMathReasoning → TIR format
│   ├── training/
│   │   ├── sft.py              # SFT with MLX (Mac)
│   │   ├── sft_cuda.py         # SFT with QLoRA (Linux)
│   │   ├── mcts_rl.py          # MCTS-RL with MLX (Mac)
│   │   └── mcts_rl_cuda.py     # MCTS-RL with QLoRA (Linux)
│   ├── search/
│   │   ├── mcts.py             # MCTS algorithm
│   │   ├── generator.py        # LLM backends (transformers/mlx/vllm)
│   │   └── verifier.py         # Symbolic + code verification
│   └── evaluation/
│       ├── metrics.py          # Answer extraction & comparison
│       ├── runner.py           # Experiment orchestration
│       └── plots.py            # Publication-quality charts
├── models/
│   ├── mlx/                    # Converted MLX models
│   ├── sft/                    # SFT model output
│   └── mcts_rl/                # MCTS-RL model output
└── experiments/                # Experiment results
```
