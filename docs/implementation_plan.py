"""
==============================================================================
IMPLEMENTATION PLAN — Deductive-State MCTS for Mathematical Reasoning
==============================================================================

This file describes WHAT to implement, in WHAT order, and WHY.
No actual code — only structure, method descriptions, and TODOs.

==============================================================================
PHASE 1: DATA PREPARATION
==============================================================================

Goal: Prepare OpenMathReasoning TIR split for training and evaluation.

File: src/data/pipeline.py (already exists, minor updates needed)

    - Download nvidia/OpenMathReasoning TIR split (1.7M examples)
    - Keep columns: problem, generated_solution, expected_answer, problem_source
    - Add "category" column derived from problem_source:
        - "aops_c4_high_school_math"        -> "algebra"
        - "aops_c6_high_school_olympiads"   -> "olympiad"
        - "aops_c7_college_math"            -> "college"
        - "MATH_training_set"               -> "competition"
        - Map all problem_source values to broad categories for per-category analysis
    - Build messages column for SFT: [{system}, {user: problem}, {assistant: solution}]
    - Save as parquet

    - NEW: Create evaluation subset
        - Sample ~500-1000 problems stratified by category
        - Save separately as data/processed/eval_subset.parquet
        - This is what we run experiments on (full dataset too expensive for all ablations)

    - NEW: Create difficulty tiers using pass_rate_72b_tir column
        - "easy": pass_rate > 0.7
        - "medium": 0.3 < pass_rate <= 0.7
        - "hard": pass_rate <= 0.3
        - This allows difficulty-stratified analysis in the paper


==============================================================================
PHASE 2: BASE MODEL SETUP
==============================================================================

Goal: Support 3 base models for comparison experiments.

File: src/models/loader.py (NEW file)

    Three base models:
    1. AI-MO/NuminaMath-7B-TIR          — Numina's 2024 competition model
    2. nvidia/OpenMath-Nemotron-7B       — NVIDIA's model trained on OpenMathReasoning
    3. Qwen/Qwen2.5-Math-7B-Instruct    — Base math model (no fine-tuning on this dataset)

    For each model, the loader should handle:
    - Mac (MLX): Convert to MLX 4-bit format via mlx_lm.convert, load with mlx_lm.load
    - Linux (CUDA): Load with QLoRA 4-bit via BitsAndBytesConfig

    The loader returns a unified interface so the rest of the code doesn't care
    about the backend.

    WHY these 3 models:
    - NuminaMath: 2024 competition winner, trained on different data, shows transfer
    - OpenMath-Nemotron: Already trained on OpenMathReasoning, shows if MCTS-RL
      can improve an already-strong model
    - Qwen2.5-Math: Clean baseline, no competition fine-tuning, shows full pipeline effect

File: convert_model.py (already exists, update to loop over all 3 models)


==============================================================================
PHASE 3: SFT TRAINING
==============================================================================

Goal: Fine-tune each base model on OpenMathReasoning TIR data.

Files: src/training/sft.py (MLX), src/training/sft_cuda.py (CUDA)

    For each of the 3 base models:
    - Load model with LoRA adapters (r=16, alpha=32, target: q/k/v/o_proj)
    - Format data using tokenizer.apply_chat_template() on messages column
    - Train for 3 epochs, save adapter + loss curve
    - Output: models/sft/{model_name}/lora_adapter

    Key: OpenMath-Nemotron-7B is already trained on this data.
    SFT on it again is technically "continued training" — the paper should
    discuss whether this helps, hurts, or plateaus. This is interesting data
    for the paper regardless of the outcome.

    Metrics to log per training run:
    - Loss curve (per step)
    - Final training loss
    - Total training time
    - GPU/memory usage


==============================================================================
PHASE 4: MCTS-RL TRAINING (THE NOVEL CONTRIBUTION)
==============================================================================

Goal: Generate verified solutions using MCTS, then fine-tune on them.

Files: src/training/mcts_rl.py (MLX), src/training/mcts_rl_cuda.py (CUDA)

    Step 1 — Solution Generation with MCTS:
        For each problem in training set (or a subset):
        - Run MCTS search to generate candidate solutions
        - Each node expansion: LLM generates continuation
        - Verifier prunes invalid branches
        - Collect all solutions that reach \\boxed{} AND pass verification
        - Cache verified solutions to JSON (expensive to regenerate)

    Step 2 — Fine-tune on verified solutions:
        - Load SFT adapter (from Phase 3) as starting point
        - Apply new LoRA on top (r=8, lower rank for refinement)
        - Train on only the MCTS-verified solutions
        - This is rejection sampling + fine-tuning

    WHY this works (paper argument):
        - SFT trains on ALL solutions (including ones with errors the model can't detect)
        - MCTS-RL trains on only VERIFIED solutions
        - The verifier catches symbolic contradictions and code execution failures
        - The model learns to avoid these error patterns

    MCTS hyperparameters to make configurable (for ablation in Phase 6):
        - max_depth: [5, 10, 15, 20]
        - candidates_per_node: [2, 3, 5]
        - uct_constant: [0.5, 1.0, 1.41, 2.0]
        - iterations: [5, 10, 20, 50]


==============================================================================
PHASE 5: MCTS SEARCH ENGINE
==============================================================================

Goal: The core MCTS algorithm with configurable verification.

File: src/search/mcts.py (exists, needs updates for ablation support)

    MCTS Algorithm (4 phases per iteration):
        1. SELECT:  Walk tree using UCT until reaching a leaf
        2. EXPAND:  Generate N candidate continuations from LLM
        3. VERIFY:  Run verifier on each candidate, prune failures
        4. SIMULATE: Score the node (has \\boxed{}? has code? has output?)
        5. BACKPROP: Update visit counts and values up the tree

    After all iterations, return the most-visited leaf path.

    Key additions needed:
    - Tree statistics collection (depth distribution, branching factor, prune points)
    - Per-node verification logging (what was caught, why)
    - Support for different UCT constants
    - Configurable depth and branching

File: src/search/verifier.py (exists, needs ablation modes)

    Four verification modes (for ablation study):
    1. "none":     NoOpVerifier — accept everything
    2. "symbolic": SymbolicVerifier — SymPy equation checking
    3. "code":     CodeExecutionVerifier — execute python blocks
    4. "both":     CompositeVerifier — symbolic + code

    Each mode is a separate experiment row in the ablation table.

File: src/search/generator.py (exists, works)

    Three backends: MLX, Transformers, vLLM
    Already supports LoRA adapter loading.


==============================================================================
PHASE 6: EXPERIMENTS
==============================================================================

Goal: Generate all tables and figures for the paper.

File: src/evaluation/runner.py (exists, needs experiment matrix)
File: src/evaluation/metrics.py (exists, needs per-category support)
File: src/evaluation/plots.py (exists, needs new plot types)

--- EXPERIMENT 1: Base Model Comparison (Table 1 in paper) ---

    Matrix: 3 models x 3 stages = 9 cells

                        | NuminaMath-7B | OpenMath-Nemotron-7B | Qwen2.5-Math-7B |
    Base (no training)  |     xx.x%     |        xx.x%         |      xx.x%      |
    + SFT               |     xx.x%     |        xx.x%         |      xx.x%      |
    + MCTS-RL (ours)    |     xx.x%     |        xx.x%         |      xx.x%      |

    Run on eval subset (~500 problems).
    Report: accuracy, avg tokens, avg solve time.

--- EXPERIMENT 2: Verifier Ablation (Table 2 in paper) ---

    Pick the best model from Experiment 1. Run MCTS with 4 verifier modes:

                   | Accuracy | Prune Rate | Avg Depth | Avg Tokens |
    No verifier    |   xx.x%  |    0.0%    |   xx.x    |    xxxx    |
    Symbolic only  |   xx.x%  |   xx.x%    |   xx.x    |    xxxx    |
    Code only      |   xx.x%  |   xx.x%    |   xx.x    |    xxxx    |
    Both (ours)    |   xx.x%  |   xx.x%    |   xx.x    |    xxxx    |

    This shows the value of each verification component.

--- EXPERIMENT 3: MCTS Hyperparameter Ablation (Table 3 in paper) ---

    Vary one parameter at a time, keep others at default:

    Depth:      [5, 10, 15, 20]  — accuracy vs depth curve
    Branching:  [2, 3, 5]        — accuracy vs branching factor
    UCT:        [0.5, 1.0, 1.41, 2.0] — exploration vs exploitation
    Iterations: [5, 10, 20, 50]  — accuracy vs compute budget

    Plot as line charts. Key finding: diminishing returns after some point.

--- EXPERIMENT 4: Per-Category Analysis (Table 4 / Figure in paper) ---

    Break down best model's results by problem category:

                | Algebra | Olympiad | College | Competition |
    Base        |  xx.x%  |  xx.x%   |  xx.x%  |    xx.x%    |
    SFT         |  xx.x%  |  xx.x%   |  xx.x%  |    xx.x%    |
    MCTS-RL     |  xx.x%  |  xx.x%   |  xx.x%  |    xx.x%    |

    Also break down by difficulty tier (easy/medium/hard).
    Key finding: MCTS-RL should help most on hard problems where
    verification catches more errors.

--- EXPERIMENT 5: Tree Analysis (Figures for paper) ---

    Visualize MCTS behavior:
    - Average tree depth per problem category
    - Prune rate per problem category (where does verification help most?)
    - Example solution trees (2-3 cherry-picked examples showing
      good pruning decisions vs bad ones)
    - Distribution of node scores at each depth level
    - Token efficiency: tokens spent vs accuracy gained

    File: src/evaluation/tree_analysis.py (NEW)
        - Parse MCTS tree logs
        - Compute depth/branching/prune statistics
        - Generate tree visualization (graphviz or matplotlib)


==============================================================================
PHASE 7: RUNNING ORDER (CHRONOLOGY)
==============================================================================

Do these IN THIS ORDER. Each step depends on the previous.

    Step 1:  Update pipeline.py — add category mapping, difficulty tiers, eval subset
    Step 2:  Run pipeline: python -m src.data.pipeline
    Step 3:  Create src/models/loader.py — unified model loading for 3 models
    Step 4:  Update convert_model.py — convert all 3 models to MLX (if on Mac)
    Step 5:  Update train.py — accept --model flag, loop over models
    Step 6:  Run SFT for all 3 models (3 training runs)
    Step 7:  Update mcts.py — add configurable hyperparameters, tree logging
    Step 8:  Update verifier.py — ensure all 4 modes work cleanly
    Step 9:  Run MCTS-RL for all 3 models (3 training runs, using SFT adapters)
    Step 10: Update runner.py — experiment matrix, per-category breakdown
    Step 11: Run Experiment 1 (base model comparison)
    Step 12: Run Experiment 2 (verifier ablation)
    Step 13: Run Experiment 3 (MCTS hyperparameters)
    Step 14: Run Experiment 4 (per-category analysis)
    Step 15: Add tree_analysis.py, run Experiment 5
    Step 16: Update plots.py — generate all figures
    Step 17: Compile results into LaTeX tables (see docs/main.tex)


==============================================================================
FILES TO CREATE OR MODIFY
==============================================================================

    NEW FILES:
    - src/models/loader.py           — unified model loading
    - src/evaluation/tree_analysis.py — MCTS tree statistics and visualization
    - docs/main.tex                   — paper LaTeX source

    MODIFY:
    - src/data/pipeline.py           — add category, difficulty, eval subset
    - src/search/mcts.py             — configurable hyperparams, tree logging
    - src/search/verifier.py         — clean up 4 ablation modes
    - src/training/sft.py            — accept any model ID
    - src/training/sft_cuda.py       — accept any model ID
    - src/training/mcts_rl.py        — accept any model ID, load SFT adapter
    - src/training/mcts_rl_cuda.py   — accept any model ID, load SFT adapter
    - src/evaluation/runner.py       — experiment matrix, categories
    - src/evaluation/metrics.py      — per-category metrics
    - src/evaluation/plots.py        — ablation plots, tree plots
    - train.py                       — --model flag
    - run_experiment.py              — full experiment matrix
    - convert_model.py               — multi-model support
"""
