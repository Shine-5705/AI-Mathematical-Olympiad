# AIMO3 7B System Status And Roadmap

## Current Objective
Build the strongest possible `Qwen2.5-Math-7B-Instruct` AIMO3 system on Kaggle using smarter inference before attempting full RL training.

## What We Have Done

### 1. Baseline Solver
- Loaded `Qwen2.5-Math-7B-Instruct` as the main workhorse model.
- Kept the `72B` path available for larger GPUs, but tuned defaults for `7B`.
- Added GPU-aware runtime settings for safer Kaggle execution.

### 2. TIR Pipeline
- Implemented Tool-Integrated Reasoning (TIR).
- The model can generate Python code during reasoning.
- The notebook executes generated Python code.
- Code output is injected back into the prompt so the model can continue solving.

### 3. Answer Extraction
- Implemented integer-only answer extraction for AIMO3.
- Supports boxed answers like `\boxed{42}`.
- Supports fallback extraction from final integer-style conclusions.
- Added smoke tests for extraction behavior.

### 4. Candidate Scoring
- Added lightweight verifier-style scoring.
- Scores candidates using:
  - whether a final answer was extracted
  - whether the answer is boxed
  - whether code execution succeeded or failed
  - whether the solution looks complete
- Added weighted answer selection instead of relying only on plain majority vote.

### 5. Two Solver Modes
- `solve_majority(...)`
  - multi-sample TIR generation
  - final answer chosen by scored voting
- `solve_beam(...)`
  - beam-search version for `7B`
  - expands multiple reasoning continuations
  - keeps the best intermediate states each round
- `solve(...)`
  - dispatches to beam search or majority mode based on config

### 6. Benchmarking
- Added reference-set benchmarking using `reference.csv`.
- Added side-by-side comparison between:
  - `beam`
  - `majority`
- Added result exports:
  - `results/reference_benchmark.json`
  - `results/reference_benchmark.csv`
  - `results/reference_benchmark_summary.json`
  - `results/reference_benchmark_comparison.json`

## What Is Not Done Yet
- true symbolic verifier
- PRM / reward model
- RL fine-tuning
- MCTS
- UCB / tree policy scoring
- bootstrapping loops
- multi-strategy ensemble
- full ablation study beyond beam vs majority

## Current Project Status
- Baseline system: done
- TIR execution loop: done
- Answer extraction: done
- Lightweight candidate scoring: done
- Beam search: done
- Beam vs majority benchmark pipeline: done
- Verifier training: not done
- PRM / RL: not done
- MCTS: not done

## Next Checkpoints

### Checkpoint 1. Run And Validate
- Run the notebook top to bottom in Kaggle.
- Confirm the smoke test still returns `40` for `7^2 - 3^2`.
- Run the reference benchmark.
- Record runtime and accuracy for both `beam` and `majority`.

Success condition:
- notebook runs cleanly
- result files are generated
- benchmark numbers are available for both modes

### Checkpoint 2. Pick The Better Inference Mode
- Compare `beam` vs `majority` using the saved benchmark files.
- Decide which mode is the default submission strategy.
- Note whether beam improves accuracy enough to justify extra runtime.

Success condition:
- one inference mode is clearly chosen as primary

### Checkpoint 3. Failure Analysis
- Review incorrect predictions from the reference benchmark.
- Categorize failures:
  - bad final extraction
  - bad reasoning path
  - code execution failure
  - correct reasoning but wrong final integer
  - search not exploring enough
- Identify the most common failure type.

Success condition:
- top 2 to 3 failure modes are documented

### Checkpoint 4. Improve The Verifier Layer
- Replace heuristic-only scoring with a stronger verifier-style module.
- Add signals such as:
  - answer consistency
- code-result consistency
  - stronger reasoning-completion checks
  - invalid-path penalties
- Re-run the benchmark.

Success condition:
- verifier improves ranking quality or beam pruning quality

### Checkpoint 5. Beam Search Tuning
- Tune:
  - `BEAM_WIDTH`
  - `BEAM_BRANCH_FACTOR`
  - `MAX_ROUNDS`
  - temperature and `top_p`
- Run small controlled comparisons.
- Keep the best accuracy/runtime tradeoff.

Success condition:
- tuned beam setup outperforms untuned beam or is proven not worth it

### Checkpoint 6. Submission Strategy
- Choose final Kaggle inference configuration.
- Lock:
  - solver mode
  - number of samples
  - beam settings
  - max rounds
  - runtime-safe parameters
- Prepare final submission notebook.

Success condition:
- one stable submission configuration is ready

## Optional Later Roadmap
Only continue here if the inference/search path shows clear benefit.

### Optional A. True Symbolic Verifier
- Build a stronger symbolic verification module.
- Use it to score intermediate reasoning states.

### Optional B. Small-Scale Training Data
- Create a compact curated dataset.
- Use it for LoRA, verifier training, or PRM experiments.
- Do not start with a huge full dataset inside Kaggle.

### Optional C. PRM / RL Layer
- Train a process reward model or verifier-guided scorer.
- Use it to rank partial trajectories better than heuristics.

### Optional D. MCTS
- Add tree search only after verifier quality is good enough.
- Otherwise tree search will explore bad branches expensively.

## Immediate Next Action
Run the benchmark and inspect:
- `results/reference_benchmark.csv`
- `results/reference_benchmark_summary.json`
- `results/reference_benchmark_comparison.json`

That result should determine the next engineering decision.
