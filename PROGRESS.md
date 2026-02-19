# Competition-Ready Math Solver — Progress Tracker

---

## Phase 0: Foundation & Validation (Week 1)

### Step 1: Environment Setup
- [x] Install vLLM, SymPy, transformers, datasets libraries
- [x] Set up GPU environment (H100 80GB HBM3)
- [x] Download base model (`qingy2024/Qwen2.5-Math-14B-Instruct-Preview`)

### Step 2: Data Understanding
- [x] Load OpenMathReasoning dataset (TIR split, streaming)
- [x] Collect and deduplicate problems (500 unique collected)
- [x] Sample benchmark set (100 problems, stratified)
- [x] Inspect problem sources (aops_c6, aops_c4, aops_c7, aops_c5)
- [x] Inspect difficulty distribution (`pass_rate_72b_tir`)
- [ ] Manually inspect 50 problems across difficulty levels
- [ ] Categorize problems: algebra, geometry, number theory, combinatorics
- [ ] Identify which categories SymPy handles well vs. poorly

### Step 3: Reality Check Experiment
- [ ] Take 20 problems with step-by-step solutions
- [ ] Manually extract mathematical expressions from each step
- [ ] Try to verify each step with SymPy
- [ ] Measure SymPy verification success rate (target: >60%)
- [ ] Document failure modes (word problems, geometry, etc.)

---

## Phase 1: Solver Baseline

### Verification Engine
- [x] Answer extraction from `\boxed{}`
- [x] Fallback extraction ("the answer is X")
- [x] Answer normalization (numeric, symbolic via SymPy)
- [x] Answer matching (numeric tolerance + symbolic)
- [x] Python code block execution (subprocess, 10s timeout)
- [x] Solution scoring (answer presence + code pass/fail)

### Best-of-N Solver
- [x] Prompt builder with system prompt
- [x] Best-of-N generation (N=16, temperature=0.7, top_p=0.95)
- [x] Weighted majority vote (votes weighted by solution score)
- [x] Baseline benchmark on 100 problems
- [ ] Record and save baseline accuracy results

### Beam Search Solver
- [x] Step-by-step continuation generation
- [x] Per-step scoring (code exec + answer presence + length penalty)
- [x] Beam pruning (keep top-k, discard rest)
- [x] Majority vote across completed beams
- [ ] Run full benchmark (beam_width=4, max_steps=5)
- [ ] Compare beam search vs best-of-N results

---

## Phase 2: Improvement (TBD based on results)

- [ ] Determine next direction based on benchmark gap:
  - If coverage high, accuracy low → improve ranking/voting
  - If coverage low → fine-tune model
  - If beam search helps → explore MCTS
  - If beam search doesn't help → scale best-of-N + better verification

---

## Notes

- GPU: NVIDIA H100 80GB HBM3
- Model: `qingy2024/Qwen2.5-Math-14B-Instruct-Preview`
- Dataset: `nvidia/OpenMathReasoning` (TIR split)
- Problem sources in benchmark: mostly `aops_c6_high_school_olympiads` (54%), `aops_c4_high_school_math` (34%)
- Notebook: `notebooks/aimo-3-in-progress.ipynb`
