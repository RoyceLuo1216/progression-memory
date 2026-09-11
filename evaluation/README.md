# Evaluation and Benchmark Ledger

This directory is the source of truth for progression-memory evaluation results. Results are split into **reproduced measurements** and **published external reference numbers**. A published number is never treated as an apples-to-apples comparison until the same dataset, reader model, retrieval budget, and evaluator are reproduced here.

## Current reproduced results

### 1. Synthetic irregular-transition benchmark

The v1 synthetic harness contains 45 episodes with true phase boundaries at episodes 12 and 29. It tests whether a temporal memory policy discovers state-dependent boundaries rather than imposing calendar windows.

| System | Boundary precision | Boundary recall | Boundary F1 | Segments | Episodes / segment |
|---|---:|---:|---:|---:|---:|
| Fixed 14-day windows | 0.1667 | 0.5000 | 0.2500 | 7 | 6.43 |
| Adaptive linear controller | **1.0000** | **1.0000** | **1.0000** | **3** | **15.00** |

Raw result: `results/synthetic_v1.json`. This is mechanism validation, not end-to-end LongMemEval/LoCoMo QA.

### 2. Equal-top-k real-data retrieval

Full public LoCoMo and LongMemEval-S are downloaded during GitHub Actions. All systems use the same dependency-free BM25 retriever and evidence-aware metrics. LongMemEval excludes 30 abstention questions, leaving 470 evidence-bearing questions; LoCoMo excludes four questions without annotated evidence.

Progression systems use deterministic hashed lexical vectors in this benchmark. Every progression/segment retains provenance links to source evidence.

#### LongMemEval-S — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR@8 | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 97.87% | 88.72% | 94.11% | 91.91% | 0.9149 | **101,661** | 47.73 |
| Fixed progression | 98.30% | 92.13% | 95.83% | 94.83% | **0.9263** | 144,636 | 43.05 |
| Adaptive progression v1 | 98.51% | **94.04%** | 96.77% | 96.18% | 0.9257 | 155,653 | 42.52 |
| Adaptive progression v2 | **98.72%** | 93.40% | 96.76% | 95.96% | 0.9223 | **149,694** | 41.89 |
| **Progression only** | 98.51% | **94.89%** | **97.16%** | **96.63%** | 0.9229 | 173,672 | **40.61** |

Adaptive v2 essentially preserves v1's LongMemEval macro recall (96.76% vs 96.77%) while reducing the average top-8 payload by about 6.0k characters and slightly reducing the number of retrieval units. The all-evidence score is 0.64 pp lower than v1, so this is not an unconditional top-k improvement.

#### LongMemEval-S — hypothesis-relevant categories at top-8

| Category | Append-only | Fixed progression | Adaptive v1 | Adaptive v2 | Progression only |
|---|---:|---:|---:|---:|---:|
| Temporal reasoning | 91.56% | 93.22% | 95.18% | **95.55%** | **95.58%** |
| Multi-session | 89.68% | 93.82% | **94.99%** | **94.99%** | **96.09%** |
| Knowledge update | 99.31% | 99.31% | **100.00%** | 99.31% | **100.00%** |

#### LoCoMo — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR@8 | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 60.60% | 52.67% | 56.18% | 45.72% | 0.4047 | **1,172** | 588.2 |
| Fixed progression | 66.55% | 57.62% | 61.58% | 50.94% | 0.4529 | 3,546 | 320.5 |
| Adaptive progression v1 | 58.68% | 50.91% | 54.34% | 44.01% | 0.4003 | 1,323 | 746.3 |
| **Adaptive progression v2** | 65.34% | 56.10% | **60.20%** | **49.56%** | **0.4501** | 2,945 | **339.7** |
| Progression only | **70.13%** | **60.24%** | **64.73%** | **55.17%** | **0.4572** | 9,379 | 163.6 |

Adaptive v2 fixes the principal structural failure of v1 on LoCoMo. Average retrieval units fall from **746.3 to 339.7 (-54.5%)**, average segment size rises from 1.12 to 2.47 events, and the single-event-segment fraction falls from **93.3% to 27.1%**. At the same time, top-8 macro evidence recall rises by **+5.86 pp** (54.34% -> 60.20%) and MRR@8 rises from 0.4003 to 0.4501.

This validates the confidence-aware damping/minimum-evidence/hysteresis idea as a segmentation stabilization mechanism. It does **not** yet establish context efficiency because the stabilized segments are larger.

**MRR correction:** commit `947381757156ec5a9251485ea8aebfeb99ce6af6` corrected MRR so `MRR@k` only credits evidence within the requested cutoff. Recall metrics were unaffected.

### 3. Equal-character-budget retrieval

The budget benchmark applies an identical **strict ranked-prefix whole-unit character budget** to every system. A retrieval unit is included only if its full text fits in the remaining budget. It reports paired 2,000-sample bootstrap confidence intervals for per-question macro evidence recall and exact paired all-evidence tests against append-only.

Initial v1 provenance: run `34608897574`, source commit `947381757156ec5a9251485ea8aebfeb99ce6af6`.

Adaptive-v2 provenance: run `34618743057`, source commit `7b1a17695c7f10693371bf1de1ba1a5eb5043d0b`. Compact result: `results/adaptive_v2_summary.json`.

#### LoCoMo — matched budgets

| Budget | Append | Adaptive v1 | Adaptive v2 | V2 Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 1,200 chars | **55.20%** | 52.52% | 46.21% | **-8.99 pp** [-10.55, -7.45] |
| 2,400 chars | **63.11%** | 60.78% | 56.74% | **-6.37 pp** [-7.73, -5.04] |
| 4,800 chars | **68.65%** | 66.42% | 65.18% | **-3.47 pp** [-4.64, -2.31] |

V2 substantially improves raw LoCoMo structure and ranking, but performs worse than v1 at strict whole-unit budgets. This is a useful failure: once v2 stops over-splitting, its larger segments become harder to admit atomically. The result isolates the next bottleneck as **retrieval-unit expansion policy rather than segmentation instability**.

#### LongMemEval-S — matched budgets

| Budget | Append | Adaptive v1 | Adaptive v2 | V2 Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 100k chars | **93.59%** | 89.15% | 92.74% | **-0.85 pp** [-2.12, +0.33] |
| 150k chars | **95.77%** | 94.15% | 95.58% | **-0.19 pp** [-1.12, +0.63] |
| 200k chars | 96.69% | 97.02% | **97.21%** | **+0.52 pp** [-0.13, +1.21] |

On LongMemEval, v2 nearly closes the matched-budget gap at 100k/150k and has a +0.52 pp point estimate over append-only at 200k. However, every v2 confidence interval still crosses zero; the 200k all-evidence exact paired test is also non-significant (`p=0.344`). We therefore do **not** claim a statistically established win.

### 4. What adaptive v2 establishes

The v2 experiment sharpens the architecture story:

1. **The v1 LoCoMo failure was substantially a segmentation-stability problem.** Confidence-aware damping and hysteresis cut retrieval units by 54.5% while improving raw recall by 5.86 pp.
2. **Stabilizing segments alone cannot solve context efficiency.** Larger, better-formed segments are penalized by strict whole-unit admission, especially under LoCoMo's small budgets.
3. **LongMemEval becomes much more competitive under equal budgets.** V2 reduces the 100k deficit from -4.44 pp to -0.85 pp and the 150k deficit from -1.61 pp to -0.19 pp, while reaching a non-significant +0.52 pp at 200k.
4. **The next controlled experiment must decouple structural indexing from reader payload.** We should rank compact progression/segment representations, then expand and rerank only source episodes that fit the final reader budget.
5. **Only after that flat two-stage baseline should hierarchy/DAG retrieval be added.** This lets the DAG be evaluated as an incremental routing advantage rather than conflating hierarchy with selective expansion.

The complete CI artifacts retain all top-k and budget JSON outputs.

## Published reference targets

These are external paper results used only as targets until reproduced under a common reader/evaluator setup.

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Published end-to-end QA; reports 52.20% lower recalled memory length on LoCoMo. |
| Temporal Semantic Memory | 74.8% reported setup | 76.69% reported setup | Published end-to-end QA under its reported setup. |
| HeLa-Mem | 65.4% reported setup | - | Published end-to-end QA under its reported setup. |

These percentages are **end-to-end QA scores, not retrieval recall**, and must not be numerically compared to the retrieval tables above.

Primary references:
- TiMem: https://aclanthology.org/2026.findings-acl.1091/
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo

## Evaluation roadmap

1. **DONE** — dataset adapters + evidence-aware retrieval metrics.
2. **DONE** — append-only lexical baseline.
3. **DONE** — progression-only routing/grouping ablation.
4. **DONE** — fixed-window progression retrieval.
5. **DONE** — adaptive progression v1.
6. **DONE** — strict equal-character-budget retrieval + paired uncertainty/significance.
7. **DONE** — adaptive v2 confidence-aware damping, minimum evidence, and hysteresis.
8. **NEXT** — two-stage compact progression/segment index followed by budgeted source-evidence expansion.
9. **NEXT** — DAG/hierarchical structured retrieval on top of the two-stage baseline, with an explicit flat-vs-DAG ablation.
10. **NEXT** — fixed reader + official LongMemEval/LoCoMo end-to-end QA evaluation under one model and context budget.
11. **LATER** — learned temporal policy, consolidation/knowledge objects, reinforcement/decay, and hot/warm/cold memory.

For every stage we record evidence coverage, recalled context size, memory/index size, paired uncertainty, and answer accuracy/F1 once a fixed reader is available.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
