# Evaluation and Benchmark Ledger

This directory is the source of truth for progression-memory evaluation results. Results are split into **reproduced measurements** and **published external reference numbers**. A published number is never treated as an apples-to-apples comparison until the same dataset, reader model, retrieval budget, and evaluator are reproduced here.

## Current reproduced results

### 1. Synthetic irregular-transition benchmark

The v1 synthetic harness contains 45 episodes with true phase boundaries at episodes 12 and 29. The benchmark tests whether a temporal memory policy discovers state-dependent boundaries rather than imposing calendar windows.

| System | Boundary precision | Boundary recall | Boundary F1 | Segments | Episodes / segment |
|---|---:|---:|---:|---:|---:|
| Fixed 14-day windows | 0.1667 | 0.5000 | 0.2500 | 7 | 6.43 |
| Adaptive linear controller | **1.0000** | **1.0000** | **1.0000** | **3** | **15.00** |

Raw result: `results/synthetic_v1.json`.

This establishes mechanism validity only; it is not an end-to-end LongMemEval/LoCoMo result.

### 2. Equal-top-k real-data retrieval

Full public LoCoMo and LongMemEval-S are downloaded during GitHub Actions. All systems use the same dependency-free BM25 retriever and evidence-aware metrics. LongMemEval excludes 30 abstention questions, leaving 470 evidence-bearing questions; LoCoMo excludes four questions without annotated evidence.

Progression systems use deterministic hashed lexical vectors in this benchmark. Every progression/segment retains provenance links to source evidence.

#### LongMemEval-S — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR@8 | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 97.87% | 88.72% | 94.11% | 91.91% | 0.9149 | **101,661** | 47.73 |
| Fixed progression | 98.30% | 92.13% | 95.83% | 94.83% | **0.9263** | 144,636 | 43.05 |
| Adaptive progression v1 | 98.51% | 94.04% | 96.77% | 96.18% | 0.9257 | 155,653 | 42.52 |
| **Progression only** | **98.51%** | **94.89%** | **97.16%** | **96.63%** | 0.9229 | 173,672 | **40.61** |

Progression-only is the raw top-k recall leader, but its retrieved payload is much larger. Adaptive v1 recovers almost as much evidence with less text than progression-only, but still uses substantially more context than append-only.

#### LongMemEval-S — hypothesis-relevant categories at top-8

| Category | Append-only | Fixed progression | Adaptive v1 | Progression only |
|---|---:|---:|---:|---:|
| Temporal reasoning | 91.56% | 93.22% | 95.18% | **95.58%** |
| Multi-session | 89.68% | 93.82% | 94.99% | **96.09%** |
| Knowledge update | 99.31% | 99.31% | **100.00%** | **100.00%** |

#### LoCoMo — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR@8 | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 60.60% | 52.67% | 56.18% | 45.72% | 0.4047 | **1,172** | 588.2 |
| Fixed progression | 66.55% | 57.62% | 61.58% | 50.94% | 0.4529 | 3,546 | 320.5 |
| Adaptive progression v1 | 58.68% | 50.91% | 54.34% | 44.01% | 0.4003 | 1,323 | 746.3 |
| **Progression only** | **70.13%** | **60.24%** | **64.73%** | **55.17%** | **0.4572** | 9,379 | **163.6** |

Progression-only improves LoCoMo raw top-k recall substantially and reduces searchable memory units, but whole-progression expansion is expensive. Adaptive v1 over-splits short conversational turns: 746.3 retrieval units from 588.2 raw turns on average.

**MRR correction:** commit `947381757156ec5a9251485ea8aebfeb99ce6af6` corrected MRR so `MRR@k` only credits evidence within the requested cutoff. Earlier README commits used the first relevant result from the full ranking. Recall metrics were unaffected.

### 3. Equal-character-budget retrieval

Equal-top-k retrieval is not a fair efficiency comparison because progression units carry different amounts of text. The budget benchmark therefore applies an identical **strict ranked-prefix whole-unit character budget** to every system: a retrieval unit is included only if the full unit fits in the remaining budget. It also reports paired 2,000-sample bootstrap confidence intervals for per-question macro evidence recall and exact paired all-evidence tests against append-only.

GitHub Actions provenance: run `34608897574`, source commit `947381757156ec5a9251485ea8aebfeb99ce6af6`.

#### LoCoMo — matched budgets

| Budget | Append | Progression only | Fixed | Adaptive v1 | Adaptive Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 1,200 chars | **55.20%** | 27.49% | 43.88% | 52.52% | **-2.68 pp** [-3.52, -1.81] |
| 2,400 chars | **63.11%** | 39.74% | 55.41% | 60.78% | **-2.33 pp** [-3.11, -1.61] |
| 4,800 chars | **68.65%** | 53.36% | 64.22% | 66.42% | **-2.23 pp** [-2.91, -1.56] |

Adaptive v1 is the closest structured system at equal LoCoMo budgets, but it is significantly worse than append-only at all three budgets. The issue is no longer hidden by equal top-k: coarse or over-split memory units consume budget inefficiently.

#### LongMemEval-S — matched budgets

| Budget | Append | Progression only | Fixed | Adaptive v1 | Adaptive Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 100k chars | **93.59%** | 87.28% | 92.21% | 89.15% | **-4.44 pp** [-6.64, -2.45] |
| 150k chars | **95.77%** | 93.56% | 95.12% | 94.15% | **-1.61 pp** [-3.18, -0.10] |
| 200k chars | 96.69% | 96.81% | 96.88% | **97.02%** | **+0.33 pp** [-0.73, +1.36] |

At 200k characters adaptive v1 has a small positive point estimate, but the bootstrap interval crosses zero and the paired all-evidence test is not significant (`p=0.118`). We therefore do **not** claim a budget-matched win.

### 4. Current interpretation

The experiments now support a narrower and stronger conclusion:

1. **Progression grouping improves raw ranking/coverage.** Progression-only is the equal-top-k recall leader on both benchmarks.
2. **That gain does not yet translate into context efficiency.** Under strict equal-character budgets, append-only remains stronger at practical budgets.
3. **The current memory payload is too coarse.** Progression-only often ranks the right region but cannot fit enough whole progression units into the same context budget.
4. **Adaptive v1 is not robust across memory regimes.** It over-splits LoCoMo and still loses at matched budgets.
5. **The next architecture target is twofold:** stabilize adaptive segmentation, then separate compact memory indexing from source-evidence expansion so progression structure can improve ranking without forcing the reader to ingest entire progression payloads.

The complete CI artifacts retain all top-k and budget JSON outputs.

## Published reference targets

These are external paper results used only as targets until reproduced under a common reader/evaluator setup.

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Published end-to-end QA result; reports 52.20% lower recalled memory length on LoCoMo. |
| Temporal Semantic Memory | 74.8% reported setup | 76.69% reported setup | Published end-to-end QA result under its reported setup. |
| HeLa-Mem | 65.4% reported setup | - | Published end-to-end QA result under its reported setup. |

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
7. **NEXT** — adaptive v2: confidence-aware state-change damping, minimum segment evidence, and split hysteresis; preserve v1 as an ablation.
8. **NEXT** — two-stage compact progression/segment index followed by budgeted source-evidence expansion.
9. **NEXT** — fixed reader + official LongMemEval/LoCoMo end-to-end QA evaluation under one model and context budget.
10. **LATER** — learned temporal policy, hierarchy, consolidation, knowledge objects, and hot/warm/cold memory.

For every stage we record evidence coverage, recalled context size, memory/index size, paired uncertainty, and answer accuracy/F1 once a fixed reader is available.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
