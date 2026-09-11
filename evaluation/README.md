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

**Established:** adaptive EXTEND/SPLIT segmentation can recover irregular state transitions more faithfully and compactly than a fixed-window baseline in a controlled setting.

**Not established by this test:** superior LongMemEval/LoCoMo question answering. This is mechanism validation only.

### 2. Real-dataset retrieval benchmark

Full public LoCoMo and LongMemEval-S are downloaded during GitHub Actions. All systems use the same dependency-free BM25 retriever and evidence-aware metric implementation. LongMemEval retrieval excludes the 30 abstention questions (`*_abs`), leaving 470 evidence-bearing questions; LoCoMo excludes four questions without annotated evidence.

The progression systems use deterministic hashed lexical vectors so the benchmark is fully reproducible without an external embedding API. Every progression/segment retains provenance links to its source evidence.

#### LongMemEval-S — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 97.87% | 88.72% | 94.11% | 91.91% | 0.9164 | **101,661** | 47.73 |
| Fixed progression | 98.30% | 92.13% | 95.83% | 94.83% | **0.9277** | 144,636 | 43.05 |
| Adaptive progression v1 | 98.51% | 94.04% | 96.77% | 96.18% | 0.9269 | 155,653 | 42.52 |
| **Progression only** | **98.51%** | **94.89%** | **97.16%** | **96.63%** | 0.9241 | 173,672 | **40.61** |

The progression-only ablation is the current raw top-k recall leader: **+3.05 percentage points macro recall** and **+6.17 points all-evidence recall** over append-only. It also compresses the retrieval index from 47.73 raw sessions to 40.61 progression units on average. However, each progression is large, so its recalled payload is about **70.8% larger** than append-only at top-8. This is not yet an efficiency win.

Adaptive temporal splitting recovers almost as much evidence as progression-only while recalling less text (155.7k vs 173.7k characters), but it still exceeds the append-only context budget. The next decisive experiment is therefore **equal-character-budget retrieval**, not another equal-top-k comparison.

#### LongMemEval-S — hypothesis-relevant categories at top-8

| Category | Append-only | Fixed progression | Adaptive v1 | Progression only |
|---|---:|---:|---:|---:|
| Temporal reasoning | 91.56% | 93.22% | 95.18% | **95.58%** |
| Multi-session | 89.68% | 93.82% | 94.99% | **96.09%** |
| Knowledge update | 99.31% | 99.31% | **100.00%** | **100.00%** |

This result sharpens the hypothesis: **progression organization itself is already valuable**, while the extra contribution of temporal segmentation must be demonstrated as an efficiency/compactness improvement under a matched budget.

#### LoCoMo — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 60.60% | 52.67% | 56.18% | 45.72% | 0.4162 | **1,172** | 588.2 |
| Fixed progression | 66.55% | 57.62% | 61.58% | 50.94% | 0.4646 | 3,546 | 320.5 |
| Adaptive progression v1 | 58.68% | 50.91% | 54.34% | 44.01% | 0.4121 | 1,323 | 746.3 |
| **Progression only** | **70.13%** | **60.24%** | **64.73%** | **55.17%** | **0.4709** | 9,379 | **163.6** |

Progression-only improves LoCoMo macro evidence recall by **+8.55 percentage points** over append-only and reduces the searchable memory units by about **72.2%**. The cost is a much larger retrieved payload because whole progressions are expanded at retrieval time.

Fixed temporal segmentation trades some of that recall for a much smaller payload. Adaptive v1 fails on LoCoMo because it over-splits short conversational turns: 746.3 retrieval units from 588.2 raw turns on average. This is strong evidence that the current per-event hashed lexical state proxy is too noisy for short utterances.

### 3. What the current ablation establishes

The real-data results currently support three separate conclusions:

1. **Progression routing/grouping is useful.** Progression-only is the raw-recall leader on both LongMemEval-S and LoCoMo.
2. **Temporal segmentation is not yet proven to improve retrieval quality by itself.** Its potential value is compactness and selective expansion, which must be measured at equal context budgets.
3. **Adaptive v1 is not robust across memory regimes.** It behaves reasonably on session-scale LongMemEval histories but over-segments short LoCoMo turns. The next controller must include state smoothing / hysteresis and information-aware split calibration rather than simply increasing model complexity.

Raw reproduced summaries:
- `results/longmemeval_append_bm25_v1.json`
- `results/locomo_append_bm25_v1.json`
- `results/longmemeval_fixed_progression_v1.json`
- `results/longmemeval_adaptive_progression_v1.json`
- `results/locomo_fixed_progression_v1.json`
- `results/locomo_adaptive_progression_v1.json`
- `results/longmemeval_progression_only_v1.json`
- `results/locomo_progression_only_v1.json`

Progression-only GitHub Actions provenance: run `34555762273`, source commit `6a37be42be612d13c6146690f01055a1b4e23d22`; tests passed **9/9**. The exact full JSON outputs are retained as the Actions artifact for that run.

## Published reference targets

These are external paper results used only as targets until reproduced under a common reader/evaluator setup.

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Reports 52.20% lower recalled memory length on LoCoMo. Findings of ACL 2026. |
| Temporal Semantic Memory (Beyond Dialogue Time) | 74.8% reported setup | 76.69% reported setup | Temporal-semantic/durative memory; exact model/evaluator settings must be matched before direct comparison. |
| HeLa-Mem | 65.4% reported setup | - | Dynamic associative memory; exact setup must be matched before direct comparison. |

These published percentages are **end-to-end QA scores**, not retrieval recall. They must not be compared numerically to the retrieval tables above.

Primary references:
- TiMem: https://aclanthology.org/2026.findings-acl.1091/
- Temporal Semantic Memory: ACL 2026, "Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents"
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo

## Evaluation roadmap

1. **DONE** — dataset adapters + evidence-aware retrieval metrics on LongMemEval-S and LoCoMo.
2. **DONE** — append-only lexical retrieval baseline.
3. **DONE** — progression-only routing/grouping ablation.
4. **DONE** — fixed-window progression retrieval.
5. **DONE** — adaptive progression v1 using the initial EXTEND/SPLIT controller.
6. **NEXT** — budget-matched retrieval: compare systems at equal recalled character budgets and report paired bootstrap confidence intervals / discrete paired tests.
7. **NEXT** — adaptive v2 with smoothed state, short-event damping, minimum segment evidence, and split hysteresis; preserve v1 as an ablation.
8. **NEXT** — two-stage compact progression/segment index followed by evidence expansion, so structure can improve retrieval without paying the full progression payload cost.
9. **NEXT** — fixed reader + official LongMemEval/LoCoMo end-to-end QA evaluation under a common model and context budget.
10. **LATER** — learned temporal policy, hierarchical consolidation, knowledge objects, reinforcement/decay, and hot/warm/cold memory.

For every stage we record evidence coverage, answer accuracy/F1 where a reader is available, recalled context size, retrieval-unit/memory size, and paired statistical uncertainty.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
