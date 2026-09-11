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

The progression systems use deterministic hashed lexical vectors so the benchmark is fully reproducible without an external embedding API. Every segment retains provenance links to its source evidence.

#### LongMemEval-S — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 97.87% | 88.72% | 94.11% | 91.91% | 0.9164 | **101,661** | 47.73 |
| Fixed progression | 98.30% | 92.13% | 95.83% | 94.83% | **0.9277** | 144,636 | 43.05 |
| **Adaptive progression** | **98.51%** | **94.04%** | **96.77%** | **96.18%** | 0.9269 | 155,653 | **42.52** |

Against append-only retrieval, adaptive progression improves macro evidence recall by **+2.66 percentage points** and all-evidence recall by **+5.32 points**. It also reduces the average number of retrieval units from 47.73 sessions to 42.52 segments. However, the retrieved segments are larger, so recalled text rises from about 101.7k to 155.7k characters. This is therefore a **retrieval-quality gain, not yet an efficiency/Pareto win**.

#### LongMemEval-S — categories most relevant to the hypothesis

| Category | Append-only macro recall | Fixed progression | Adaptive progression | Adaptive vs append |
|---|---:|---:|---:|---:|
| Temporal reasoning | 91.56% | 93.22% | **95.18%** | **+3.62 pp** |
| Multi-session | 89.68% | 93.82% | **94.99%** | **+5.30 pp** |
| Knowledge update | 99.31% | 99.31% | **100.00%** | **+0.69 pp** |

These are the categories where progression-aware temporal structure is expected to help, so the direction of the result is encouraging. End-to-end answer accuracy is still required before making a paper-level performance claim.

#### LoCoMo — top-8

| System | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR | Avg retrieved chars | Avg retrieval units |
|---|---:|---:|---:|---:|---:|---:|---:|
| Append-only BM25 | 60.60% | 52.67% | 56.18% | 45.72% | 0.4162 | **1,172** | 588.2 |
| **Fixed progression** | **66.55%** | **57.62%** | **61.58%** | **50.94%** | **0.4646** | 3,546 | **320.5** |
| Adaptive progression | 58.68% | 50.91% | 54.34% | 44.01% | 0.4121 | 1,323 | 746.3 |

Fixed progression gives a clear retrieval gain over append-only on LoCoMo: **+5.41 pp macro evidence recall** and roughly **45.5% fewer retrieval units**. The current adaptive controller does **not** generalize to LoCoMo: it over-splits short conversational turns, producing 746 retrieval units from 588 raw turns and reducing recall below append-only.

This failure is important. In the current real-data benchmark the adaptive controller's state vector is a deterministic lexical proxy (`0.5 * hashed_embedding`) rather than the semantic/state representation envisioned in the research design. LoCoMo's short turns make lexical state distance noisy, causing excessive SPLIT decisions. The next adaptive experiment therefore needs a better semantic state representation and calibrated/learned temporal policy rather than simply adding more hierarchy.

Raw reproduced result summaries:
- `results/longmemeval_append_bm25_v1.json`
- `results/locomo_append_bm25_v1.json`
- `results/longmemeval_fixed_progression_v1.json`
- `results/longmemeval_adaptive_progression_v1.json`
- `results/locomo_fixed_progression_v1.json`
- `results/locomo_adaptive_progression_v1.json`

GitHub Actions provenance for the fixed/adaptive sweep: run `34555484115`, source commit `86e6b84744e37840db34156969c7861e4452b001`; tests passed **9/9**. The exact full JSON outputs are also retained as the Actions artifact for that run.

## Progression-only ablation

Commit `6a37be42be612d13c6146690f01055a1b4e23d22` adds a **progression-only** system: routing is enabled, but each progression remains one temporal unit. This isolates the contribution of progression organization from fixed or adaptive temporal segmentation. Its full-data benchmark is executed by the same GitHub Actions workflow and will be recorded here separately from the fixed/adaptive sweep.

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
3. **DONE** — fixed-window progression retrieval.
4. **DONE** — adaptive progression retrieval using the initial EXTEND/SPLIT controller.
5. **RUNNING** — progression-only ablation to isolate routing from temporal segmentation.
6. **NEXT** — budget-matched retrieval: compare systems at equal recalled character/token budgets rather than equal top-k units.
7. **NEXT** — replace the lexical state proxy with semantic/state embeddings and calibrate or learn the temporal controller on development data only.
8. **NEXT** — two-stage compact progression/segment index followed by evidence expansion, so structure can improve retrieval without paying the full segment payload cost.
9. **NEXT** — paired statistical significance analysis and end-to-end reader/evaluator using one fixed answering model and the official benchmark evaluators.
10. **LATER** — hierarchical consolidation, knowledge objects, reinforcement/decay, and hot/warm/cold memory.

For every stage we record evidence coverage, answer accuracy/F1 where a reader is available, recalled context size, retrieval-unit/memory size, and eventually latency and paired significance tests.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
