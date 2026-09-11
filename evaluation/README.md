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

### 2. Real-dataset append-only retrieval baseline

Full public LoCoMo and LongMemEval-S data are downloaded during GitHub Actions. Retrieval uses the same dependency-free BM25 implementation and evidence-aware metrics that will be used for the progression ablations. LongMemEval retrieval excludes the 30 abstention questions (`*_abs`), leaving 470 evidence-bearing questions; LoCoMo excludes four questions that have no annotated evidence.

#### Top-8 retrieval

| Dataset | System | Eligible Q | Any evidence | All evidence | Macro evidence recall | Micro evidence recall | MRR | Avg retrieved chars |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LongMemEval-S | Append-only BM25 | 470 | 97.87% | 88.72% | **94.11%** | 91.91% | 0.9164 | **101,661** |
| LoCoMo | Append-only BM25 | 1,982 | 60.60% | 52.67% | **56.18%** | 45.72% | 0.4162 | **1,172** |

Raw results:
- `results/longmemeval_append_bm25_v1.json`
- `results/locomo_append_bm25_v1.json`

GitHub Actions provenance: run `34555086472`, source commit `b2d5663c9f4b9645939e5c8a3d638f9f13260d3d`. The test suite for that run passed 8/8.

The LongMemEval number makes the core efficiency problem visible: session-level BM25 can recover most annotated evidence at top-8, but it retrieves roughly 102k characters per question. A useful progression memory system should approach or exceed that evidence coverage with a much smaller recalled context. LoCoMo is the more difficult lexical-retrieval case and leaves substantially more headroom for structured memory.

## Progression ablation now running

Commit `be9490834a1579c5faacc978b9e53cb6069a95c6` adds two evidence-preserving real-data systems:

- **fixed progression**: deterministic progression routing + fixed 14-day temporal segmentation;
- **adaptive progression**: the same routing + the current adaptive EXTEND/SPLIT controller.

Both use deterministic hashed lexical vectors in this benchmark so no external embedding service is required. Segment retrieval units retain links to every source evidence item. The benchmark compares append-only, fixed progression, and adaptive progression at identical top-k values and records retrieval-unit counts and context size. These are retrieval-stage ablations, not yet the final paper-style answer-accuracy experiment.

## Published reference targets

These are external paper results used only as targets until reproduced under a common reader/evaluator setup.

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Reports 52.20% lower recalled memory length on LoCoMo. Findings of ACL 2026. |
| Temporal Semantic Memory (Beyond Dialogue Time) | 74.8% reported setup | 76.69% reported setup | Temporal-semantic/durative memory; exact model/evaluator settings must be matched before direct comparison. |
| HeLa-Mem | 65.4% reported setup | - | Dynamic associative memory; exact setup must be matched before direct comparison. |

These published percentages are **end-to-end QA scores**, not retrieval recall. They must not be compared numerically to the retrieval table above.

Primary references:
- TiMem: https://aclanthology.org/2026.findings-acl.1091/
- Temporal Semantic Memory: ACL 2026, "Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents"
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo

## Evaluation roadmap

The staged comparison is deliberately ablation-first:

1. **DONE** — dataset adapters + evidence-aware retrieval metrics on LongMemEval-S and LoCoMo.
2. **DONE** — append-only lexical retrieval baseline.
3. **RUNNING** — fixed-window progression retrieval.
4. **RUNNING** — adaptive progression retrieval (current EXTEND/SPLIT controller).
5. **NEXT** — budget-matched retrieval and category-level analysis (temporal, multi-session, updates, etc.).
6. **NEXT** — end-to-end reader/evaluator using one fixed answering model and the official benchmark evaluators.
7. **LATER** — learned temporal policy, hierarchical consolidation, knowledge objects, and hot/warm/cold memory.

For every stage we record evidence coverage, answer accuracy/F1 where a reader is available, recalled context size, retrieval-unit/memory size, and eventually latency and paired significance tests.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
