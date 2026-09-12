# Evaluation and Benchmark Ledger

This directory is the source of truth for progression-memory evaluation. Reproduced measurements are kept separate from published external reference numbers. External QA scores are not compared numerically with retrieval recall until the same reader, context budget, and evaluator are reproduced.

## 1. Synthetic temporal-boundary mechanism check

The synthetic benchmark contains 45 episodes with true boundaries at episodes 12 and 29.

| System | Boundary precision | Boundary recall | Boundary F1 | Segments |
|---|---:|---:|---:|---:|
| Fixed 14-day windows | 0.1667 | 0.5000 | 0.2500 | 7 |
| Adaptive linear controller | **1.0000** | **1.0000** | **1.0000** | **3** |

Raw result: `results/synthetic_v1.json`. This validates the mechanism only.

## 2. Equal-top-k retrieval

Full public LoCoMo and LongMemEval-S are run in GitHub Actions with the same dependency-free BM25/evidence-aware evaluation code. LongMemEval excludes 30 abstention questions, leaving 470 evidence-bearing questions; LoCoMo excludes four questions without annotated evidence.

### LongMemEval-S — top-8

| System | Macro evidence recall | All evidence | MRR@8 | Avg retrieved chars |
|---|---:|---:|---:|---:|
| Append-only BM25 | 94.11% | 88.72% | 0.9149 | 101,661 |
| Fixed progression | 95.83% | 92.13% | **0.9263** | 144,636 |
| Adaptive progression v1 | 96.77% | 94.04% | 0.9257 | 155,653 |
| Adaptive progression v2 | 96.76% | 93.40% | 0.9223 | 149,694 |
| Progression only | **97.16%** | **94.89%** | 0.9229 | 173,672 |
| Two-stage retrieval v2 | 94.35% | 88.72% | 0.9262 | 101,505 |
| **DAG two-stage v1** | 94.35% | 88.72% | 0.9262 | **101,498** |

The source-level systems intentionally measure the final raw-memory ranking rather than giving evidence credit to structural index nodes. Structural progression/segment rows can therefore have higher raw top-k recall because one retrieved structural unit exposes multiple evidence items.

### LoCoMo — top-8

| System | Macro evidence recall | All evidence | MRR@8 | Avg retrieved chars | Avg structural units |
|---|---:|---:|---:|---:|---:|
| Append-only BM25 | 56.18% | 52.67% | 0.4047 | **1,172** | 588.2 raw turns |
| Fixed progression | 61.58% | 57.62% | 0.4529 | 3,546 | 320.5 |
| Adaptive progression v1 | 54.34% | 50.91% | 0.4003 | 1,323 | 746.3 |
| Adaptive progression v2 | 60.20% | 56.10% | 0.4501 | 2,945 | 339.7 |
| Progression only | **64.73%** | **60.24%** | **0.4572** | 9,379 | 163.6 |
| Two-stage retrieval v2 | **56.49%** | **52.72%** | 0.4083 | 1,203 | 339.7 index units |
| DAG two-stage v1 | 56.40% | 52.62% | **0.4102** | 1,206 | 339.7 segment nodes |

Adaptive v2 remains the better raw structural-ranking system. The two-stage systems deliberately give up whole-segment evidence credit in exchange for source-level payload control.

## 3. Adaptive segmentation v2

Adaptive v2 added confidence-aware change damping, minimum segment evidence, and split hysteresis without changing the original eight-dimensional formal feature vector.

On LoCoMo it reduces retrieval units from **746.3 to 339.7 (-54.5%)**, increases average segment size from 1.12 to 2.47 events, and lowers the single-event-segment fraction from **93.3% to 27.1%**. Top-8 macro evidence recall improves **54.34% -> 60.20%**.

On LongMemEval it essentially preserves macro top-8 recall (96.77% -> 96.76%) while reducing average top-8 payload from about 155.7k to 149.7k characters.

Compact result: `results/adaptive_v2_summary.json`. Provenance run: `34618743057`.

## 4. Two-stage retrieval v2

The v2 retrieval path separates structural search from reader payload:

1. Build compact adaptive-v2 progression/segment index units.
2. Rank structural units with BM25.
3. Expand only raw source documents under selected units.
4. De-duplicate raw memories that appear under multiple structures.
5. Re-rank those raw sources with BM25.
6. Charge evidence credit and context budget only to raw source text actually exposed to the reader.

The first implementation used a fixed 16-unit Stage-1 candidate set. That exposed Stage 1 as a recall ceiling on LoCoMo, with only 66.66% average ground-truth evidence coverage. Commit `474c9bdbef0f97e3a739669181b77ec214c8a400` scales candidate breadth as `max(16, min(64, ceil(0.10 * structural_units)))`. This raises LoCoMo Stage-1 evidence coverage to **73.54%**, with 35.29 structural candidates and 75.08 candidate source documents on average.

### LoCoMo — equal character budgets

| Budget | Append | Adaptive v2 whole-segment | Two-stage v2 | Two-stage Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 1,200 chars | **55.20%** | 46.21% | 55.05% | -0.14 pp [-0.78, +0.46] |
| 2,400 chars | 63.11% | 56.74% | **63.55%** | +0.44 pp [-0.21, +1.10] |
| 4,800 chars | 68.65% | 65.18% | **68.90%** | +0.24 pp [-0.48, +0.99] |

Selective source expansion eliminates almost all of the LoCoMo context-efficiency regression caused by larger Adaptive-v2 segments. The 2.4k and 4.8k point estimates exceed append-only, but the confidence intervals cross zero, so these are not statistically established wins.

### LongMemEval-S — equal character budgets

| Budget | Append | Adaptive v2 whole-segment | Two-stage v2 | Two-stage Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 100k | **93.59%** | 92.74% | 93.49% | -0.10 pp [-0.54, +0.33] |
| 150k | **95.77%** | 95.58% | 95.70% | -0.06 pp [-0.52, +0.40] |
| 200k | 96.69% | **97.21%** | 97.18% | +0.49 pp [-0.14, +1.22] |

Retrieval-v2 implementation provenance: `326ea8f0d28af51dba45f1259c2686ae57a3127c`; scaled Stage-1 policy: `474c9bdbef0f97e3a739669181b77ec214c8a400`; successful benchmark run: `34626603557`. Compact result: `results/retrieval_v2_summary.json`.

## 5. DAG v1 — incremental hierarchy ablation

DAG v1 adds an explicit `ProgressionGraph` with progression and segment nodes, typed edges, multi-parent support, cycle prevention, ancestor/descendant traversal, and mirrored progression parent/child invariants. For retrieval, each progression is a parent of its adaptive-v2 segment nodes. Parent and child BM25 rankings are fused with reciprocal-rank fusion (`child + 0.35 * parent`, `k=60`), followed by the exact same raw-source Stage-2 reranker as flat two-stage retrieval.

This is deliberately an incremental ablation: the candidate breadth is unchanged, structural nodes receive no evidence credit, and only hierarchy/ranking differs.

### Candidate-routing effect

| Dataset | Flat Stage-1 evidence coverage | DAG Stage-1 evidence coverage | Flat candidate raw docs | DAG candidate raw docs |
|---|---:|---:|---:|---:|
| LoCoMo | 73.54% | **73.65%** | 75.08 | 75.14 |
| LongMemEval-S | 98.24% | **98.40%** | 20.25 | 20.28 |

The hierarchy effect is therefore small. It is positive at the candidate-coverage level, but DAG v1 does not materially solve LoCoMo routing.

### LoCoMo — equal character budgets

| Budget | Append | Flat two-stage | DAG v1 | DAG Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 1,200 | **55.20%** | 55.05% | 55.01% | -0.19 pp [-0.79, +0.39] |
| 2,400 | 63.11% | **63.55%** | 63.54% | +0.43 pp [-0.18, +1.09] |
| 4,800 | 68.65% | **68.90%** | 68.88% | +0.23 pp [-0.48, +0.93] |

### LongMemEval-S — equal character budgets

| Budget | Append | Flat two-stage | DAG v1 | DAG Δ vs append (95% CI) |
|---:|---:|---:|---:|---:|
| 100k | 93.59% | 93.49% | **93.74%** | +0.15 pp [-0.40, +0.80] |
| 150k | 95.77% | 95.70% | **95.81%** | +0.04 pp [-0.45, +0.55] |
| 200k | 96.69% | 97.18% | **97.29%** | +0.60 pp [0.00, +1.31] |

The 200k LongMemEval point is encouraging, but DAG-v1 must still be compared directly against flat two-stage with paired uncertainty before claiming a hierarchy-specific improvement.

### DAG-v1 cost regression

The first parent representation is an uncapped union of child index tokens. That makes the hierarchy unnecessarily expensive. On LoCoMo, segment-index text averages ~73.4k characters and parent-index text adds ~60.2k, for **133.6k structural index characters versus 83.7k raw characters (1.60x raw)**. LongMemEval total hierarchy index is ~0.59x raw.

This overhead is not justified by the tiny LoCoMo routing gain. The next hierarchy iteration should therefore (a) cap/compact parent representations, (b) use parent-aware child diversification rather than only a rank nudge, and (c) report direct DAG-vs-flat paired significance.

DAG implementation: `5f2e8091d584875cb0cb7414ff6d89277b45bd2d`; successful benchmark run: `34644525092`; compact result: `results/dag_v1_summary.json`.

## 6. Current interpretation

The current evidence supports the following architecture story:

1. **Progression grouping carries useful signal.** Coarse progression retrieval remains the raw top-k recall leader.
2. **Adaptive v2 fixes the main segmentation-stability failure.** It substantially reduces LoCoMo over-segmentation while improving raw retrieval quality.
3. **Two-stage retrieval fixes the payload problem.** Selective source expansion makes structured retrieval approximately budget-neutral versus append-only instead of strongly worse.
4. **Flat Stage-1 candidate coverage is the main LoCoMo retrieval bottleneck.** The current hierarchy improves that ceiling by only ~0.11 pp.
5. **DAG v1 validates the graph abstraction, not yet a strong retrieval advantage.** We need better hierarchy construction/routing, not merely more graph structure.
6. **Index efficiency is now a first-class metric.** A hierarchy should improve routing without storing more lexical index text than the raw memory itself.

## Published reference targets

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Published end-to-end QA; not retrieval recall. |
| Temporal Semantic Memory | 74.8% reported setup | 76.69% reported setup | Published end-to-end QA. |
| HeLa-Mem | 65.4% reported setup | - | Published end-to-end QA. |

Primary references: TiMem (Findings ACL 2026), LongMemEval public benchmark, and Snap Research LoCoMo. These external percentages must not be directly compared with the retrieval numbers above.

## Evaluation roadmap

1. **DONE** — dataset adapters and evidence-aware retrieval metrics.
2. **DONE** — append-only BM25 baseline.
3. **DONE** — progression-only routing/grouping ablation.
4. **DONE** — fixed-window and adaptive-v1 progression retrieval.
5. **DONE** — strict equal-character-budget evaluation and paired uncertainty.
6. **DONE** — adaptive segmentation v2.
7. **DONE** — two-stage compact structural index -> source-evidence expansion.
8. **DONE** — DAG v1 graph abstraction + hierarchical two-stage retrieval.
9. **NEXT** — hierarchy v2: compact parent indexes + parent-aware candidate diversification + paired DAG-vs-flat ablation.
10. **NEXT** — graph-aware progression decider with `CREATE_CHILD`, multi-parent support, and graph-aware candidates.
11. **NEXT** — fixed-reader end-to-end LongMemEval/LoCoMo QA.
12. **LATER** — LLM progression decider, learned temporal policy, higher-level consolidation, knowledge objects, and hot/warm/cold lifecycle.

## Reproducibility rule

A row is labeled reproduced only when it was generated by code in this repository. Every architecture change is benchmarked and recorded before the next stage is introduced.
