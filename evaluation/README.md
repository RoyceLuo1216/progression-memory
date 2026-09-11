# Evaluation and Benchmark Ledger

This directory is the source of truth for progression-memory evaluation results. Results are split into **reproduced local measurements** and **published external reference numbers**. A published number is never treated as an apples-to-apples comparison until the same dataset, reader model, retrieval budget, and evaluator are reproduced here.

## Current reproduced results

### Synthetic irregular-transition benchmark

The v1 synthetic harness contains 45 episodes with true phase boundaries at episodes 12 and 29. The benchmark tests whether a temporal memory policy discovers state-dependent boundaries rather than imposing calendar windows.

| System | Boundary precision | Boundary recall | Boundary F1 | Segments | Episodes / segment |
|---|---:|---:|---:|---:|---:|
| Fixed 14-day windows | 0.1667 | 0.5000 | 0.2500 | 7 | 6.43 |
| Adaptive linear controller | **1.0000** | **1.0000** | **1.0000** | **3** | **15.00** |

Raw result: `results/synthetic_v1.json`.

**What this establishes:** adaptive EXTEND/SPLIT segmentation can recover irregular state transitions more faithfully and compactly than a fixed-window baseline in a controlled setting.

**What it does not establish:** superior LongMemEval/LoCoMo question answering. The synthetic benchmark is a mechanism validation, not an end-to-end long-term-memory benchmark.

## Published reference targets

These are external paper results used only as targets until reproduced under a common setup.

| System | LongMemEval-S | LoCoMo | Notes |
|---|---:|---:|---|
| TiMem | 76.88% | 75.30% | Reports 52.20% lower recalled memory length on LoCoMo. Findings of ACL 2026. |
| Temporal Semantic Memory (Beyond Dialogue Time) | 74.8% reported setup | 76.69% reported setup | Temporal-semantic/durative memory; exact model/evaluator settings must be matched before direct comparison. |
| HeLa-Mem | 65.4% reported setup | - | Dynamic associative memory; exact setup must be matched before direct comparison. |

Primary references:
- TiMem: https://aclanthology.org/2026.findings-acl.1091/
- Temporal Semantic Memory: ACL 2026, "Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents"
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo

## Evaluation roadmap

The staged comparison is deliberately ablation-first:

1. **Dataset adapters + evidence-aware retrieval metrics** on LongMemEval-S and LoCoMo.
2. **Append-only lexical retrieval baseline** with identical retrieval budgets.
3. **Progression retrieval without adaptive segmentation**.
4. **Fixed-window progression retrieval**.
5. **Adaptive progression retrieval (current EXTEND/SPLIT controller)**.
6. **End-to-end reader/evaluator** using one fixed answering model and the official benchmark evaluators.
7. Only after the above: learned temporal policy, hierarchical consolidation, and hot/warm/cold memory.

For every stage we will record: retrieval recall@k/evidence coverage, answer accuracy/F1 where a reader is available, recalled context size, active-memory size, and latency.

## Reproducibility rule

A row is labeled `reproduced` only if it was generated from code in this repository in the current evaluation pipeline. External paper scores remain `published reference` values until independently reproduced.
