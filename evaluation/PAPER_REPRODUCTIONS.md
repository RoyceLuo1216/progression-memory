# Paper-native progression experiments

The temporal/semantic component ablations in `baseline_retrieval.py` are **not**
reproductions of TiMem or Temporal Semantic Memory (TSM). Do not compare their
retrieval recall with the papers' reported end-to-end QA accuracy.

`paper_pairs.json` pins the public source revisions and records the native
entrypoints. Run `python -m progression_memory.benchmarks.paper_pairs prepare
tsm /tmp/tsm --fetch` (or `timem`) to obtain the exact revision. The source
contract CI only checks code provenance, configuration validation and Python
syntax. It does **not** execute a benchmark or claim a paper result.

The native runner verifies source SHA and the original dataset SHA256, checks
for Python 3.11, Docker and a configured model key, and records each upstream
command and exit code. It runs the paper's original entrypoints, not the
component-ablation code. Set up the paper's native services and dependencies
first; no credentials are stored in this repository. For TSM:

```
python -m progression_memory.benchmarks.paper_native tsm locomo /tmp/tsm
python -m progression_memory.benchmarks.paper_native tsm locomo /tmp/tsm \
  --output /tmp/tsm-locomo-native --acknowledge-model-calls
```

For TiMem, place the same checksum-verified LoCoMo file at
`data/locomo10.json` or LongMemEval-S at `data/longmemeval_s_cleaned.json`.
Configure its native PostgreSQL/Qdrant and model provider, then run with
`--isolated-database --acknowledge-model-calls`. The isolation flag is
mandatory: TiMem's native generation scripts clear existing memory tables.
The runner first calls TiMem's dataset splitter, then native generation,
retrieval and evaluation scripts. Its output is a run manifest and per-step
logs; preserve the paper-generated predictions and evaluator files as well.

Manual GitHub workflows `paper-native-tsm` and `paper-native-timem` expose
each dataset as an input and archive native outputs. They require an
`OPENAI_API_KEY` repository secret. TiMem additionally downloads its pinned
Qwen3-Embedding-0.6B model; its released settings select that local embedding
provider. These workflows are live model-backed runs and can take substantial
time and tokens. They have not yet produced benchmark results. TiMem's native
evaluation and TSM's released diagnostic metrics must still be aligned to a
single pinned evaluator before claiming a cross-paper QA ranking.

## Required experiment matrix

Run each dataset (LoCoMo and LongMemEval-S) under three paired comparisons:

| Native baseline | Same pipeline with progressions | Change allowed |
|---|---|---|
| TiMem | TiMem + progression | Construction after native L1 evidence; native retriever and reader unchanged |
| TSM | TSM + progression | Construction after temporal fact extraction; native retriever and reader unchanged |
| TiMem + TSM | TiMem + TSM + progression | Only progression construction differs |

The combined architecture does not exist in either source repository. Specify
its writer, temporal index, retriever and reader, then freeze it before the
last pair. A combined result cannot be inferred by pooling independent scores.

## Acceptance gates

1. Reproduce each untouched paper on the exact dataset bytes, native model
   configuration and official evaluator. Record the source and evaluator SHAs,
   prompts/models, dataset checksum, split, outputs, cost and latency. TSM's
   public release explicitly distinguishes offline checks from live API runs.
2. Add progressions to a *separate* revision of each paper's writer. Preserve
   source pointers and event time. Show that native retrieval can see the
   constructed progression while holding the same query-time evidence budget.
   Do not simply append more context. Log create/attach/transition decisions,
   group membership, unrelated attachments, state updates and depth.
3. Evaluate native retrieval evidence and end-to-end QA on identical question
   IDs with the same reader, judge and budget within each pair. Report category
   breakdowns, paired improvements/regressions, token cost and construction
   coherence. If source pointers cannot be mapped to benchmark evidence IDs,
   mark retrieval recall unavailable; do not count entire groups as hits.
4. Only then freeze and test the combined architecture, with and without the
   same progression intervention.

The comparison CLI accepts two normalized JSON files and rejects mismatched
paper revisions, dataset bytes/splits, models, budgets, evaluator revisions,
question IDs, missing provenance or an undeclared pair. Fields are
`system`, `dataset`, `paper_revision`, `implementation_revision`, `dataset_sha256`, `dataset_split`,
`llm_model`, `embedding_model`, `reader_model`, `retrieval_budget`,
`evaluator_revision`, `native_run_manifest`, `official_evaluator_output`, and
`question_results` (question ID to official evaluator boolean). The manifest
and evaluator fields are artifact references, not self-contained proof; archive
the referenced original files beside each report.

```
python -m progression_memory.benchmarks.paper_pairs compare \
  runs/tsm_baseline.json runs/tsm_progression.json \
  --output runs/tsm_pair.json
```

**Status:** the published paper repositories are identified and pinned. The
upstream configurations validate in source-contract CI. No paper-native live
baseline or progression intervention has run through this project yet, so
there are no paired accuracy numbers to report.
