# Strict evidence visibility — corrected stress test

The source-attribution correction changes our conclusion: **the old 99% V2
evidence-ID scores do not establish source-visible retrieval advantage**.

This evaluates 10 fictional projects and 220 questions over 1/4/8 distractors
per decision stage (210/510/910 observations). Writer proposals are scripted
oracle decisions; no LLM writer or answering model was run.

## Strict source-visible recall at 1,600 characters

| System | 210 observations | 510 observations | 910 observations |
|---|---:|---:|---:|
| Append-only RAG BM25 | 59.09% | 54.55% | 54.55% |
| V2 flat BM25 | 50.00% | 50.00% | 50.00% |
| V2 hierarchical BM25 | 50.00% | 50.00% | 50.00% |
| V2 oracle-time historical | 59.55% | 59.55% | 59.55% |

The strict evaluator credits **only original source observation documents**
actually present within a matching character budget. It never credits source
IDs from summarized progression states. Thus it is a conservative source
visibility metric, not a metric of whether a structured state entails the answer.

The historical variant materializes time-filtered observations and reconstructed
state as of a timestamp derived from gold source IDs. It is an **oracle time
router**, not a learned/automated temporal segment selector; benchmark-label
timestamps cannot be provided to a deployed retriever.

Limitations: synthetic repeated project patterns, scripted perfect writer,
no answer generation, no hybrid dense retrieval, no independent human labels,
and original evidence-only attribution may undervalue genuinely sufficient
state snapshots.

## Reproduce

```bash
python -m progression_memory.benchmarks.v2_strict_visibility \
  --distractors 8 \
  --output results/v2_strict_8.json
```

CI artifacts for 1/4/8 distractors:
https://github.com/RoyceLuo1216/progression-memory/actions/runs/40427540027

## Next necessary tests

1. Add temporal segment units and query-time inference without gold labels.
2. Test whether state text or historical snapshot actually entails the gold
   answer; source-only visibility is an intentionally different metric.
3. Run the same reader LLM over equal-budget contexts for answer-level accuracy.
4. Compare live LLM writer against oracle writer with gold identity and state
   snapshots.
5. Add dense and time-aware RAG baselines and diverse project histories.
