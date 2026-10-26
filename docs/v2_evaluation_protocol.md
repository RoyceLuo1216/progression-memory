# Progression V2: evaluation protocol (v1)

The paired benchmark is a **retrieval diagnostic**, not a claim of end-to-end
answering accuracy or a substitute for writer construction evaluation.

## Inputs and leakage rules

A JSON fixture has `observations`, `proposals` (indexed by observation ID),
and `questions` with held-out gold `evidence_ids` and `subject_id`.
Construct all representations solely from observations and pre-recorded writer
proposals. Questions and gold evidence IDs are never shown to the writer.
The runner validates identifiers and fails closed on inconsistent proposals.

For LoCoMo/LongMemEval, use their existing loaders to prepare source turns,
then construct writer traces independently, **before** evaluation. Do not
synthesize state from QA answers or evidence annotations. Keep training,
development, and test dialogue identities separated.

## Systems and fair budgets

- `rag`: append-only raw source observations.
- `v2_state`: materialized per-progression state units (no raw-source rescue).
- `v2_full`: state plus original source evidence; parent-derived state can
  surface descendant evidence, but provenance should be audited.

All systems use the repository's identical BM25 scoring implementation and
whole-unit *character* budgets `400,800,1600,3200`. Retrieval stops at the first
oversized ranked unit. We intentionally do **not** give partial-unit credit.
Report macro evidence recall, all-evidence recall, MRR, average utilized
characters, per-category results, and per-question outcomes. Keep budgets
constant across systems; compare *quality versus cost*, not just recall.

## Run

```bash
python -m progression_memory.benchmarks.v2_paired \
    --fixture path/to/frozen_trace.json \
    --budgets 400,800,1600,3200 \
    --output results/v2_paired.json
```

Output includes the SHA256 of the complete input file, protocol version,
aggregate metrics and individual question outcomes. Store frozen inputs,
model identifiers, writer traces, prompts, inference settings, seed, elapsed
time, token usage, failure rates, and outputs alongside each experiment.

## Additional evaluation required before drawing conclusions

1. **Writer validity and attribution:** manually label held-out multi-turn
   streams; measure attachment precision/recall, duplicate processes,
   false creates, transition/split/close/revive correctness, unsupported
   patch rate, correction behavior, and time-alignment accuracy.
2. **Longitudinal QA:** use questions for current state, historical state,
   conflicting evidence, task continuity and archival revival. Require
   exact citations to original source IDs; human-review failure categories.
3. **End-to-end answers:** use one frozen downstream answer model, prompt,
   and context allocator for every system. Report grounded answer accuracy
   (not just source recall), invalid citation rate and uncertainty.
4. **Baselines/ablations:** append-only RAG, existing V1 variants,
   V2 state-only and full V2. Compare identical dialogue evidence,
   question splits, query-time budgets and writer construction costs.
5. **Uncertainty:** bootstrap **by conversation/corpus**, not by individual
   highly correlated questions; publish intervals and paired differences.
6. **Cost/growth:** memory units/bytes, write calls/tokens, consolidation
   calls, retrieval latency, query tokens, and stale/revival failure rates.

## Known limitations

This runner takes *recorded* writer proposals rather than invoking an LLM;
no model accuracy claim follows from passing smoke tests. It does not yet
include V1 in the paired fixture adapter, support as-of queries, nor measure
generation accuracy. State-document evidence associations are coarse and
must not be mistaken for direct entailment. The old LoCoMo and LongMemEval
loaders are reusable but need a frozen writer-trace conversion step to run
this specific runner on either dataset. Avoid calling smoke tests a
benchmark result.
