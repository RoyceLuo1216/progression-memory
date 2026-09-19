# Process-aware SLM construction experiment

This is a process-construction experiment, not a replacement for all V2/V3/V4 systems.
`real_retrieval --system process_slm` actually instantiates a local language model and
uses it for admission, process assignment and state updates. The older systems remain
heuristic baselines. No hierarchy depth or downstream QA improvement is claimed.

## Construction

Instructions and development demonstrations are separate chat messages; evidence is
passed in a user message. Each turn is presented with the previous three turns (reference resolution only) and
up to six existing processes. The SLM emits zero to four updates with subject,
process identity, factual claim, current state and delta. It chooses create, extend,
or transition. A transition updates the same process. Exact proposed identity and ID
must be preserved for attachment. Multiple independent processes in a turn can receive
separate updates. A greeting/question need not create a process.

An intentionally conservative gate skips only obvious greetings, acknowledgements,
praise and question-only turns. All uncertain statements reach the SLM. Local
Transformers calls use grammar-constrained generation from the Pydantic response
schema; semantic validation still rejects unknown IDs or changed process identities.

Candidate proposal weights are 0.35 semantic identity, 0.25 current state, 0.20 most
recent extracted fact, 0.10 speaker/subject match and 0.10 recency. These are uncalibrated proposal
weights, not an attachment threshold. The SLM makes the final membership decision.
The identity is stable; current state is independently generated rather than copied
from an episode vector or a growing transcript. Each update retains its source ID,
prior state, new state and action for inspection.

Validated updates are executed through the canonical `ProgressionStore`: the source
becomes an `Episode`, membership remains many-to-many, every update creates a
`ProgressionStateCheckpoint`, and a model-declared transition starts a new temporal
segment. This removes the previous duplicate flat `Process` representation.

All original source text is retained. A turn without a process update remains an
unassigned retrieval unit. This is essential for useful standalone facts and admission
mistakes. Process summaries are indexing text only and receive no evidence credit;
retrieval expands to and scores original source turns. The candidate stage and final
ranking are explicitly BM25, so a semantic ranking is not silently replaced by the
evaluator. Stage-one limit is max(16, min(64, ceil(0.10 * units))).

## Running

```bash
pip install -e '.[dev,semantic,slm]'
# Real model calls on the labelled development fixture:
OMP_NUM_THREADS=2 python -m progression_memory.benchmarks.process_writer_benchmark \
  --dataset fixture --enforce-quality \
  --output evaluation/results/process-dev/fixture.json

# Real model calls on a bounded LoCoMo construction smoke:
OMP_NUM_THREADS=2 python -m progression_memory.benchmarks.real_retrieval \
  --dataset locomo --data-path evaluation/data/locomo10.json --system process_slm \
  --writer-model Qwen/Qwen2.5-1.5B-Instruct --corpus-limit 1 --max-turns 32 \
  --output evaluation/results/process-dev/locomo-smoke.json

# Full LoCoMo construction is explicitly more expensive: omit --max-turns,
# set --corpus-limit 10. Run after inspecting smoke output.

# Strict model comparison on the same labelled construction prefix:
OMP_NUM_THREADS=2 python -m progression_memory.benchmarks.writer_model_shootout \
  --max-turns 12 --output evaluation/results/writer-shootout/summary.json
```

`.github/workflows/process-writer-benchmark.yml` installs the SLM dependencies and
runs a four-turn quality-gated fixture and an eight-turn LoCoMo smoke. The model cache
is reused across CI runs and artifacts upload even after failure. The small default model is an experimental CPU baseline, not a selected
best-performing writer or a demonstrated distilled classifier.

The shootout ranks only completed, schema-valid runs, then construction grouping,
admission/action quality and latency. It does not use the older heuristic fallback.

## Reading results

- `writer`: actual calls, valid/invalid responses, model errors, zero fallback calls,
  no-update decisions, creates/transitions/extensions and total generation time.
- `corpora`: complete process memberships and state histories, largest-group fraction,
  singleton fraction, assigned/unassigned source counts and membership duplication.
- Development fixture: pair precision/recall/F1, cluster purity, fragmentation,
  cross-subject contamination, admission precision/recall, gate event recall and action
  accuracy. Labels are hand-authored development expectations, not a held-out test set.
- LoCoMo smoke: only questions whose **complete** gold evidence is present in the
  selected prefix are evaluated; total/eligible/excluded counts are explicit.
  This conditional subset is not comparable to full-LoCoMo published/previous scores.
  The append baseline uses exactly the same source prefix and eligible questions.
- `.trace.jsonl`: input source, proposed candidates, raw model response, parsed updates,
  failures and latency. No benchmark questions or gold answers reach the writer.

A single enclosing JSON markdown fence is removed and counted in `fenced_responses`
for backends without constrained decoding;
the original response remains in the trace. Invalid JSON/schema, unknown IDs and
changed attachment identities fail the run.
Runtime errors also fail. Failure status/counters and the raw failing response are
saved. There is no automatic heuristic fallback. Zero calls or all-no-update output
cannot pass the construction smoke. Schema validity alone cannot prove factuality or
coherence: inspect source-linked histories and add independently labelled real
conversation examples before broad benchmarking or model selection.
