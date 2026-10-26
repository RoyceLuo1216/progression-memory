# Domain-specific long-horizon dataset (synthetic oracle)

This fixture is designed for **one persistent agent-memory research program**,
rather than generic trivia or short, unrelated dialogues.

The default generator produces 197 interleaved updates over >12 months:
one parent research progression, four related workstreams (retrieval,
writer, evaluation, hierarchy), 48 updates per workstream, and 65 labeled
questions. It includes time-separated decisions, repeated topic vocabulary,
phase transitions, late-recorded corrections, and evidence across workstreams.

The default data is **fully synthetic, labeled with scripted oracle updates**.
Passing these experiments establishes retriever/engine execution and reveals
whether routing helps under controlled structured memory. It does *not*
measure LLM writer quality or end-to-end response accuracy.

## Generate and test

```bash
python -m progression_memory.benchmarks.v2_long_horizon_dataset \
  --output data/v2/long_horizon_agent_research.json

python -m progression_memory.benchmarks.v2_paired \
  --fixture data/v2/long_horizon_agent_research.json \
  --budgets 400,800,1600,3200 \
  --output results/v2_long_horizon_paired.json

python -m progression_memory.benchmarks.v2_retrieval_ablation \
  --fixture data/v2/long_horizon_agent_research.json \
  --budgets 400,800,1600,3200 \
  --output results/v2_long_horizon_ablation_lexical.json
```

Dense modes need `sentence-transformers` and a real embedding model:

```bash
pip install sentence-transformers
python -m progression_memory.benchmarks.v2_retrieval_ablation \
  --fixture data/v2/long_horizon_agent_research.json \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --output results/v2_long_horizon_ablation_dense.json
```

Do not interpret omitted/skipped dense variants as measured scores.
Avoid tuning on this fixed fixture and reporting the same queries as held-out.

## Research-grade next dataset

For the real study, supplement this oracle fixture with **human-authored or
consented real development logs** for one persistent project over six to
twelve months, emphasizing changes in implementation strategy, constraints,
bug/correction cycles, unresolved decisions and resumed tasks.

Use two splits: a development split to debug retrieval and a held-out project
history for measurement. Annotate each query with the correct source
observations, optional query-time cutoff and independently determined state;
freeze labels before model-based memory construction. Run the real V2 model
writer without exposing questions or gold evidence during ingestion.

LoCoMo and LongMemEval are useful external validity checks but contain
broader memory topics. This domain dataset deliberately targets sustained,
same-domain research-workflow continuity.

The current temporal mode still uses corpus-end time, so as-of historical
accuracy is NOT measured by this version. A question-specific temporal
benchmark must be implemented separately.
