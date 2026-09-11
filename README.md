# progression-memory

Research implementation for progression-aware long-term memory in LLM agents.

The system organizes episodic evidence into evolving user synopses called **progressions** and
maintains a reversible temporal hierarchy over the evidence.

## Implemented v1 slice

- idempotent episode ingestion and evidence versioning;
- progression statistics and evidence-preserving storage;
- active progression cache;
- deterministic progression routing (`ATTACH`, `CREATE`, `MULTI_ATTACH`, `STATIC`);
- adaptive temporal segmentation with `EXTEND` / `SPLIT`;
- rule-based and lightweight linear temporal controllers;
- fixed-window temporal abstraction baseline;
- deterministic synthetic benchmark for phase-boundary detection and compression;
- tests for idempotency, routing, segmentation invariants, and benchmark behavior.

## Architecture

```text
Episode
  -> progression routing
  -> progression store
  -> temporal controller
       -> EXTEND current segment
       -> SPLIT into a new segment
  -> evidence-backed temporal segments
```

Every episode attached to a progression belongs to exactly one finest-level temporal segment.
Source episodes are never destructively overwritten by segmentation.

## Run tests

```bash
python -m pytest
```

## Run the first benchmark

```bash
python -m progression_memory.benchmarks.synthetic
```

The synthetic benchmark compares a fixed 14-day baseline against the initial adaptive linear
controller on irregular state transitions. It reports boundary precision/recall/F1, segment
count, and the episode-to-segment compression ratio.

## Next benchmark integration

This local deterministic harness is intentionally small. The next step is to plug the same
ablation interface into LoCoMo / LongMemEval so progression representation and temporal
segmentation can be measured against downstream retrieval/reasoning quality as well as memory
growth and latency.
