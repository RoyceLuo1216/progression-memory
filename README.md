# progression-memory

Research implementation for progression-aware long-term memory in LLM agents.

The system organizes episodic evidence into hierarchical, evolving user synopses called **progressions**. Initial milestones focus on:

1. idempotent episode ingestion,
2. progression DAG storage,
3. active progression caching and routing,
4. progression synopsis updates,
5. fixed vs. adaptive temporal segmentation,
6. hierarchical consolidation,
7. hot / warm / cold memory tiers.

## Initial architecture

```text
Interaction
  -> structured episode extraction
  -> active progression cache
  -> progression routing
  -> progression DAG
  -> synopsis/current-state update
  -> temporal segmentation
  -> background consolidation
  -> hot/warm/cold storage
```

## Development principle

Each addition should be evaluated as an ablation against the same LoCoMo / LongMemEval evaluation harness.
