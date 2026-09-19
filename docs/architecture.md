# Architecture Notes

## Progression

A progression is a persistent hierarchical synopsis of an evolving area of a user's life, goals, preferences, or activities. It is not merely a semantic topic cluster.

Examples:
- career trajectory
- 2027 new-grad job search
- graduate-school applications
- research project
- fitness / recovery process

## Invariants

- Episode IDs are stable and ingestion is idempotent.
- Derived statistics are recomputable from source episodes.
- Semantic summaries are versioned by their evidence set.
- Progression structure is a DAG, not a strict tree.
- An episode may attach to multiple progressions.
- v1 does not destructively delete source evidence.
- Temporal resolution and storage tier are independent dimensions.

## Construction path

1. A conservative deterministic gate removes only obvious non-updates.
2. Weighted identity, state, recent-evidence, subject and recency signals propose candidates.
3. The SLM chooses zero or more validated create, extend or transition actions.
4. The deterministic executor ingests the episode, attaches evidence, writes a state
   checkpoint and starts a segment for explicit transitions.
5. Unassigned source evidence remains directly retrievable.

The SLM cannot directly mutate the store, select an unproposed progression, rewrite a
candidate identity, or delete evidence. Hierarchical split/merge maintenance is only
enabled after flat construction passes coherence and contamination quality gates.
