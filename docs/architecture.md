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
