# Progression writer development data

| Source | Count | Supervision | Independence and use |
|---|---:|---|---|
| `progression_attachment/msc_train_pairs_v1.json` | 100 pairs from 25 events, 6 MSC conversations; 33 positive | Provisional event-to-candidate attachment | Same underlying MSC source as Personal Facts; conversation-held-out within this file only. Review labels before estimating accuracy. |
| `progression_routing/career_school_trace_v1.json` | 29 fictional events | Full action, hierarchy, subject and state contract | Constructed example; validates storage, not generalization. |
| `progression_routing/msc_graph_traces_v1.json` | 48 turns in 2 complete MSC train conversations; 25 updates, 23 no-op decisions | Provisional model-labeled subject, root, attach, multiattach, state and phase | Real conversation context, single annotator; replays against source in CI. No child actions occur in these two conversations. Not independently verified or gold. Overlaps the 100 attachment pairs. |
| `progression_routing/msc_graph_traces_v2.json` | 309 turns from 12 complete MSC train conversations; 13 proposed children, 112 default no-update review placeholders | Model-proposed graph labels with four user review corrections; source-backed review CSV generated in CI | Supersedes v1 for review. Eight training-review, two development, two holdout-review conversations. Same source as attachment pairs; do not sum counts. Resolve remaining ambiguities and defaults before training or validation. |
| `../progression_calibration_scenarios.json` | 25 invented scenarios × 4 candidates = 100 pairs | Synthetic attachment calibration | Prototype contract only. |
| `external_personal_facts/train/` | 2,223 published train facts, 1,368 self/long-term/valid candidates | Referent, temporal class, duration, validity, fact paraphrase | MSC-derived, single annotator, no conversation IDs; normalized source training rows checked in, raw Parquet fetched by pinned CI importer. Not independent of MSC pairs. |
| `external_alpsbench/dev_task2_audit_v1.json` | 469 dev update cases | Before/after memory sets and source evidence | Real WildChat-derived dialogue; every case has one removed and two added IDs. Pin and fetch via CI; no hierarchy or representative no-op labels. Test held out. |

**Do not sum these counts into a model training size.** Pairs from one event
and updates from one conversation are correlated; each source labels a
different question. LoCoMo and LongMemEval in `evaluation/data/` are held-out
retrieval/QA benchmarks and must not supply writer training labels.

There are now twelve **provisional model-annotated real full-sequence traces**.
User review identified four corrections; 112 default no-update placeholders,
the remaining child proposals and explicit ambiguous boundaries still need
source-grounded adjudication. Then add more conversations for rare actions.
The reserved
development and holdout-review conversations must stay out of model fitting.
Model labels alone are neither gold evaluation nor evidence of a benchmark gain.
