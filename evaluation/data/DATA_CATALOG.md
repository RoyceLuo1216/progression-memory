# Progression writer development data

| Source | Count | Supervision | Independence and use |
|---|---:|---|---|
| `progression_attachment/msc_train_pairs_v1.json` | 100 pairs from 25 events, 6 MSC conversations; 33 positive | Provisional event-to-candidate attachment | Same underlying MSC source as Personal Facts; conversation-held-out within this file only. Review labels before estimating accuracy. |
| `progression_routing/career_school_trace_v1.json` | 29 fictional events | Full action, hierarchy, subject and state contract | Constructed example; validates storage, not generalization. |
| `../progression_calibration_scenarios.json` | 25 invented scenarios × 4 candidates = 100 pairs | Synthetic attachment calibration | Prototype contract only. |
| `external_personal_facts/train_audit_v1.json` | 2,223 published train facts, 1,368 self/long-term/valid candidates | Referent, temporal class, duration, validity, fact paraphrase | MSC-derived, single annotator, no conversation IDs; raw data fetched by pinned CI importer. Not independent of MSC pairs. |
| `external_alpsbench/dev_task2_audit_v1.json` | 469 dev update cases | Before/after memory sets and source evidence | Real WildChat-derived dialogue; every case has one removed and two added IDs. Pin and fetch via CI; no hierarchy or representative no-op labels. Test held out. |

**Do not sum these counts into a model training size.** Pairs from one event
and updates from one conversation are correlated; each source labels a
different question. LoCoMo and LongMemEval in `evaluation/data/` are held-out
retrieval/QA benchmarks and must not supply writer training labels.

There is still no independently reviewed **real full-sequence progression
trace** with root/child/attach/multiattach/no-update, phase and grounded state
labels. The next dataset milestone is a conversation-level reviewed set with
disagreements recorded. Sample additional conversations for rare actions,
then reserve whole people/conversations for validation. Synthetic or model
generated labels can seed review, never serve as independent gold results.
