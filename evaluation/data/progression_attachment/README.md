# Real conversation attachment pairs (development only)

`msc_train_pairs_v1.json` contains 25 real dialogue turns × four candidate
progressions = 100 event-candidate pairs. The source is the **train** split of
Meta/ParlAI Multi-Session Chat v0.1 (archive SHA256
`e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`).
Each row contains the verbatim event and the earlier verbatim candidate anchor,
speaker IDs, source IDs, and a provisional attachment label. Candidate identity
and state are concise annotations of the anchor, not dataset fields.

`attach=1` asks whether this event develops the **same ongoing process** as
the candidate, not merely whether the texts share a topic. Some events have
no matching candidate; multi-attachment is possible. These labels were
single-pass annotations and have **not** been independently adjudicated. Do
not call this a gold benchmark or tune on LoCoMo/LongMemEval. Split by
conversation ID for evaluation: four rows from the same event and every event
from the same conversation must stay together.

This small convenience set is strongly biased toward easily identifiable
ongoing processes and should be used for writer inspection, not as evidence
of general accuracy. The 100-pair synthetic prototype remains in
`evaluation/progression_calibration_scenarios.json` for its original contract
test; this file is a separate real dialogue development set.
