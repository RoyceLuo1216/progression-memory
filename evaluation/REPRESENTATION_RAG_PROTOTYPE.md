# Progression representation and RAG prototype

## Question and controlled experiment

The proposed intervention changes stored memory records, not the query model.
`representation_rag.retrieve` uses the same encoder, cosine ranking, explicit
search depth, source event store, and context character budget for `append`,
`structure`, and `progression`. All modes deliver original events to the reader.
The full progression mode additionally delivers identity/state/change text when
it fits; this text is charged to the same budget. An indexed progression record
does not get credit for an event whose original text was not delivered.

`structure` indexes progression identity, scope, and ordered raw events;
`progression` also indexes each segment's current state and change description.
The reader and official benchmark evaluator remain to be integrated. The tiny
encoder used in the smoke test is not a benchmark encoder.

## Write contract

`calibrated_progression.Memory` stores each source event once. Progressions
reference ordered segments; segments reference event IDs. A progression has
identity, scope, parents, children, and evidence-backed current state. Decisions
are `NO_UPDATE`, `ATTACH`, `CREATE_ROOT`, `CREATE_CHILD`, `MULTI_ATTACH`.
`ScriptedWriter` exercises these actions without pretending a similarity model
can infer phase changes or hierarchy. Each update supplies explicit state and
whether a new phase starts. The memory checks source IDs, subject isolation,
unique identity, target existence, and duplicate targets before mutation.

The mock does not yet validate the truth of generated state text against source
text, decide when to create children, or extract real events from conversations.
Those are necessary writer tasks before a live LoCoMo/LongMemEval comparison.

## Attachment calibration

`progression_calibration_scenarios.json` contains 25 hand-authored event
scenarios. Each has four candidate process labels, making exactly 100 labeled
event/candidate pairs. Some events intentionally have two valid attachments.
These examples are invented and are **not** benchmark labels or evidence that
the routing model generalizes to real dialogue.

The exploratory model fits ordinary least squares to binary attachment labels.
It uses cosine and token-overlap signals for identity and state, plus subject
match and log time gap. There are no manually selected score weights. The
attachment cutoff maximizes balanced accuracy over leave-one-scenario-out
scores; the model is then refitted on the full synthetic set. Scores are linear
regression outputs, not calibrated probabilities. Selecting the cutoff using
these same out-of-fold labels makes the reported accuracy optimistic; reserve
untouched real examples for final evaluation. All examples have one subject and
no timestamps, so the subject/time coefficients are unidentifiable and zero in
the current fit. No downstream hierarchy or phase decision is learned here.

Run the prototype with:

```bash
PYTHONPATH=src python -m progression_memory.benchmarks.calibrate_progression \
  --output evaluation/results/progression_attachment_synthetic_v1.json
python -m pytest -q tests/test_calibrated_progression.py
```

Observed synthetic leave-one-scenario-out fit: 100 pairs, 25 scenarios,
balanced accuracy 0.9683, two false attachments and one missed attachment.
This is an interface check, not a result against baseline RAG.

## Next implementation gates

1. Curate independently reviewed, subject-grounded real event/candidate pairs
   and phase/parent decisions. Split by conversation/person before calibration.
2. Replace scripted updates with a model writer constrained to the same schema;
   validate state assertions and transitions against source pointers.
3. Pin a real dense encoder and reader, choose equal search/context budgets
   before test evaluation, and run paired retrieval/QA on identical questions.
4. Report process coherence, false links, missed links, evidence actually seen
   by the reader, answer quality by question type, and write/search cost.
