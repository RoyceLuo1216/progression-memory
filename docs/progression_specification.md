# Progression memory: implementation contract

## Purpose and controlled comparison

A progression represents an **evolving process about one person**, such as a
degree, job search, employment at a particular company, or recovery goal.
Progressions change **how evidence is written and represented**. The dense
encoder, candidate retrieval, reranker, answer reader, source corpus, and
retrieval context budget must be shared with the append-only RAG control.
Store every source turn in both arms. A turn that does not update a progression
is still retrievable from the raw event store.

An isolated fact or temporary outing need not become a progression. Create
one when a process has an identifiable subject, identity, and scope that can
receive meaningful later updates. One observation can start a genuine process;
its later absence in a short sample is not proof the creation was wrong.

## Data model

- `Event`: unique source ID, original text, observed timestamp or source
  order, speaker, attributed subject, and provenance. Speaker and subject can
  differ for offers or statements directed at another participant. Abstain
  on ambiguous subject attribution instead of silently assigning the speaker.
- `Progression`: stable ID, one subject, identity, scope, zero or more parents,
  children, current grounded state, state embedding, and ordered segment IDs.
  A root has no parent; a child has a narrower evolving scope under an
  existing same-subject parent. A parent is not automatically rewritten when
  a child is created.
- `Segment`: progression ID, ordered event IDs, before/after state, change
  description, and source provenance. A new segment marks a meaningful phase
  transition within the same process. It is not a substitute for a distinct
  child process.
- `State`: concise current facts and unresolved goals, with contributing
  source IDs. The embedding is an index over this state, never the source of
  truth. Raw evidence and earlier segments remain immutable and queryable.

The current prototype implements events, progressions, segments and textual
state in `calibrated_progression.py`. It has no persisted state embedding or
independently verified state entailment check yet.

## Per-turn writer contract

1. Read the next original turn in order. Resolve its factual subject(s) and
   extract source-grounded assertions. Preserve the raw turn regardless of
   subsequent routing.
2. Retrieve candidate progressions for the attributed subject using the same
   encoder as the RAG experiment. Candidate scores are proposals; do not
   attach solely because two texts mention the same topic.
3. Decide whether any ongoing process changes. If not, emit `no_update` for
   the progression graph. A stable standalone fact can remain only in raw RAG.
4. For a changed process, distinguish:
   - `attach`: same subject and same evolving process;
   - `create_root`: a new process without an existing defensible parent;
   - `create_child`: a new durable narrower process under a same-subject parent;
   - `multi_attach`: material changes to multiple *existing* processes, with
     separate proposed states per target;
   - `create_and_attach`: one new root or child **and** changes to one or more
     existing processes from the same source turn. Its `state` describes the
     new process; `target_updates` gives each existing process its own state,
     change and segment boundary. Optional `parent_id` selects child creation.
5. Produce a grounded next state and concise change for each target. Mark a
   phase boundary only when the process meaningfully changes stage. Keep
   uncertainty in the state; do not turn a plan into an accomplished outcome.
6. Validate all IDs, subjects, parent links, distinct targets, states and
   evidence before mutation. Commit the event once and all effects atomically.
   A failed validation leaves the event, graph and segments untouched.

`create_and_attach` currently allows **one** new progression plus any number
of existing targets. A single turn that starts two independent new processes
still needs event decomposition or a general multi-operation transaction.
The `ordered_909:s2:t8` review flag illustrates this remaining limit.

## Example transaction

Existing `progression:1` is a person's decision about getting a pet snake.
One turn reports the death of their husband and reconsiders that decision:

```json
{
  "action": "create_and_attach",
  "identity": "Family bereavement",
  "scope": "Managing husband's death and its effect on family life",
  "state": "Husband has died; grieving and reconsidering wishes he expressed",
  "parent_id": null,
  "target_ids": ["progression:1"],
  "target_updates": [
    {
      "progression_id": "progression:1",
      "state": "After husband's death, reconsidering a pet snake",
      "change": "Reconsidering previous pet preference",
      "new_segment": true
    }
  ],
  "reason": "The current turn supplies both changes"
}
```

Both progressions cite the **same original event ID**. A later family-dinner
turn attaches to the existing bereavement process rather than starting it
late. This is implemented in the MSC review trace.

## Learning and validation

- Split by complete conversation (and person where identifiable), before
  fitting or selecting prompts. Do not count turns or four candidate pairs
  from one event as independent validation examples.
- Keep the synthetic career trace for interface tests. Use reviewed real
  conversations for training, and separate reviewed conversations for
  development and held-out writer evaluation. LoCoMo and LongMemEval remain
  held-out retrieval/QA tests, not writer training labels.
- Evaluate subject accuracy, no-update precision/recall, action confusion,
  attachment to the *correct* process, hierarchy correctness, phase decisions,
  state entailment, and graph contamination over complete sequences. Report
  per-action counts and uncertainty; rare actions must appear in evaluation.
- Learn attachment weights and decision boundaries on reviewed training data.
  Tune model choices on development only. Do not set unexplained similarity
  thresholds or promote model-authored silver labels to independent gold.
- Then replay the writer on unseen corpora and compare append RAG against
  progression representation under the same encoder, search depth, delivered
  source budget and reader. Report source recall and QA by question type,
  alongside write cost, token use and graph coherence.

## Current data status and known limitations

The MSC silver pool contains 15 complete train conversations, 393 turns.
Eight conversations (215 turns) are assigned to training review; three
(82 turns) to development; four (96 turns) to holdout review. The added
conversations put two multiattachments in development and two in holdout;
all labels remain model-authored until source-grounded independent review.
The baseline 12-conversation set still contains many default no-update
placeholders and singleton roots that need process-versus-fact adjudication.
The prototype writer also operates at one turn per event and cannot yet
decompose a turn into multiple independently attributed assertions.
