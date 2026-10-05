# Real conversation development cohort

The official [Multi-Session Chat corpus](https://parl.ai/projects/msc/) provides
independent human dialogue for writer development. The workflow fetches the
v0.1 archive with the checksum pinned in ParlAI's `parlai/tasks/msc/build.py`,
reads only `msc_dialogue/session_{2,3,4}/train.txt`, and exports a deterministic
conversation-level sample as a short-lived CI artifact. LoCoMo and LongMemEval
remain held out. No raw MSC utterances are committed to this repository.

Each turn carries its original speaker slot and source ID. The speaker slot is
not by itself the subject of every claim: one speaker can discuss the other
speaker or a third party. Reviewers must mark the subject or abstain, identify
evidence-backed events, and annotate attachment, phase, hierarchy, and state.
The exported JSON is an **annotation queue**, not a writer trace or gold labels.
The existing benchmark can replay a reviewed writer trace but cannot report a
progression gain based on the queue alone.

Next: audit a small sample for speaker consistency across sessions and
attachment errors; resolve disagreements on subject and state labels; reserve
whole conversations for validation; fit attachment weights only on reviewed
training labels; then compare append RAG with the same encoder, reader and
context budget. Report both source recall and QA on the held-out benchmarks.
