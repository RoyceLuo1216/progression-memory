# Career and school routing trace

`career_school_trace_v1.json` is a **fictional, fully labeled** 29-event sequence
built by `build_career_school_trace.py`. It exercises all writer actions,
speaker/subject distinction, parent links, per-target state, and temporal
segments against the current `Memory.apply` contract. It is an executable
example of the intended representation, **not** evidence that a learned
writer can infer these decisions from real conversation.

## Labeling rule

1. Extract a concrete assertion and its actual subject. Questions without
   assertions are not state changes. A third-party fact does not update the
   tracked user's progression. In this trace all source turns have one user
   speaker; the friend's promotion is assigned to the friend.
2. `create_root`: an ongoing process has no defensible existing parent.
   Education and Career begin as separate roots.
3. `create_child`: a distinct, durable process has a defensible narrower
   parent. A particular degree sits under Education; each academic year and
   the research project sit under that degree. An employer and a job search
   are distinct children of Career. A master's application process is a child
   of Education, rather than another year of undergraduate study.
4. `attach`: the same subject and the same process continue. Shared topic
   words alone do not suffice. A new course in year one extends that year;
   a second employer starts a new child.
5. `multi_attach`: one assertion materially changes multiple **existing**
   processes. A return offer updates both employer history and job search.
   Each target gets its own before/after state and phase flag.
6. `no_update`: no durable process changes for the tracked user. Source turns
   remain accessible in the raw event store.
7. `new_segment=true`: a meaningful phase boundary within the *same* process,
   such as study to completion, job search to interview, or internship to
   return offer. It is not another child progression. Routine incremental
   work extends the latest segment.

The year children are a **modeling choice**: they keep year-specific courses
and achievements inspectable. If a year contains only a date boundary and no
independent evolving process, it should instead be a segment of the degree.
This distinction should be tested against real annotated conversations.

The current action schema supports either creation or attachment in one
decision, so a child-creation event does not simultaneously rewrite its
parent's state. Milestone events explicitly update both when both states
change. This is a contract limitation worth reviewing before production.

## What this closes and what remains

The trace supplies example labels for root, child, attach, multi-attach,
no-update, subject, state, and segment boundary. It can validate storage and
serve as a prompt example. It does **not** replace independently reviewed
full-sequence labels on real dialogue, nor does it prove state entailment
automatically. The separate MSC 100-pair file concerns attachment only.

## Provisional real-conversation graph traces

`msc_graph_traces_v1.json` annotates every turn in two **MSC train**
conversations: `ordered_416` and `ordered_4998`. The 48 turns include 8 root
creations, 14 attachments, 3 multiattachments, and 23 no-update decisions.
The offer of a plant job is assigned to its recipient even though the other
speaker makes the offer. Each updating row records a paraphrase of its source
evidence and an explicit new state; a multiattach row gives each target its
own state. The entire dialogue was visible during annotation. Raw MSC text
is not included in the tracked file; the pinned audit artifact supplies it
during CI replay (`MSC_DEV_AUDIT_PATH` for local testing).

These are **single-model silver labels**, with no second annotator or
adjudication. The replay checks chronological completeness, target existence,
subject consistency, and storage invariants; it cannot establish that every
paraphrase is entailed or every no-update is correct. There are no natural
`create_child` cases in these two conversations. Do not pool turns as
independent samples or measure accuracy against these same model labels.
Review source entailment and routing disagreements before using them as
training data, then obtain held-out conversation-level human labels for
evaluation. The existing MSC attachment pairs overlap these conversations.

## Twelve-conversation review set

`msc_graph_traces_v2.json` covers every turn in the pinned 12-conversation
MSC train audit (309 turns, both participants). Its authoring script is
`build_msc_graph_traces_v2.py`. There are 65 proposed root creations, 13
children, 96 attachments, 3 multiattachments, and 132 no-update decisions.
Of the latter, **109 are default no-update placeholders** for turns not
selected as changes. They are explicitly tagged `default_no_update` and
should be checked against the source before inclusion in supervised training.
Four source-grounded corrections proposed during user review are tagged
`user_review_correction`: `ordered_6210:s3:t7` (GreenChef),
`ordered_7807:s2:t10` (next resale sourcing trip),
`ordered_6340:s3:t10` (friend's remote job referral; subject is recipient),
and `ordered_6469:s3:t6` (eating caught fish; lower confidence).
`ordered_416:s3:t2` and `ordered_7807:s3:t4` remain explicit ambiguity cases.
The second source pass also corrected two names that leaked future evidence:
the cat was not named Bella until after its first turn, and the phone app was
not named Lostfound until two turns later. Competitive eating now begins at
its first supporting turn (`ordered_792:s3:t1`); two later responses continue
it, and a dance lesson scheduling offer updates its recipient's dance plan.

The source-backed review script validates chronological completeness, replay,
subject consistency and target existence. In CI it exports a short-lived
`msc-graph-review-queue` CSV with the source text, proposed label, subject,
state, parent/targets, and empty reviewer fields. It is not checked into git.
Review `high` priority rows first: all default no-updates, root/child choices,
multiattachments, and explicit ambiguity cases (193 rows). Then review every remaining state against
its source turn, including cases where a turn contains multiple claims but
the current writer contract permits only one creation action. A source-backed
replay is an integrity check, **not** a semantic correctness score.

Conversation-level partitions are `training_review` (8 conversations),
`development` (2), and `holdout_review` (2). The latter two do not overlap
the earlier 100 MSC attachment pairs. Freeze them out of fitting and prompt
selection. These are model-produced labels across all partitions; only a
separate reviewer can establish a usable validation reference. Twelve
conversations and 13 provisional child examples are a starting annotation
set, not adequate evidence of generalized writer quality. LoCoMo and
LongMemEval remain untouched retrieval/QA benchmarks.

### Remaining coverage gaps

| Split | Conversations | Turns | Root | Child | Attach | Multiattach | No-update |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training review | 8 | 215 | 44 | 7 | 69 | 3 | 92 |
| Development | 2 | 46 | 8 | 2 | 9 | 0 | 27 |
| Holdout review | 2 | 48 | 13 | 4 | 18 | 0 | 13 |

The development and holdout splits have **no multiattachment examples**;
the holdout has only 48 turns. Sample additional complete conversations
containing multi-process changes, subject-directed updates, and ambiguous
root-versus-child boundaries, then reserve whole conversations for evaluation.
The writer contract also cannot create a new progression and attach an
existing one in one action; `ordered_416:s3:t2` exposes that missing action.
Do not turn the 109 default no-updates into trusted negative training rows
just because graph replay succeeds. After review, run a held-out writer
evaluation by action and process quality before comparing retrieval/QA to RAG.

The second-pass graph audit found **28 of 78 progressions have only one
supporting turn**, including **24 of 65 roots**. Some may be real processes
whose later events fall outside this short sample. Others appear to be static
facts or one-off activities, such as a cat allergy, church attendance, or a
single concert invitation. Treat these as explicit `progression` versus
`raw fact only` review candidates. Without that boundary, a writer trained on
this set could over-create shallow roots, which is the construction failure
the progression intervention is supposed to avoid. The review CSV includes
`final_process_events` on creation rows to identify these cases.
