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
