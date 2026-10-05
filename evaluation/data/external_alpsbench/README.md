# AlpsBench development update labels

The official [AlpsBench](https://github.com/ThisIsCosine/AlpsBench) development
`task2` split contains real human–LLM dialogue with before/after memory
references. CI fetches only `dataset/dev/task2` from pinned
Hugging Face revision `238961f49e4fe6de8b09a4fefd77d76d759d620f` and
audits schema, row count and checksums. The test split stays untouched. Source
data are not copied into this repository.

There are **469 dev cases**, each from a distinct session. Comparing memory
IDs in the input and reference reveals the *same pattern for every case*:
one old memory ID removed, two new IDs added, and 807 other memory entries
retained across all cases. There are zero same-ID value changes. The committed
`dev_task2_audit_v1.json` stores the exact source checksums and distributions.
This is an ID-level diff; matching a new item to the removed item's *meaning*
would require further alignment or review.

This source can teach what a particular before/after memory update looks like,
with evidence pointers. Its fixed construction **cannot teach the prevalence
or choice of no-update versus add versus modify**. It also cannot directly
teach `create_child` versus `create_root`, progression identity, or phase
boundaries. Source labels must be mapped to our ontology explicitly and
reviewed before training a full writer. The source dataset is a benchmark;
using its dev labels for development means its dev metrics cannot be claimed
as an independent evaluation of the resulting writer.
