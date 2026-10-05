# AlpsBench development update labels

The official [AlpsBench](https://github.com/ThisIsCosine/AlpsBench) development
`task2` split contains real human–LLM dialogue with memory-retention, addition,
and modification references. CI fetches only `dataset/dev/task2` from pinned
Hugging Face revision `238961f49e4fe6de8b09a4fefd77d76d759d620f` and
audits schema, row count and checksums. The test split stays untouched. Source
data are not copied into this repository.

These operations can teach no-update versus add versus modify. They cannot
directly teach `create_child` versus `create_root`, progression identity, or
phase boundaries. Source labels must be mapped to our ontology explicitly and
reviewed before training a full writer. The source dataset is a benchmark;
using its dev labels for development means its dev metrics cannot be claimed
as an independent evaluation of the resulting writer.
