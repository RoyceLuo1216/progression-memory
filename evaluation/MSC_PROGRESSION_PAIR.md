# Paired MSC progression continuation experiment

The held-out MSC graph conversations `ordered_909` and `ordered_7748` provide
11 events that attach to at least one already existing progression. At each
event, use its raw future utterance as the query and index **only earlier
utterances**. The relevant evidence is the latest earlier source event in each
annotated target progression. This is an online continuation retrieval task,
not a question-answering score. The graph labels are provisional silver labels.

Run three representations over exactly the same preceding event store:

- `append`: each utterance indexed independently (naive dense RAG).
- `structure`: annotated progression segments with identity/scope and source text.
- `progression`: structure plus annotated current state and change text.

All three use the pinned MiniLM encoder, cosine ranking, retrieval depths 1,
3 and 5, and a 4,800-character source-plus-summary budget. The depth sweep is
reported in full; none is chosen after seeing the result. Credit is granted
only when the original relevant source event fits in the delivered context.
The output contains paired per-case recall and complete-evidence rates, and
records actual context characters so grouping costs are visible.

This measures the **representation given supplied writer decisions**. It does
not evaluate automatic writer accuracy, establish a LoCoMo or LongMemEval QA
gain, or replace an independently reviewed graph benchmark. The continuation
label and grouping both originate from the same provisional annotations, so
an independent QA test remains necessary before a general performance claim.

CI: `msc-progression-pair.yml` downloads checksum-verified official MSC train,
rebuilds the source audit, replays the stored held-out labels, and uploads the
paired JSON result. The source dialogue is not committed to the repository.

## First pinned dense run

[CI run 37411886705](https://github.com/RoyceLuo1216/progression-memory/actions/runs/37411886705)
passed. Mean latest-evidence recall across the 11 paired cases:

| Top records | Append | Structure | Progression |
| ---: | ---: | ---: | ---: |
| 1 | 0.000 | 0.091 | 0.091 |
| 3 | 0.364 | 0.636 | 0.636 |
| 5 | 0.591 | 0.758 | 0.758 |

At depth 5, complete-evidence rates are 0.545, 0.727 and 0.727 respectively.
The average delivered context is 501, 619 and 952 characters. The structure
and full progression variants tie on retrieval, so this pilot attributes the
gain over append to **grouping**, not to the added state text. The data are too
small and the relevant process is defined by the same silver graph labels used
to build the representation; this is not an independently adjudicated result.
