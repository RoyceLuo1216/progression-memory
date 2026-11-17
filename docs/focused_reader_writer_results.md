# Focused research follow-up: stronger reader and real writer

## Frozen questions

1. Does a larger answering model exploit compact temporal memory to improve downstream answer correctness?
2. Can a real model construct V2 state updates as reliably and economically as a topic-wiki writer, without oracle source-fact inputs?

## Experiment A — Qwen3.5-2B reader (completed)

Model: `Qwen/Qwen3.5-2B`, CPU/bfloat16, deterministic `do_sample=False`, thinking disabled, 128 maximum output tokens, same 1,600-character context across arms. Real model inference archived in [Actions run 44005769200](https://github.com/RoyceLuo1216/progression-memory/actions/runs/44005769200).

Same deterministic 12 MEME historical and first 4 cascading items as the 0.8B pilot, using the pinned MEME release. Identical reader system prompt; two contexts constructed using oracle annotated source facts. These are not live writer outputs.

| Task and model | Versioned topic wiki | Compact V2 |
| --- | ---: | ---: |
| Historical chronology, 0.8B (12 items) | 5/12 (41.7%) | 6/12 (50.0%) |
| **Historical chronology, 2B (12 items)** | **8/12 (66.7%)** | **11/12 (91.7%)** |
| Cascading updates, 0.8B (4 items, strict exact) | 1/4 | 2/4 |
| Cascading updates, 2B (4 items, strict exact) | 2/4 | 2/4 |

- For the 12 historical cases, expected values are all included in 10/12 wiki contexts and 12/12 compact V2 contexts.
- Paired 2B V2-minus-wiki chronology accuracy = +25.0 pp, episode-bootstrap 95% CI approx **[-8.3, +58.3] pp** (12 episode clusters, 10,000 paired resamples; model and prompts held fixed). Four discordant cases favor V2, one favors wiki. Not statistically reliable.
- Reader inference: 16 calls each; wiki 7,300 input/450 output tokens and 202.4 CPU generation seconds; compact V2 6,821 input/503 output tokens and 200.2 seconds.
- **Interpretation:** A stronger reader improves use of both representations. V2's smaller, history-aware context often supplies more complete historical evidence and produces more correct sequences; however this cannot establish unique progression semantics because a similarly compact, adaptive wiki might match it. N=12 is exploratory.

## Experiment B — 2B model-written V2 vs model-written wiki

Status: **inference in progress; no result reported yet**. Ten authored cases (six explicit changes, four no-change cases), matched observations, model, decoding budget, and prior subject facts. Contrasts canonical V2 `WriterProposal` against a simpler `WikiProposal`. Measures schema, change detection, anchored grounding, actual `WritePipeline` application, prompt/completion tokens and CPU seconds. [Actions run 44005787589](https://github.com/RoyceLuo1216/progression-memory/actions/runs/44005787589).

Interpretation boundary: the score's salient-anchor check is a simple proxy, not blinded semantic grading; writer outputs must be independently reviewed. The two schemas differ in complexity, and no comprehensive long-term update stream is claimed.

## Supporting diagnostic

0.8B schema-constrained `WriterProposal` decoding, eight previously tested observations. Its purpose is to separate formatting failures from semantic mistakes; do not count a JSON-valid hallucination as a correct write. CI records [run 44007119659](https://github.com/RoyceLuo1216/progression-memory/actions/runs/44007119659).

## Remaining for definitive closeout

- Independently graded answer-level accuracy across held-out multi-user histories, including abstention, deletions and cascading dependencies.
- Paired model-maintained wiki and progression paths with provenance, dated revisions, write/read cost and realistic error-recovery.
- A larger reader and writer sample with clustered uncertainty; the present pilot cannot support broad superiority claims.
