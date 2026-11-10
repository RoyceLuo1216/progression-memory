# MEME large-scale memory evaluation (2026-11-09)

## Data and provenance
- Released MEME no-filler dataset: `meme-benchmark/MEME`, revision `e14c4e685772014deb96f91dc7eddfbeb21d90d8`.
- 100 episodes (50 personal life, 50 software projects), 1,188 before/after questions.
- 1,075 questions had at least one annotated ground-truth source turn and were eligible for the strict source-retrieval test. Exactly 2,171 / 2,334 fact values were found verbatim in their annotated turns (~93% alignment); treat evidence metrics with this qualification.
- 725 positive-value questions (Cas, Tr, Agg, ER) evaluated for gold-value visibility, not response correctness; Abs/Del excluded because retrieving obsolete values can be misleading.
- No user test/answer gold is indexed as raw RAG evidence. Oracle topic wiki and V2 share source *fact annotations* as idealized writer updates; they are NOT real model-written memory. Gold question values are reserved for scoring.

## 1. Real conversation source evidence retrieval

Mean fraction of required annotated source turns visible in the chosen context.

| Budget (characters) | Raw BM25 | Newest-first | BM25 + recency RRF |
|---:|---:|---:|---:|
| 400 | 38.4% | 2.0% | 22.3% |
| 800 | 49.5% | 8.1% | 35.6% |
| 1600 | **57.9%** | 19.0% | 46.1% |
| 3200 | 70.0% | 35.5% | 60.5% |
| 6400 | 80.0% | 52.5% | 74.6% |

At 1600 characters, recency minus BM25 = approximately -38.8 percentage points; 95% episode-clustered paired bootstrap interval -42.4 to -35.1 points. RRF here weights BM25 and recency equally; its worse score is a diagnostic, not proof all recency-aware retrieval is inferior.

## 2. Gold-value accessibility: maintained wiki vs actual V2 state engine, with oracle writes

The same annotated facts were converted into topic-wiki entries and V2 writer proposals. Production V2 StateEngine and read_path were used without modification.

| Budget (characters) | Raw BM25 | Current wiki | Versioned wiki | V2 current-only | V2 + history |
|---:|---:|---:|---:|---:|---:|
| 400 | 32.0% | 44.0% | 33.7% | 36.2% | 34.7% |
| 800 | 42.5% | 49.8% | 51.3% | 38.2% | 44.4% |
| 1600 | 46.7% | 54.8% | **58.4%** | 45.1% | 49.6% |
| 3200 | 51.8% | 54.8% | **63.7%** | 47.9% | 56.8% |
| 6400 | 54.9% | 54.8% | **63.7%** | 47.9% | 56.8% |

At 1600 characters, V2 with history minus versioned wiki = -8.78 pp, 95% persona/episode-clustered bootstrap CI approximately [-10.51, -6.90] pp (2000 replicates; 100 episode clusters).

### Task ablation at 1600 characters
- Historical sequences (Tr, 100 questions): current-only V2 **33.0%**, V2 with history **96.0%**, current wiki **33.0%**, versioned wiki **82.0%**.
- At historical tasks V2-history minus versioned wiki **+14.0 pp**, approximate 95% episode-clustered CI [+8.0, +21.0] pp.
- Multi-topic aggregation (Agg, 100 questions): V2 history **51.7%** vs versioned wiki **78.3%**.
- Cascading changes (Cas, 325 questions): V2 history **49.8%** vs wiki history **50.2%**.
- Exact/verbatim retrieval (ER, 200 questions): V2 history **25.0%** vs wiki history **50.0%**.

**Interpretation:** Explicitly storing history improves access to historical facts; this is true for both V2 and the wiki. V2 exposes historical sequences particularly efficiently but remains worse overall, notably on aggregation and verbatim facts. These results measure accessibility of known gold values, not whether a planner generates correct answers or infers changed dependencies.

## 3. Real Qwen3.5-0.8B writer pilot

Eight authored first-person/filler/no-change observations, two prompts each (16 real CPU generations).

| Prompt | Schema-valid writer proposals | Correct change/no-change classification |
|---|---:|---:|
| Production ModelWriter prompt | 0/8 | 0/8 |
| Compact JSON-schema example prompt | 3/8 | 3/8 |

Compact prompting helped on a location constraint and two no-change cases but still failed on other positive observations and hallucinated a friend's hypothetical interest. Neither prompt is yet a reliable production writer. This diagnostic is too small for reliable rate estimates and does not test coherent multi-event progression attachment.

## Reproduction

- [MEME evidence CI run](https://github.com/RoyceLuo1216/progression-memory/actions/runs/42756063676)
- [MEME V2/current/history/wiki CI run](https://github.com/RoyceLuo1216/progression-memory/actions/runs/42756865926)
- [Qwen writer CI run](https://github.com/RoyceLuo1216/progression-memory/actions/runs/42756545792)

All runs archive raw JSON with per-question rows; commit history and released data revision are pinned.

## Inference boundary and next experiments

This is the first independent, 100-episode evidence for which memory **read representations** expose useful facts under constrained budgets. It does not establish a production V2 advantage. Next: schema-constrained/larger writer on held-out histories; actual model-generated answers on MEME absence/deletion and dependency tasks; model-backed wiki summaries (instead of annotation oracle) and equivalent write/read cost accounting; user-disjoint validation on HorizonBench, PersonaMem and LifeBench.
