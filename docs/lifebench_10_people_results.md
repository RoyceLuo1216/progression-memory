# LifeBench v2: year-long personal memory retrieval baseline

## Dataset and protocol

Pinned Hugging Face release `C1754955896/Lifebenchv2.0` at
`3d06052d5ef2d94d9042bd2845787bfd37ba925d`.

- 10 fictional user histories, 57,486 daily events.
- 3,380 official QA checkpoints; 2,794 have required daily-event evidence IDs;
  2,783 are eligible at the query's chronological cutoff.
- 13 required event references were unavailable in the selected daily-event
  sources at their query times; those questions are excluded, rather than
  counted as retrieval misses.
- Retrieval indexes daily event `name`, `description`, `type`, and event time;
  uses Chinese character 2/3-grams plus ASCII terms.
- Query-time event subset and BM25 IDF are both as-of filtered, so future events
  cannot affect answers or query-time ranking statistics.
- Whole-document character budgets, strict stop-before-overflow; up to 60 ranked
  daily events per question. This is **evidence recall**, not generated answer accuracy.
- Phone records, photo, SMS and agent chat are not indexed; a full multimodal
  LifeBench evaluation would require those additional sources.

## Results

| Character budget | CJK BM25: source recall | Recency: source recall | BM25 all-evidence rate |
|---:|---:|---:|---:|
| 800 | 48.2% | 0.1% | 35.8% |
| 1,600 | **54.8%** | 0.4% | 42.0% |
| 3,200 | 59.8% | 0.5% | 46.9% |
| 6,400 | 64.7% | 1.5% | 51.5% |
| 12,800 | 66.3% | 3.8% | 52.8% |

At 1,600 characters, Chinese-aware BM25 minus recency = **+54.36 pp**,
95% **person-clustered** paired bootstrap interval approximately **[+52.63, +56.28] pp**,
resampling 10 persons (2,000 replicates, seed 42).

## Interpretation

The previous synthetic benchmark had the latest authoritative update repeat
the whole user state, favoring a recency baseline. Here thousands of intervening
events make recency uncompetitive for historical evidence retrieval. Search
dramatically improves access, but this does not show that V2 progression state
outperforms either search or a maintained topic wiki.

These results must not be described as a full LifeBench answer-generation
benchmark or evidence for real writer quality.

## Reproducibility

- [Successful full 10-person run](https://github.com/RoyceLuo1216/progression-memory/actions/runs/42757577116)
- [Dataset revision and source inspection](https://github.com/RoyceLuo1216/progression-memory/actions/runs/42757296050)
- Script: `src/progression_memory/benchmarks/lifebench_evidence_retrieval.py`
- Output artifact: `lifebench_all_users.json`, containing all per-question scores.
