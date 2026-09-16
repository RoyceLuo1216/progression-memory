from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit, _segment_indexes
from progression_memory.evaluation.datasets import load_locomo
from progression_memory.evaluation.hierarchical_retrieval import _beam_traverse, build_progression_hierarchy
from progression_memory.evaluation.hierarchical_retrieval_v2 import _soft_stage1
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument


def _evidence(document: RetrievalDocument) -> set[str]:
    return set(document.evidence_ids or {document.id})


def analyze(data_path: str, *, top_k: int = 8, sample_size: int | None = None) -> dict[str, object]:
    queries, = (load_locomo(data_path),)
    if sample_size is not None:
        queries = queries[:sample_size]
    cache: dict[int, tuple] = {}
    failures: Counter[str] = Counter()
    examples: dict[str, list[dict[str, object]]] = {}
    total_gold = 0
    final_hits = 0

    for query in queries:
        key = id(query.documents)
        if key not in cache:
            segments, by_progression, _ = _segment_indexes(query.documents)
            nodes, roots, _ = build_progression_hierarchy(by_progression)
            cache[key] = (segments, by_progression, nodes, roots)
        segments, by_progression, nodes, roots = cache[key]
        gold = set(query.evidence_ids)
        total_gold += len(gold)

        gold_progressions: set[str] = set()
        represented: set[str] = set()
        for pid, progression_segments in by_progression.items():
            for segment in progression_segments:
                segment_evidence: set[str] = set()
                for source in segment.metadata.get("source_documents", ()):
                    if isinstance(source, RetrievalDocument):
                        segment_evidence.update(_evidence(source))
                if segment_evidence & gold:
                    gold_progressions.add(pid)
                    represented.update(segment_evidence & gold)

        routed, _ = _beam_traverse(query.text, nodes, roots)
        selected, active, _ = _soft_stage1(
            query.text, segments, by_progression, routed,
            candidate_limit=_adaptive_candidate_limit(len(segments)),
        )
        selected_evidence: set[str] = set()
        sources: dict[str, RetrievalDocument] = {}
        for segment in selected:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    sources[source.id] = source
                    selected_evidence.update(_evidence(source))
        final_rank = [doc for _, doc in BM25Index(tuple(sources.values()) or query.documents).score(query.text)]
        final_evidence: set[str] = set()
        for doc in final_rank[:top_k]:
            final_evidence.update(_evidence(doc))
        final_hits += len(final_evidence & gold)

        for evidence_id in gold:
            if evidence_id not in represented:
                stage = "progression_construction"
            elif not (gold_progressions & routed):
                if gold_progressions & active:
                    stage = "hierarchy_miss_rescued"
                else:
                    stage = "hierarchy_and_rescue_miss"
            elif evidence_id not in selected_evidence:
                stage = "stage1_segment_selection"
            elif evidence_id not in final_evidence:
                stage = "final_reranking_or_topk"
            else:
                stage = "recovered"
            failures[stage] += 1
            if stage != "recovered" and len(examples.setdefault(stage, [])) < 20:
                examples[stage].append({
                    "query_id": query.id, "question": query.text, "evidence_id": evidence_id,
                    "gold_progressions": sorted(gold_progressions), "routed_progressions": sorted(routed),
                    "active_progressions": sorted(active),
                })

    recoverable = max(1, total_gold)
    return {
        "dataset": "locomo", "system": "hierarchical_beam_v2", "top_k": top_k,
        "queries": len(queries), "gold_evidence_items": total_gold,
        "gold_item_recall_at_k": final_hits / recoverable,
        "failure_counts": dict(failures),
        "failure_fractions": {name: count / recoverable for name, count in failures.items()},
        "oracle_ceiling_if_stage_fixed": {
            name: min(1.0, (final_hits + count) / recoverable)
            for name, count in failures.items() if name != "recovered"
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.data_path, top_k=args.top_k, sample_size=args.sample_size)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(payload + "\n")


if __name__ == "__main__":
    main()
