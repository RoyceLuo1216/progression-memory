from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit, _segment_indexes
from progression_memory.evaluation.datasets import load_locomo
from progression_memory.evaluation.hierarchical_retrieval import _beam_traverse, build_progression_hierarchy
from progression_memory.evaluation.hierarchical_retrieval_v2 import _soft_stage1
from progression_memory.evaluation.hierarchical_retrieval_v3 import (
    DEFAULT_BRANCH_BUDGET_FRACTION,
    DEFAULT_GLOBAL_RESCUE_FRACTION,
    DEFAULT_TEMPORAL_BUDGET_FRACTION,
    _beam_traverse_semantic,
    build_semantic_progression_hierarchy,
)
from progression_memory.evaluation.hierarchical_retrieval import _query_is_temporal
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.evaluation.semantic import HybridIndex, SentenceTransformerEmbedder


def _evidence(document: RetrievalDocument) -> set[str]:
    return set(document.evidence_ids or {document.id})


def _segment_evidence(segment: RetrievalDocument) -> set[str]:
    evidence: set[str] = set()
    for source in segment.metadata.get("source_documents", ()):
        if isinstance(source, RetrievalDocument):
            evidence.update(_evidence(source))
    return evidence


def _gold_progressions_by_item(by_progression: dict[str, list[RetrievalDocument]], gold: set[str]) -> dict[str, set[str]]:
    result = {evidence_id: set() for evidence_id in gold}
    for pid, progression_segments in by_progression.items():
        for segment in progression_segments:
            for evidence_id in _segment_evidence(segment) & gold:
                result[evidence_id].add(pid)
    return result


def _select_v3(query, segments, by_progression, nodes, roots, embedder, *, semantic_weight: float):
    routed, visited = _beam_traverse_semantic(query.text, nodes, roots, embedder, beam_width=4)
    candidate_limit = _adaptive_candidate_limit(len(segments))
    global_rank = HybridIndex(segments, embedder, semantic_weight=semantic_weight).score(query.text)
    rescue_count = min(len(global_rank), max(6, math.ceil(DEFAULT_GLOBAL_RESCUE_FRACTION * candidate_limit)))
    rescued = {str(doc.metadata.get("progression_id")) for _, doc in global_rank[:rescue_count]}
    active = routed | rescued or set(by_progression)
    branch_pool = tuple(doc for pid in active for doc in by_progression.get(pid, []))
    branch_rank = HybridIndex(branch_pool, embedder, semantic_weight=semantic_weight).score(query.text) if branch_pool else []

    selected: list[RetrievalDocument] = []
    selected_ids: set[str] = set()
    temporal_budget = min(candidate_limit, max(1, math.ceil(DEFAULT_TEMPORAL_BUDGET_FRACTION * candidate_limit))) if _query_is_temporal(query.text) else 0
    for _, anchor in branch_rank:
        if len(selected) >= temporal_budget:
            break
        pid = str(anchor.metadata.get("progression_id"))
        siblings = sorted(by_progression.get(pid, []), key=lambda d: (d.timestamp or "", d.id))
        idx = next((i for i, d in enumerate(siblings) if d.id == anchor.id), 0)
        for neighbor in siblings[max(0, idx - 1):min(len(siblings), idx + 2)]:
            if neighbor.id not in selected_ids:
                selected.append(neighbor); selected_ids.add(neighbor.id)
            if len(selected) >= temporal_budget:
                break
    branch_target = min(candidate_limit, max(len(selected), math.ceil(DEFAULT_BRANCH_BUDGET_FRACTION * candidate_limit)))
    for _, doc in branch_rank:
        if len(selected) >= branch_target:
            break
        if doc.id not in selected_ids:
            selected.append(doc); selected_ids.add(doc.id)
    for _, doc in global_rank:
        if len(selected) >= candidate_limit:
            break
        if doc.id not in selected_ids:
            selected.append(doc); selected_ids.add(doc.id)
    return selected, routed, rescued, active, visited


def analyze(data_path: str, *, system: str = "hierarchical_beam_v3", top_k: int = 8, sample_size: int | None = None, semantic_weight: float = 0.7) -> dict[str, object]:
    if system not in {"hierarchical_beam_v2", "hierarchical_beam_v3"}:
        raise ValueError(f"Unsupported system: {system}")
    queries = load_locomo(data_path)
    if sample_size is not None:
        queries = queries[:sample_size]

    embedder = SentenceTransformerEmbedder() if system == "hierarchical_beam_v3" else None
    cache: dict[int, tuple] = {}
    failures: Counter[str] = Counter()
    examples: dict[str, list[dict[str, object]]] = {}
    total_gold = 0
    final_hits = 0
    stage_hits = Counter()
    visited_counts: list[int] = []

    for query in queries:
        key = id(query.documents)
        if key not in cache:
            segments, by_progression, construction_stats = _segment_indexes(query.documents, embedder=embedder)
            if system == "hierarchical_beam_v3":
                nodes, roots, depth = build_semantic_progression_hierarchy(by_progression, embedder)
            else:
                nodes, roots, depth = build_progression_hierarchy(by_progression)
            cache[key] = (segments, by_progression, nodes, roots, depth, construction_stats)
        segments, by_progression, nodes, roots, _, _ = cache[key]
        gold = set(query.evidence_ids)
        total_gold += len(gold)
        gold_pids = _gold_progressions_by_item(by_progression, gold)
        represented = {eid for eid, pids in gold_pids.items() if pids}

        if system == "hierarchical_beam_v3":
            selected, routed, rescued, active, visited = _select_v3(
                query, segments, by_progression, nodes, roots, embedder, semantic_weight=semantic_weight
            )
        else:
            routed, visited = _beam_traverse(query.text, nodes, roots)
            selected, active, _ = _soft_stage1(
                query.text, segments, by_progression, routed,
                candidate_limit=_adaptive_candidate_limit(len(segments)),
            )
            rescued = active - routed
        visited_counts.append(visited)

        selected_evidence: set[str] = set()
        sources: dict[str, RetrievalDocument] = {}
        for segment in selected:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    sources[source.id] = source
                    selected_evidence.update(_evidence(source))

        candidate_documents = tuple(sources.values()) or query.documents
        if system == "hierarchical_beam_v3":
            final_rank = [doc for _, doc in HybridIndex(candidate_documents, embedder, semantic_weight=semantic_weight).score(query.text)]
        else:
            final_rank = [doc for _, doc in BM25Index(candidate_documents).score(query.text)]
        final_evidence: set[str] = set()
        for doc in final_rank[:top_k]:
            final_evidence.update(_evidence(doc))
        final_hits += len(final_evidence & gold)

        for evidence_id in gold:
            pids = gold_pids[evidence_id]
            if evidence_id not in represented:
                stage = "progression_construction"
            elif not (pids & routed):
                if pids & rescued:
                    stage_hits["rescued_gold_items"] += 1
                    if evidence_id in selected_evidence:
                        stage = "final_reranking_or_topk" if evidence_id not in final_evidence else "recovered"
                    else:
                        stage = "stage1_segment_selection_after_rescue"
                else:
                    stage = "hierarchy_and_rescue_miss"
            elif evidence_id not in selected_evidence:
                stage = "stage1_segment_selection"
            elif evidence_id not in final_evidence:
                stage = "final_reranking_or_topk"
            else:
                stage = "recovered"
            failures[stage] += 1
            if evidence_id in represented:
                stage_hits["represented"] += 1
            if pids & routed:
                stage_hits["hierarchy_reached"] += 1
            if pids & active:
                stage_hits["active_after_rescue"] += 1
            if evidence_id in selected_evidence:
                stage_hits["stage1_selected"] += 1
            if evidence_id in final_evidence:
                stage_hits["final_topk"] += 1
            if stage != "recovered" and len(examples.setdefault(stage, [])) < 20:
                examples[stage].append({
                    "query_id": query.id,
                    "question": query.text,
                    "evidence_id": evidence_id,
                    "gold_progressions": sorted(pids),
                    "routed_progressions": sorted(routed),
                    "rescued_progressions": sorted(rescued),
                    "active_progressions": sorted(active),
                })

    denominator = max(1, total_gold)
    stage_recall = {name: count / denominator for name, count in stage_hits.items() if name != "rescued_gold_items"}
    stage_recall["final_topk"] = final_hits / denominator
    corpora = list(cache.values())
    return {
        "dataset": "locomo",
        "system": system,
        "top_k": top_k,
        "queries": len(queries),
        "gold_evidence_items": total_gold,
        "gold_item_recall_at_k": final_hits / denominator,
        "stage_item_recall": stage_recall,
        "rescued_gold_items": stage_hits["rescued_gold_items"],
        "failure_counts": dict(failures),
        "failure_fractions": {name: count / denominator for name, count in failures.items()},
        "oracle_ceiling_if_stage_fixed": {
            name: min(1.0, (final_hits + count) / denominator)
            for name, count in failures.items() if name != "recovered"
        },
        "avg_nodes_visited": sum(visited_counts) / max(1, len(visited_counts)),
        "avg_progressions_per_corpus": sum(len(row[1]) for row in corpora) / max(1, len(corpora)),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--system", choices=("hierarchical_beam_v2", "hierarchical_beam_v3"), default="hierarchical_beam_v3")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.data_path, system=args.system, top_k=args.top_k, sample_size=args.sample_size, semantic_weight=args.semantic_weight)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")


if __name__ == "__main__":
    main()
