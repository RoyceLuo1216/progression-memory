from __future__ import annotations

import math
import numpy as np

from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit, _segment_indexes
from progression_memory.evaluation.hierarchical_retrieval_v3 import (
    DEFAULT_GLOBAL_RESCUE_FRACTION, build_semantic_progression_hierarchy, _beam_traverse_semantic,
)
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery
from progression_memory.evaluation.semantic import SentenceTransformerEmbedder, TextEmbedder, _minmax, cosine_scores


def _multi_view_rank(query: str, docs: tuple[RetrievalDocument, ...] | list[RetrievalDocument],
                     embedder: TextEmbedder) -> list[tuple[float, RetrievalDocument]]:
    """Hybrid lexical/semantic ranker over evidence plus progression state views."""
    if not docs: return []
    qe = embedder.encode([query])[0]
    by_id = {d.id: d for d in docs}
    scores = {d.id: .30 * v for d, v in (
        (d, _minmax({x.id: s for s, x in BM25Index(tuple(docs)).score(query)})[d.id])
        for d in docs
    )}
    fields = (("text", .30), ("progression_identity", .15),
              ("segment_state_after", .20), ("segment_delta", .05))
    active = .30
    for field, weight in fields:
        texts = [d.text if field == "text" else str(d.metadata.get(field, "")) for d in docs]
        if field != "text" and not any(texts): continue
        values = _minmax(dict(zip(by_id, cosine_scores(qe, embedder.encode(texts)).tolist(), strict=True)))
        for doc_id in scores: scores[doc_id] += weight * values[doc_id]
        active += weight
    return sorted(((score / active, by_id[doc_id]) for doc_id, score in scores.items()),
                  key=lambda x: (-x[0], x[1].id))

def build_hierarchical_beam_v4_queries(
    queries: list[RetrievalQuery], *, embedder: TextEmbedder | None = None,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Hybrid state-trajectory retrieval: lexical anchors plus semantic identity/state/delta."""
    embedder = embedder or SentenceTransformerEmbedder()
    cache, transformed = {}, []
    coverages, selected_counts, source_counts, visited_counts, rescue_counts = [], [], [], [], []
    for query in queries:
        key = id(query.documents)
        if key not in cache:
            segments, by, construction = _segment_indexes(query.documents, embedder=embedder)
            nodes, roots, depth = build_semantic_progression_hierarchy(by, embedder)
            cache[key] = segments, by, nodes, roots, depth, construction
        segments, by, nodes, roots, _, _ = cache[key]
        routed, visited = _beam_traverse_semantic(query.text, nodes, roots, embedder, 4)
        limit = _adaptive_candidate_limit(len(segments))
        global_rank = _multi_view_rank(query.text, segments, embedder)
        rescue_n = min(len(global_rank), max(6, math.ceil(DEFAULT_GLOBAL_RESCUE_FRACTION * limit)))
        rescued = {str(d.metadata.get("progression_id")) for _, d in global_rank[:rescue_n]}
        active = routed | rescued or set(by)
        branch = tuple(d for pid in active for d in by.get(pid, ()))
        selected = [d for _, d in _multi_view_rank(query.text, branch, embedder)[:limit]]
        sources, evidence = {}, set()
        for segment in selected:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    sources.setdefault(source.id, source); evidence.update(source.evidence_ids or {source.id})
        candidates = tuple(sources.values()) or query.documents
        # Raw evidence remains source of truth and is semantically reranked.
        raw_rank = _multi_view_rank(query.text, list(candidates), embedder)
        final = tuple(d for _, d in raw_rank)
        transformed.append(RetrievalQuery(id=query.id, text=query.text, answer=query.answer,
            evidence_ids=query.evidence_ids, documents=final, category=query.category))
        visited_counts.append(visited); rescue_counts.append(len(rescued))
        selected_counts.append(len(selected)); source_counts.append(len(candidates))
        if query.evidence_ids: coverages.append(len(evidence & query.evidence_ids) / len(query.evidence_ids))
    rows = list(cache.values())
    return transformed, {
        "unique_corpora": len(cache), "hierarchy": "state_trajectory_semantic_v4",
        "embedding_model": getattr(embedder, "model_name", type(embedder).__name__),
        "lexical_scoring": True, "views": ["lexical", "evidence", "identity", "state", "delta"],
        "avg_hierarchy_depth": sum(r[4] for r in rows) / max(1, len(rows)),
        "avg_nodes_visited_per_query": sum(visited_counts) / max(1, len(visited_counts)),
        "avg_rescued_progressions": sum(rescue_counts) / max(1, len(rescue_counts)),
        "avg_stage1_selected_units": sum(selected_counts) / max(1, len(selected_counts)),
        "avg_candidate_source_documents": sum(source_counts) / max(1, len(source_counts)),
        "avg_stage1_evidence_coverage": sum(coverages) / max(1, len(coverages)),
    }
