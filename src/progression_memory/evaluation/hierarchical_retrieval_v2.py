from __future__ import annotations

import math

from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit, _segment_indexes
from progression_memory.evaluation.hierarchical_retrieval import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BRANCH_FACTOR,
    DEFAULT_MAX_PILLARS,
    _beam_traverse,
    _query_is_temporal,
    build_progression_hierarchy,
)
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery

DEFAULT_GLOBAL_RESCUE_FRACTION = 0.35
DEFAULT_BRANCH_BUDGET_FRACTION = 0.60
DEFAULT_TEMPORAL_BUDGET_FRACTION = 0.25


def _soft_stage1(
    query: str,
    segments: tuple[RetrievalDocument, ...],
    by_progression: dict[str, list[RetrievalDocument]],
    routed: set[str],
    *,
    candidate_limit: int,
    rescue_fraction: float = DEFAULT_GLOBAL_RESCUE_FRACTION,
) -> tuple[list[RetrievalDocument], set[str], int]:
    """Use hierarchy as a prior while retaining global leaf rescue.

    Unlike v1, a wrong coarse branch cannot make globally strong fine-grained evidence
    unreachable. A bounded global anchor set can reopen progressions, and the final
    candidate set is filled from the global ranking after branch-focused selection.
    """
    global_rank = BM25Index(segments).score(query)
    rescue_count = min(
        len(global_rank),
        max(6, math.ceil(rescue_fraction * candidate_limit)),
    )
    rescued_progressions = {
        str(segment.metadata.get("progression_id"))
        for score, segment in global_rank[:rescue_count]
        if score > 0.0
    }
    active = set(routed) | rescued_progressions
    if not active:
        active = set(by_progression)

    branch_pool = tuple(
        segment
        for pid in active
        for segment in by_progression.get(pid, [])
    )
    branch_rank = BM25Index(branch_pool).score(query) if branch_pool else []

    selected: list[RetrievalDocument] = []
    selected_ids: set[str] = set()
    temporal_budget = (
        min(candidate_limit, max(1, math.ceil(DEFAULT_TEMPORAL_BUDGET_FRACTION * candidate_limit)))
        if _query_is_temporal(query)
        else 0
    )

    # Temporal questions reserve only a bounded prefix for local state continuity.
    for score, anchor in branch_rank:
        if len(selected) >= temporal_budget:
            break
        if score <= 0.0:
            break
        pid = str(anchor.metadata.get("progression_id"))
        siblings = sorted(by_progression.get(pid, []), key=lambda doc: (doc.timestamp or "", doc.id))
        index = next((i for i, doc in enumerate(siblings) if doc.id == anchor.id), 0)
        for neighbor in siblings[max(0, index - 1): min(len(siblings), index + 2)]:
            if neighbor.id not in selected_ids:
                selected.append(neighbor)
                selected_ids.add(neighbor.id)
                if len(selected) >= temporal_budget:
                    break

    branch_target = min(
        candidate_limit,
        max(len(selected), math.ceil(DEFAULT_BRANCH_BUDGET_FRACTION * candidate_limit)),
    )
    for score, segment in branch_rank:
        if len(selected) >= branch_target:
            break
        if score <= 0.0 and selected:
            break
        if segment.id not in selected_ids:
            selected.append(segment)
            selected_ids.add(segment.id)

    # Global fill is the safety valve: hierarchy focuses the early budget but never
    # permanently deletes a strong fine-grained candidate.
    for _, segment in global_rank:
        if len(selected) >= candidate_limit:
            break
        if segment.id not in selected_ids:
            selected.append(segment)
            selected_ids.add(segment.id)

    return selected, active, len(rescued_progressions)


def build_hierarchical_beam_v2_queries(
    queries: list[RetrievalQuery],
    *,
    branch_factor: int = DEFAULT_BRANCH_FACTOR,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_pillars: int = DEFAULT_MAX_PILLARS,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Soft coarse-to-fine traversal with global rescue and bounded temporal expansion."""
    cache: dict[int, tuple[tuple[RetrievalDocument, ...], dict[str, list[RetrievalDocument]], dict, tuple[str, ...], int]] = {}
    transformed: list[RetrievalQuery] = []
    visited_counts: list[int] = []
    progression_counts: list[int] = []
    rescued_counts: list[int] = []
    segment_counts: list[int] = []
    source_counts: list[int] = []
    coverages: list[float] = []

    for query in queries:
        key = id(query.documents)
        if key not in cache:
            segments, by_progression, _ = _segment_indexes(query.documents)
            nodes, roots, depth = build_progression_hierarchy(
                by_progression, branch_factor=branch_factor, max_pillars=max_pillars
            )
            cache[key] = (segments, by_progression, nodes, roots, depth)
        segments, by_progression, nodes, roots, _ = cache[key]
        routed, visited = _beam_traverse(query.text, nodes, roots, beam_width=beam_width)
        selected_segments, active, rescued_count = _soft_stage1(
            query.text,
            segments,
            by_progression,
            routed,
            candidate_limit=_adaptive_candidate_limit(len(segments)),
        )

        sources: dict[str, RetrievalDocument] = {}
        candidate_evidence: set[str] = set()
        for segment in selected_segments:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    sources.setdefault(source.id, source)
                    candidate_evidence.update(source.evidence_ids or {source.id})
        candidate_documents = tuple(sources.values()) or query.documents
        final_docs = tuple(doc for _, doc in BM25Index(candidate_documents).score(query.text))

        visited_counts.append(visited)
        progression_counts.append(len(active))
        rescued_counts.append(rescued_count)
        segment_counts.append(len(selected_segments))
        source_counts.append(len(candidate_documents))
        if query.evidence_ids:
            coverages.append(len(candidate_evidence & query.evidence_ids) / len(query.evidence_ids))
        transformed.append(RetrievalQuery(
            id=query.id,
            text=query.text,
            answer=query.answer,
            evidence_ids=query.evidence_ids,
            documents=final_docs,
            category=query.category,
        ))

    corpora = list(cache.values())
    stats: dict[str, object] = {
        "unique_corpora": len(cache),
        "hierarchy": "soft_inferred_pillars->coarse_progressions->fine_progressions->temporal_segments",
        "branch_factor": branch_factor,
        "beam_width": beam_width,
        "max_pillars": max_pillars,
        "avg_hierarchy_depth": sum(row[4] for row in corpora) / max(1, len(corpora)),
        "avg_pillars": sum(len(row[3]) for row in corpora) / max(1, len(corpora)),
        "avg_leaf_progressions": sum(len(row[1]) for row in corpora) / max(1, len(corpora)),
        "avg_hierarchy_nodes": sum(len(row[2]) for row in corpora) / max(1, len(corpora)),
        "avg_nodes_visited_per_query": sum(visited_counts) / max(1, len(visited_counts)),
        "avg_selected_progressions": sum(progression_counts) / max(1, len(progression_counts)),
        "avg_rescued_progressions": sum(rescued_counts) / max(1, len(rescued_counts)),
        "avg_stage1_selected_units": sum(segment_counts) / max(1, len(segment_counts)),
        "avg_candidate_source_documents": sum(source_counts) / max(1, len(source_counts)),
        "avg_stage1_evidence_coverage": sum(coverages) / max(1, len(coverages)),
        "global_rescue_fraction": DEFAULT_GLOBAL_RESCUE_FRACTION,
        "branch_budget_fraction": DEFAULT_BRANCH_BUDGET_FRACTION,
        "temporal_budget_fraction": DEFAULT_TEMPORAL_BUDGET_FRACTION,
        "hard_branch_pruning": False,
    }
    return transformed, stats
