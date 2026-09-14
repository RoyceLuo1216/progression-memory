from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from progression_memory.evaluation.dag_retrieval import (
    _adaptive_candidate_limit,
    _segment_indexes,
)
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery, tokenize
from progression_memory.evaluation.two_stage import _sources_for_segment

DEFAULT_BRANCH_FACTOR = 4
DEFAULT_BEAM_WIDTH = 4
DEFAULT_ANCHOR_RESCUE = 6
DEFAULT_MAX_PILLARS = 8
DEFAULT_NODE_TERMS = 16


@dataclass(frozen=True)
class HierarchyNode:
    id: str
    level: int
    text: str
    child_ids: tuple[str, ...]
    leaf_progression_ids: tuple[str, ...]


def _compact_terms(documents: list[RetrievalDocument], max_terms: int = DEFAULT_NODE_TERMS) -> str:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    offset = 0
    for document in documents:
        seen = set(tokenize(document.text))
        for token in seen:
            counts[token] += 1
            first_seen.setdefault(token, offset)
            offset += 1
    terms = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    return " ".join(terms[:max_terms])


def _node_similarity(left: HierarchyNode, right: HierarchyNode) -> float:
    a = set(tokenize(left.text))
    b = set(tokenize(right.text))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _leaf_nodes(
    by_progression: dict[str, list[RetrievalDocument]],
) -> dict[str, HierarchyNode]:
    return {
        progression_id: HierarchyNode(
            id=progression_id,
            level=0,
            text=_compact_terms(children),
            child_ids=tuple(child.id for child in children),
            leaf_progression_ids=(progression_id,),
        )
        for progression_id, children in sorted(by_progression.items())
    }


def _group_level(nodes: list[HierarchyNode], level: int, branch_factor: int) -> list[HierarchyNode]:
    remaining = sorted(nodes, key=lambda node: node.id)
    groups: list[list[HierarchyNode]] = []
    while remaining:
        seed = remaining.pop(0)
        ranked = sorted(
            remaining,
            key=lambda node: (-_node_similarity(seed, node), node.id),
        )
        companions = ranked[: max(0, branch_factor - 1)]
        companion_ids = {node.id for node in companions}
        remaining = [node for node in remaining if node.id not in companion_ids]
        groups.append([seed, *companions])

    parents: list[HierarchyNode] = []
    for index, group in enumerate(groups):
        child_docs = [
            RetrievalDocument(id=node.id, text=node.text, evidence_ids=frozenset())
            for node in group
        ]
        leaves = sorted({leaf for node in group for leaf in node.leaf_progression_ids})
        parents.append(
            HierarchyNode(
                id=f"hierarchy:l{level}:{index}",
                level=level,
                text=_compact_terms(child_docs),
                child_ids=tuple(node.id for node in group),
                leaf_progression_ids=tuple(leaves),
            )
        )
    return parents


def build_progression_hierarchy(
    by_progression: dict[str, list[RetrievalDocument]],
    *,
    branch_factor: int = DEFAULT_BRANCH_FACTOR,
    max_pillars: int = DEFAULT_MAX_PILLARS,
) -> tuple[dict[str, HierarchyNode], tuple[str, ...], int]:
    """Build an inferred coarse-to-fine progression hierarchy.

    Fine progressions are the leaves. Successive layers group semantically similar
    progression nodes, producing durable coarse roots ("pillars"). The builder is
    deterministic and evidence-free; explicit user-defined roots can later replace
    the inferred top layer without changing traversal.
    """
    if branch_factor < 2:
        raise ValueError("branch_factor must be at least 2")
    if max_pillars < 1:
        raise ValueError("max_pillars must be positive")

    leaves = _leaf_nodes(by_progression)
    all_nodes: dict[str, HierarchyNode] = dict(leaves)
    current = list(leaves.values())
    level = 1
    while len(current) > max_pillars:
        current = _group_level(current, level, branch_factor)
        all_nodes.update({node.id: node for node in current})
        level += 1
    roots = tuple(node.id for node in current)
    return all_nodes, roots, max((node.level for node in all_nodes.values()), default=0)


def _children(nodes: dict[str, HierarchyNode], node: HierarchyNode) -> list[HierarchyNode]:
    return [nodes[child_id] for child_id in node.child_ids if child_id in nodes]


def _score_nodes(query: str, nodes: list[HierarchyNode]) -> list[tuple[float, HierarchyNode]]:
    if not nodes:
        return []
    docs = tuple(
        RetrievalDocument(id=node.id, text=node.text, evidence_ids=frozenset())
        for node in nodes
    )
    by_id = {node.id: node for node in nodes}
    return [(score, by_id[doc.id]) for score, doc in BM25Index(docs).score(query)]


def _query_is_broad(query: str) -> bool:
    q = query.casefold()
    return any(cue in q for cue in ("overall", "generally", "in general", "what have", "summarize", "summary"))


def _query_is_temporal(query: str) -> bool:
    q = query.casefold()
    return any(
        cue in q
        for cue in (
            "when", "before", "after", "first", "last", "latest", "recent",
            "currently", "now", "changed", "change", "progress", "over time",
        )
    )


def _beam_traverse(
    query: str,
    nodes: dict[str, HierarchyNode],
    roots: tuple[str, ...],
    *,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[set[str], int, int]:
    frontier = [nodes[root_id] for root_id in roots]
    visited = 0
    depth = 0
    broad = _query_is_broad(query)

    while frontier:
        scored = _score_nodes(query, frontier)
        visited += len(frontier)
        positive = [node for score, node in scored if score > 0.0]
        chosen = (positive or [node for _, node in scored])[:beam_width]
        if not chosen:
            break
        depth = max(depth, max(node.level for node in chosen))

        # Broad questions intentionally stop one abstraction layer above the leaves;
        # the selected node then contributes all fine progressions beneath it.
        if broad and all(node.level <= 1 for node in chosen):
            return {leaf for node in chosen for leaf in node.leaf_progression_ids}, visited, depth

        child_nodes = [child for node in chosen for child in _children(nodes, node)]
        if not child_nodes:
            return {leaf for node in chosen for leaf in node.leaf_progression_ids}, visited, depth
        frontier = child_nodes

    return set(), visited, depth


def _bottom_up_rescue(
    query: str,
    segment_indexes: tuple[RetrievalDocument, ...],
    *,
    anchors: int = DEFAULT_ANCHOR_RESCUE,
) -> set[str]:
    rescued: set[str] = set()
    for score, segment in BM25Index(segment_indexes).score(query)[:anchors]:
        if score <= 0.0:
            continue
        rescued.add(str(segment.metadata.get("progression_id")))
    return rescued


def _temporal_anchor_expand(
    query: str,
    progression_ids: set[str],
    by_progression: dict[str, list[RetrievalDocument]],
    *,
    candidate_limit: int,
) -> list[RetrievalDocument]:
    pool = [segment for pid in progression_ids for segment in by_progression.get(pid, [])]
    if not pool:
        return []
    ranked = BM25Index(tuple(pool)).score(query)
    temporal = _query_is_temporal(query)
    selected: list[RetrievalDocument] = []
    selected_ids: set[str] = set()

    for score, anchor in ranked:
        if len(selected) >= candidate_limit:
            break
        if score <= 0.0 and selected:
            break
        pid = str(anchor.metadata.get("progression_id"))
        siblings = by_progression.get(pid, [])
        ordered = sorted(siblings, key=lambda doc: (doc.timestamp or "", doc.id))
        try:
            index = next(i for i, doc in enumerate(ordered) if doc.id == anchor.id)
        except StopIteration:
            index = 0
        radius = 1 if temporal else 0
        neighborhood = ordered[max(0, index - radius) : min(len(ordered), index + radius + 1)]
        for segment in neighborhood:
            if segment.id not in selected_ids:
                selected.append(segment)
                selected_ids.add(segment.id)
                if len(selected) >= candidate_limit:
                    break

    for _, segment in ranked:
        if len(selected) >= candidate_limit:
            break
        if segment.id not in selected_ids:
            selected.append(segment)
            selected_ids.add(segment.id)
    return selected


def build_hierarchical_beam_queries(
    queries: list[RetrievalQuery],
    *,
    branch_factor: int = DEFAULT_BRANCH_FACTOR,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    anchor_rescue: int = DEFAULT_ANCHOR_RESCUE,
    max_pillars: int = DEFAULT_MAX_PILLARS,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Coarse-to-fine progression retrieval with bottom-up rescue and temporal expansion."""
    cache: dict[int, tuple[tuple[RetrievalDocument, ...], dict[str, list[RetrievalDocument]], dict[str, HierarchyNode], tuple[str, ...], int, dict[str, float | int]]] = {}
    transformed: list[RetrievalQuery] = []
    visited_counts: list[int] = []
    selected_progression_counts: list[int] = []
    candidate_segment_counts: list[int] = []
    candidate_source_counts: list[int] = []
    stage1_coverages: list[float] = []

    for query in queries:
        key = id(query.documents)
        if key not in cache:
            segment_indexes, by_progression, base_stats = _segment_indexes(query.documents)
            nodes, roots, depth = build_progression_hierarchy(
                by_progression,
                branch_factor=branch_factor,
                max_pillars=max_pillars,
            )
            cache[key] = (segment_indexes, by_progression, nodes, roots, depth, base_stats)
        segment_indexes, by_progression, nodes, roots, _, _ = cache[key]

        routed, visited, _ = _beam_traverse(query.text, nodes, roots, beam_width=beam_width)
        routed.update(_bottom_up_rescue(query.text, segment_indexes, anchors=anchor_rescue))
        if not routed:
            routed = set(by_progression)
        candidate_limit = _adaptive_candidate_limit(len(segment_indexes))
        selected_segments = _temporal_anchor_expand(
            query.text,
            routed,
            by_progression,
            candidate_limit=candidate_limit,
        )

        sources: dict[str, RetrievalDocument] = {}
        candidate_evidence: set[str] = set()
        for segment in selected_segments:
            for source in _sources_for_segment(segment, query.documents):
                sources[source.id] = source
                candidate_evidence.update(source.evidence_ids or {source.id})
        source_tuple = tuple(sources.values())
        final_docs = tuple(doc for _, doc in BM25Index(source_tuple).score(query.text)) if source_tuple else tuple()

        visited_counts.append(visited)
        selected_progression_counts.append(len(routed))
        candidate_segment_counts.append(len(selected_segments))
        candidate_source_counts.append(len(final_docs))
        if query.evidence_ids:
            stage1_coverages.append(len(candidate_evidence & query.evidence_ids) / len(query.evidence_ids))

        transformed.append(
            RetrievalQuery(
                id=query.id,
                text=query.text,
                answer=query.answer,
                evidence_ids=query.evidence_ids,
                documents=final_docs,
                category=query.category,
            )
        )

    corpora = list(cache.values())
    depths = [row[4] for row in corpora]
    pillar_counts = [len(row[3]) for row in corpora]
    leaf_counts = [len(row[1]) for row in corpora]
    node_counts = [len(row[2]) for row in corpora]
    raw_chars = [sum(len(doc.text) for doc in queries_for_docs[0]) for queries_for_docs in []]

    stats: dict[str, object] = {
        "unique_corpora": len(cache),
        "hierarchy": "inferred_pillars->coarse_progressions->fine_progressions->temporal_segments",
        "branch_factor": branch_factor,
        "beam_width": beam_width,
        "bottom_up_anchor_rescue": anchor_rescue,
        "max_pillars": max_pillars,
        "avg_hierarchy_depth": sum(depths) / max(1, len(depths)),
        "avg_pillars": sum(pillar_counts) / max(1, len(pillar_counts)),
        "avg_leaf_progressions": sum(leaf_counts) / max(1, len(leaf_counts)),
        "avg_hierarchy_nodes": sum(node_counts) / max(1, len(node_counts)),
        "avg_nodes_visited_per_query": sum(visited_counts) / max(1, len(visited_counts)),
        "avg_selected_progressions": sum(selected_progression_counts) / max(1, len(selected_progression_counts)),
        "avg_stage1_selected_units": sum(candidate_segment_counts) / max(1, len(candidate_segment_counts)),
        "avg_candidate_source_documents": sum(candidate_source_counts) / max(1, len(candidate_source_counts)),
        "avg_stage1_evidence_coverage": sum(stage1_coverages) / max(1, len(stage1_coverages)),
        "query_dependent_stopping": True,
        "temporal_anchor_expansion": True,
    }
    return transformed, stats
