from __future__ import annotations

from collections import OrderedDict, defaultdict
import math

from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery, tokenize
from progression_memory.evaluation.two_stage import (
    MAX_STAGE1_CANDIDATES,
    MIN_STAGE1_CANDIDATES,
    STAGE1_CORPUS_FRACTION,
    _compact_index_text,
    _sources_for_segment,
)
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_graph import GraphRelation, ProgressionGraph

DEFAULT_PARENT_RRF_WEIGHT = 0.35
DEFAULT_RRF_K = 60.0


def _adaptive_candidate_limit(structural_units: int) -> int:
    proportional = math.ceil(STAGE1_CORPUS_FRACTION * structural_units)
    return min(
        structural_units,
        max(MIN_STAGE1_CANDIDATES, min(MAX_STAGE1_CANDIDATES, proportional)),
    )


def _parent_index_text(children: list[RetrievalDocument]) -> str:
    seen: set[str] = set()
    terms: list[str] = []
    for child in children:
        for token in tokenize(child.text):
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return " ".join(terms)


def _build_hierarchy(
    raw_documents: tuple[RetrievalDocument, ...],
) -> tuple[
    tuple[RetrievalDocument, ...],
    tuple[RetrievalDocument, ...],
    ProgressionGraph,
    dict[str, float | int],
]:
    segments, stats = build_progression_documents(raw_documents, mode="adaptive_v2")
    segment_indexes: list[RetrievalDocument] = []
    by_progression: defaultdict[str, list[RetrievalDocument]] = defaultdict(list)

    for segment in segments:
        progression_id = str(segment.metadata.get("progression_id"))
        index_document = RetrievalDocument(
            id=f"index:{segment.id}",
            text=_compact_index_text(segment),
            evidence_ids=frozenset(),
            timestamp=segment.timestamp,
            metadata={
                "segment_id": segment.id,
                "progression_id": progression_id,
                "source_documents": _sources_for_segment(segment, raw_documents),
            },
        )
        segment_indexes.append(index_document)
        by_progression[progression_id].append(index_document)

    progressions = {
        progression_id: Progression(
            id=progression_id,
            title=progression_id,
            scope="",
        )
        for progression_id in by_progression
    }
    graph = ProgressionGraph(progressions)
    for segment in segment_indexes:
        progression_id = str(segment.metadata["progression_id"])
        graph.add_segment(segment.id, progression_id)

    parent_indexes: list[RetrievalDocument] = []
    for progression_id, children in sorted(by_progression.items()):
        parent_indexes.append(
            RetrievalDocument(
                id=f"parent-index:{progression_id}",
                text=_parent_index_text(children),
                evidence_ids=frozenset(),
                metadata={
                    "progression_id": progression_id,
                    "child_segment_ids": tuple(child.id for child in children),
                },
            )
        )

    if not graph.validate():
        raise ValueError("constructed progression hierarchy is not acyclic")
    return tuple(parent_indexes), tuple(segment_indexes), graph, stats


def _hierarchical_segment_ranking(
    query: str,
    parent_indexes: tuple[RetrievalDocument, ...],
    segment_indexes: tuple[RetrievalDocument, ...],
    *,
    parent_weight: float,
    rrf_k: float,
) -> list[RetrievalDocument]:
    if not segment_indexes:
        return []

    parent_rank: dict[str, int] = {}
    if parent_indexes:
        for rank, (_, parent) in enumerate(BM25Index(parent_indexes).score(query), start=1):
            parent_rank[str(parent.metadata.get("progression_id"))] = rank

    scored: list[tuple[float, RetrievalDocument]] = []
    for segment_rank, (_, segment) in enumerate(BM25Index(segment_indexes).score(query), start=1):
        progression_id = str(segment.metadata.get("progression_id"))
        p_rank = parent_rank.get(progression_id, len(parent_indexes) + 1)
        score = 1.0 / (rrf_k + segment_rank)
        score += parent_weight / (rrf_k + p_rank)
        scored.append((score, segment))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [segment for _, segment in scored]


def build_dag_two_stage_queries(
    queries: list[RetrievalQuery],
    *,
    parent_rrf_weight: float = DEFAULT_PARENT_RRF_WEIGHT,
    rrf_k: float = DEFAULT_RRF_K,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Hierarchical Stage-1 retrieval followed by the same raw-source Stage 2.

    Progressions are parent nodes and adaptive-v2 temporal segments are child nodes.
    Parent and child lexical rankings are fused with reciprocal-rank fusion. The number
    of selected segment nodes uses the exact same adaptive breadth policy as flat
    two-stage retrieval, isolating hierarchy/ranking as the independent variable.

    Structural nodes carry no evidence credit. Only de-duplicated raw source memories
    become the final query corpus and therefore consume reader context.
    """
    if parent_rrf_weight < 0.0:
        raise ValueError("parent_rrf_weight must be non-negative")
    if rrf_k <= 0.0:
        raise ValueError("rrf_k must be positive")

    hierarchy_cache: dict[
        int,
        tuple[
            tuple[RetrievalDocument, ...],
            tuple[RetrievalDocument, ...],
            tuple[RetrievalDocument, ...],
            ProgressionGraph,
            dict[str, float | int],
        ],
    ] = {}
    transformed: list[RetrievalQuery] = []
    stage1_counts: list[int] = []
    candidate_source_counts: list[int] = []
    evidence_coverages: list[float] = []

    for query in queries:
        key = id(query.documents)
        cached = hierarchy_cache.get(key)
        if cached is None:
            parents, segments, graph, structure_stats = _build_hierarchy(query.documents)
            cached = (parents, segments, query.documents, graph, structure_stats)
            hierarchy_cache[key] = cached
        parents, segments, raw_documents, _, _ = cached

        ranked_segments = _hierarchical_segment_ranking(
            query.text,
            parents,
            segments,
            parent_weight=parent_rrf_weight,
            rrf_k=rrf_k,
        )
        candidate_limit = _adaptive_candidate_limit(len(segments)) if segments else 0
        selected_segments = ranked_segments[:candidate_limit]

        candidates: OrderedDict[str, RetrievalDocument] = OrderedDict()
        for segment in selected_segments:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    candidates.setdefault(source.id, source)
        candidate_documents = tuple(candidates.values()) or raw_documents
        stage1_counts.append(len(selected_segments))
        candidate_source_counts.append(len(candidate_documents))

        if query.evidence_ids:
            candidate_evidence: set[str] = set()
            for source in candidate_documents:
                candidate_evidence.update(source.evidence_ids or {source.id})
            evidence_coverages.append(
                len(candidate_evidence & query.evidence_ids) / len(query.evidence_ids)
            )

        transformed.append(
            RetrievalQuery(
                id=query.id,
                text=query.text,
                evidence_ids=query.evidence_ids,
                documents=candidate_documents,
                category=query.category,
                answer=query.answer,
            )
        )

    stat_rows = [row[-1] for row in hierarchy_cache.values()]

    def avg_stat(name: str) -> float:
        return sum(float(row[name]) for row in stat_rows) / max(1, len(stat_rows))

    parent_counts = [len(row[0]) for row in hierarchy_cache.values()]
    segment_counts = [len(row[1]) for row in hierarchy_cache.values()]
    graph_node_counts = [len(row[3].nodes) for row in hierarchy_cache.values()]
    graph_edge_counts = [len(row[3].edges) for row in hierarchy_cache.values()]
    parent_chars = [sum(len(doc.text) for doc in row[0]) for row in hierarchy_cache.values()]
    segment_chars = [sum(len(doc.text) for doc in row[1]) for row in hierarchy_cache.values()]
    raw_chars = [sum(len(doc.text) for doc in row[2]) for row in hierarchy_cache.values()]

    stats: dict[str, object] = {
        "unique_corpora": len(hierarchy_cache),
        "stage1_candidate_policy": "adaptive:max(16,min(64,ceil(0.10*structural_units)))",
        "hierarchy_ranking": f"rrf(child + {parent_rrf_weight}*parent),k={rrf_k}",
        "avg_parent_progressions": sum(parent_counts) / max(1, len(parent_counts)),
        "avg_structural_units": sum(segment_counts) / max(1, len(segment_counts)),
        "avg_graph_nodes": sum(graph_node_counts) / max(1, len(graph_node_counts)),
        "avg_graph_edges": sum(graph_edge_counts) / max(1, len(graph_edge_counts)),
        "avg_stage1_selected_units": sum(stage1_counts) / max(1, len(stage1_counts)),
        "avg_candidate_source_documents": (
            sum(candidate_source_counts) / max(1, len(candidate_source_counts))
        ),
        "avg_stage1_evidence_coverage": (
            sum(evidence_coverages) / max(1, len(evidence_coverages))
        ),
        "avg_raw_documents": avg_stat("raw_document_count") if stat_rows else 0.0,
        "avg_raw_chars": sum(raw_chars) / max(1, len(raw_chars)),
        "avg_segment_index_chars": sum(segment_chars) / max(1, len(segment_chars)),
        "avg_parent_index_chars": sum(parent_chars) / max(1, len(parent_chars)),
        "avg_total_index_chars": (
            sum(parent_chars) + sum(segment_chars)
        ) / max(1, len(raw_chars)),
        "avg_index_to_raw_char_ratio": (
            (sum(parent_chars) + sum(segment_chars)) / max(1, sum(raw_chars))
        ),
    }
    if stat_rows:
        stats["avg_segment_size"] = avg_stat("avg_segment_size")
        stats["avg_single_event_segment_fraction"] = avg_stat("single_event_segment_fraction")
    return transformed, stats
