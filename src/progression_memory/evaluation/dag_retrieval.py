from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import math

from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import (
    BM25Index,
    RetrievalDocument,
    RetrievalQuery,
    tokenize,
)
from progression_memory.evaluation.two_stage import (
    MAX_STAGE1_CANDIDATES,
    MIN_STAGE1_CANDIDATES,
    STAGE1_CORPUS_FRACTION,
    _compact_index_text,
    _sources_for_segment,
)
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_graph import ProgressionGraph

DEFAULT_PARENT_RRF_WEIGHT = 0.35
DEFAULT_RRF_K = 60.0

# Hierarchy-v2 defaults. These are deterministic architecture constants, not
# benchmark-label-tuned parameters.
V2_PARENT_INDEX_TERMS = 12
V2_PARENT_CORE_TERMS = 4
V2_DIVERSITY_FRACTION = 0.25
V2_MAX_DIVERSITY_PARENTS = 8


def _adaptive_candidate_limit(structural_units: int) -> int:
    proportional = math.ceil(STAGE1_CORPUS_FRACTION * structural_units)
    return min(
        structural_units,
        max(MIN_STAGE1_CANDIDATES, min(MAX_STAGE1_CANDIDATES, proportional)),
    )


def _segment_indexes(
    raw_documents: tuple[RetrievalDocument, ...],
) -> tuple[
    tuple[RetrievalDocument, ...],
    dict[str, list[RetrievalDocument]],
    dict[str, float | int],
]:
    segments, stats = build_progression_documents(raw_documents, mode="adaptive_v2")
    indexes: list[RetrievalDocument] = []
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
        indexes.append(index_document)
        by_progression[progression_id].append(index_document)
    return tuple(indexes), dict(by_progression), stats


def _graph_for_segments(
    by_progression: dict[str, list[RetrievalDocument]],
    segment_indexes: tuple[RetrievalDocument, ...],
) -> ProgressionGraph:
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
        graph.add_segment(
            segment.id,
            str(segment.metadata["progression_id"]),
        )
    if not graph.validate():
        raise ValueError("constructed progression hierarchy is not acyclic")
    return graph


def _parent_index_text(children: list[RetrievalDocument]) -> str:
    """Historical DAG-v1 parent representation: uncapped union of child terms."""
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
    """Build the historical DAG-v1 hierarchy unchanged for reproducibility."""
    segment_indexes, by_progression, stats = _segment_indexes(raw_documents)
    graph = _graph_for_segments(by_progression, segment_indexes)

    parent_indexes = tuple(
        RetrievalDocument(
            id=f"parent-index:{progression_id}",
            text=_parent_index_text(children),
            evidence_ids=frozenset(),
            metadata={
                "progression_id": progression_id,
                "child_segment_ids": tuple(child.id for child in children),
            },
        )
        for progression_id, children in sorted(by_progression.items())
    )
    return parent_indexes, segment_indexes, graph, stats


def _compact_parent_index_texts(
    by_progression: dict[str, list[RetrievalDocument]],
    *,
    max_terms: int = V2_PARENT_INDEX_TERMS,
    core_terms: int = V2_PARENT_CORE_TERMS,
) -> dict[str, str]:
    """Build compact parent summaries from child-level lexical statistics.

    Each parent keeps a few persistent terms (present in many of its children) and
    fills the remaining budget with terms that are simultaneously common within the
    parent and distinctive across parents. This keeps the parent routing layer small
    while preserving both topic identity and discriminative vocabulary.
    """
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    if core_terms < 0:
        raise ValueError("core_terms must be non-negative")

    local_counts: dict[str, Counter[str]] = {}
    parent_document_frequency: Counter[str] = Counter()
    for progression_id, children in by_progression.items():
        counts: Counter[str] = Counter()
        parent_terms: set[str] = set()
        for child in children:
            child_terms = set(tokenize(child.text))
            counts.update(child_terms)
            parent_terms.update(child_terms)
        local_counts[progression_id] = counts
        parent_document_frequency.update(parent_terms)

    parent_count = max(1, len(by_progression))
    result: dict[str, str] = {}
    for progression_id, counts in local_counts.items():
        if not counts:
            result[progression_id] = ""
            continue

        persistent = sorted(
            counts,
            key=lambda token: (-counts[token], token),
        )
        selected = persistent[: min(core_terms, max_terms)]
        selected_set = set(selected)

        scored: list[tuple[float, int, str]] = []
        for token, child_frequency in counts.items():
            if token in selected_set:
                continue
            global_frequency = parent_document_frequency[token]
            parent_idf = math.log(
                1.0
                + (
                    parent_count - global_frequency + 0.5
                )
                / (global_frequency + 0.5)
            )
            scored.append((child_frequency * parent_idf, child_frequency, token))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

        for _, _, token in scored:
            if len(selected) >= max_terms:
                break
            selected.append(token)
            selected_set.add(token)

        # If IDF scoring had no remaining candidates, deterministically backfill.
        if len(selected) < max_terms:
            for token in persistent:
                if token not in selected_set:
                    selected.append(token)
                    selected_set.add(token)
                    if len(selected) >= max_terms:
                        break

        result[progression_id] = " ".join(selected)
    return result


def _build_hierarchy_v2(
    raw_documents: tuple[RetrievalDocument, ...],
    *,
    parent_index_terms: int = V2_PARENT_INDEX_TERMS,
) -> tuple[
    tuple[RetrievalDocument, ...],
    tuple[RetrievalDocument, ...],
    ProgressionGraph,
    dict[str, float | int],
]:
    segment_indexes, by_progression, stats = _segment_indexes(raw_documents)
    graph = _graph_for_segments(by_progression, segment_indexes)
    parent_texts = _compact_parent_index_texts(
        by_progression,
        max_terms=parent_index_terms,
        core_terms=min(V2_PARENT_CORE_TERMS, parent_index_terms),
    )

    parent_indexes = tuple(
        RetrievalDocument(
            id=f"parent-index:{progression_id}",
            text=parent_texts[progression_id],
            evidence_ids=frozenset(),
            metadata={
                "progression_id": progression_id,
                "child_segment_ids": tuple(child.id for child in children),
            },
        )
        for progression_id, children in sorted(by_progression.items())
    )
    return parent_indexes, segment_indexes, graph, stats


def _hierarchical_segment_ranking(
    query: str,
    parent_indexes: tuple[RetrievalDocument, ...],
    segment_indexes: tuple[RetrievalDocument, ...],
    *,
    parent_weight: float,
    rrf_k: float,
) -> list[RetrievalDocument]:
    """Historical DAG-v1 RRF ranking, preserved unchanged."""
    if not segment_indexes:
        return []

    parent_rank: dict[str, int] = {}
    if parent_indexes:
        for rank, (_, parent) in enumerate(BM25Index(parent_indexes).score(query), start=1):
            parent_rank[str(parent.metadata.get("progression_id"))] = rank

    scored: list[tuple[float, RetrievalDocument]] = []
    for segment_rank, (_, segment) in enumerate(
        BM25Index(segment_indexes).score(query), start=1
    ):
        progression_id = str(segment.metadata.get("progression_id"))
        p_rank = parent_rank.get(progression_id, len(parent_indexes) + 1)
        score = 1.0 / (rrf_k + segment_rank)
        score += parent_weight / (rrf_k + p_rank)
        scored.append((score, segment))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [segment for _, segment in scored]


def _hierarchical_segment_ranking_v2(
    query: str,
    parent_indexes: tuple[RetrievalDocument, ...],
    segment_indexes: tuple[RetrievalDocument, ...],
    *,
    candidate_limit: int,
    parent_weight: float,
    rrf_k: float,
    diversity_fraction: float = V2_DIVERSITY_FRACTION,
    max_diversity_parents: int = V2_MAX_DIVERSITY_PARENTS,
) -> tuple[list[RetrievalDocument], int, int]:
    """Fuse parent/child relevance and reserve part of Stage 1 for parent diversity.

    The total number of selected segment nodes is unchanged from flat Retrieval-v2.
    Hierarchy-v2 only reallocates that fixed budget: a bounded prefix is filled
    round-robin from the strongest *positive-scoring* parent progressions, then the
    remainder follows the normal fused ranking. This makes hierarchy affect candidate
    coverage rather than merely nudging otherwise-identical segment ranks.
    """
    if candidate_limit <= 0 or not segment_indexes:
        return [], 0, 0
    if not 0.0 <= diversity_fraction <= 1.0:
        raise ValueError("diversity_fraction must be between 0 and 1")
    if max_diversity_parents < 1:
        raise ValueError("max_diversity_parents must be positive")

    parent_scored = BM25Index(parent_indexes).score(query) if parent_indexes else []
    parent_rank = {
        str(parent.metadata.get("progression_id")): rank
        for rank, (_, parent) in enumerate(parent_scored, start=1)
    }
    positive_parents = [
        str(parent.metadata.get("progression_id"))
        for score, parent in parent_scored
        if score > 0.0
    ]

    child_scored = BM25Index(segment_indexes).score(query)
    child_rank = {
        segment.id: rank
        for rank, (_, segment) in enumerate(child_scored, start=1)
    }
    by_parent: defaultdict[str, list[RetrievalDocument]] = defaultdict(list)
    fused: list[tuple[float, RetrievalDocument]] = []
    for _, segment in child_scored:
        progression_id = str(segment.metadata.get("progression_id"))
        by_parent[progression_id].append(segment)
        p_rank = parent_rank.get(progression_id, len(parent_indexes) + 1)
        score = 1.0 / (rrf_k + child_rank[segment.id])
        score += parent_weight / (rrf_k + p_rank)
        fused.append((score, segment))

    for children in by_parent.values():
        children.sort(key=lambda segment: (child_rank[segment.id], segment.id))
    fused.sort(key=lambda item: (-item[0], item[1].id))

    parent_seed_target = min(
        len(positive_parents),
        max_diversity_parents,
        max(1, math.ceil(math.sqrt(candidate_limit))),
    )
    seed_parents = positive_parents[:parent_seed_target]
    diversity_budget = min(
        candidate_limit,
        max(parent_seed_target, math.ceil(diversity_fraction * candidate_limit)),
    )

    selected: list[RetrievalDocument] = []
    selected_ids: set[str] = set()
    cursor = {parent_id: 0 for parent_id in seed_parents}
    while len(selected) < diversity_budget and seed_parents:
        added = False
        for parent_id in seed_parents:
            children = by_parent.get(parent_id, [])
            idx = cursor[parent_id]
            while idx < len(children) and children[idx].id in selected_ids:
                idx += 1
            cursor[parent_id] = idx
            if idx >= len(children):
                continue
            child = children[idx]
            cursor[parent_id] += 1
            selected.append(child)
            selected_ids.add(child.id)
            added = True
            if len(selected) >= diversity_budget:
                break
        if not added:
            break

    for _, segment in fused:
        if len(selected) >= candidate_limit:
            break
        if segment.id not in selected_ids:
            selected.append(segment)
            selected_ids.add(segment.id)

    unique_progressions = len(
        {str(segment.metadata.get("progression_id")) for segment in selected}
    )
    return selected, len(seed_parents), unique_progressions


def _finalize_hierarchy_stats(
    hierarchy_cache: dict[
        int,
        tuple[
            tuple[RetrievalDocument, ...],
            tuple[RetrievalDocument, ...],
            tuple[RetrievalDocument, ...],
            ProgressionGraph,
            dict[str, float | int],
        ],
    ],
    *,
    stage1_counts: list[int],
    candidate_source_counts: list[int],
    evidence_coverages: list[float],
    hierarchy_ranking: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
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
        "stage1_candidate_policy": (
            "adaptive:max(16,min(64,ceil(0.10*structural_units)))"
        ),
        "hierarchy_ranking": hierarchy_ranking,
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
        stats["avg_single_event_segment_fraction"] = avg_stat(
            "single_event_segment_fraction"
        )
    if extra:
        stats.update(extra)
    return stats


def build_dag_two_stage_queries(
    queries: list[RetrievalQuery],
    *,
    parent_rrf_weight: float = DEFAULT_PARENT_RRF_WEIGHT,
    rrf_k: float = DEFAULT_RRF_K,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Historical DAG-v1 Stage-1 retrieval followed by raw-source Stage 2."""
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

    stats = _finalize_hierarchy_stats(
        hierarchy_cache,
        stage1_counts=stage1_counts,
        candidate_source_counts=candidate_source_counts,
        evidence_coverages=evidence_coverages,
        hierarchy_ranking=f"rrf(child + {parent_rrf_weight}*parent),k={rrf_k}",
    )
    return transformed, stats


def build_dag_two_stage_v2_queries(
    queries: list[RetrievalQuery],
    *,
    parent_rrf_weight: float = DEFAULT_PARENT_RRF_WEIGHT,
    rrf_k: float = DEFAULT_RRF_K,
    parent_index_terms: int = V2_PARENT_INDEX_TERMS,
    diversity_fraction: float = V2_DIVERSITY_FRACTION,
    max_diversity_parents: int = V2_MAX_DIVERSITY_PARENTS,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Compact, parent-diversified hierarchy over the same Retrieval-v2 substrate."""
    if parent_rrf_weight < 0.0:
        raise ValueError("parent_rrf_weight must be non-negative")
    if rrf_k <= 0.0:
        raise ValueError("rrf_k must be positive")
    if parent_index_terms < 1:
        raise ValueError("parent_index_terms must be positive")
    if not 0.0 <= diversity_fraction <= 1.0:
        raise ValueError("diversity_fraction must be between 0 and 1")
    if max_diversity_parents < 1:
        raise ValueError("max_diversity_parents must be positive")

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
    seeded_parent_counts: list[int] = []
    unique_progression_counts: list[int] = []

    for query in queries:
        key = id(query.documents)
        cached = hierarchy_cache.get(key)
        if cached is None:
            parents, segments, graph, structure_stats = _build_hierarchy_v2(
                query.documents,
                parent_index_terms=parent_index_terms,
            )
            cached = (parents, segments, query.documents, graph, structure_stats)
            hierarchy_cache[key] = cached
        parents, segments, raw_documents, _, _ = cached

        candidate_limit = _adaptive_candidate_limit(len(segments)) if segments else 0
        selected_segments, seeded_parents, unique_progressions = (
            _hierarchical_segment_ranking_v2(
                query.text,
                parents,
                segments,
                candidate_limit=candidate_limit,
                parent_weight=parent_rrf_weight,
                rrf_k=rrf_k,
                diversity_fraction=diversity_fraction,
                max_diversity_parents=max_diversity_parents,
            )
        )

        candidates: OrderedDict[str, RetrievalDocument] = OrderedDict()
        for segment in selected_segments:
            for source in segment.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    candidates.setdefault(source.id, source)
        candidate_documents = tuple(candidates.values()) or raw_documents
        stage1_counts.append(len(selected_segments))
        candidate_source_counts.append(len(candidate_documents))
        seeded_parent_counts.append(seeded_parents)
        unique_progression_counts.append(unique_progressions)

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

    stats = _finalize_hierarchy_stats(
        hierarchy_cache,
        stage1_counts=stage1_counts,
        candidate_source_counts=candidate_source_counts,
        evidence_coverages=evidence_coverages,
        hierarchy_ranking=(
            f"diversified_rrf(child + {parent_rrf_weight}*parent),k={rrf_k}"
        ),
        extra={
            "parent_index_max_terms": parent_index_terms,
            "diversity_fraction": diversity_fraction,
            "max_diversity_parents": max_diversity_parents,
            "avg_seeded_parent_count": (
                sum(seeded_parent_counts) / max(1, len(seeded_parent_counts))
            ),
            "avg_stage1_unique_progressions": (
                sum(unique_progression_counts) / max(1, len(unique_progression_counts))
            ),
        },
    )
    return transformed, stats
