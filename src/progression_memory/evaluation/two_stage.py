from __future__ import annotations

from collections import OrderedDict
import math

from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import (
    BM25Index,
    RetrievalDocument,
    RetrievalQuery,
    tokenize,
)

MIN_STAGE1_CANDIDATES = 16
MAX_STAGE1_CANDIDATES = 64
STAGE1_CORPUS_FRACTION = 0.10


def _compact_index_text(document: RetrievalDocument) -> str:
    """Create a compact lexical index without exposing evidence to the reader.

    BM25 is bag-of-words based, so retaining each distinct content token once preserves
    lexical query coverage while avoiding repetition from full segment payloads.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(document.text):
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return " ".join(terms)


def _sources_for_segment(
    segment: RetrievalDocument,
    source_documents: tuple[RetrievalDocument, ...],
) -> tuple[RetrievalDocument, ...]:
    evidence = segment.evidence_ids
    return tuple(
        source
        for source in source_documents
        if (source.evidence_ids or frozenset({source.id})) & evidence
    )


def build_two_stage_queries(
    queries: list[RetrievalQuery],
    *,
    stage1_candidates: int | None = None,
) -> tuple[list[RetrievalQuery], dict[str, object]]:
    """Restrict raw retrieval to sources under top-ranked compact memory regions.

    Stage 1 indexes adaptive-v2 temporal segments using compact lexical representations.
    Structural index documents intentionally carry no evidence credit.

    Stage 2 gathers raw source memories beneath the selected structural regions and
    returns those raw memories as the query corpus. The normal BM25 evaluator then
    reranks only raw source memories, so final evidence credit and reader-context cost
    are both measured exclusively on text actually exposed to the reader.
    """
    if stage1_candidates is not None and stage1_candidates < 1:
        raise ValueError("stage1_candidates must be positive")

    structure_cache: dict[
        int,
        tuple[
            tuple[RetrievalDocument, ...],
            tuple[RetrievalDocument, ...],
            dict[str, float | int],
        ],
    ] = {}
    transformed: list[RetrievalQuery] = []
    candidate_source_counts: list[int] = []
    selected_stage1_counts: list[int] = []
    candidate_evidence_coverages: list[float] = []

    for query in queries:
        key = id(query.documents)
        cached = structure_cache.get(key)
        if cached is None:
            segments, stats = build_progression_documents(
                query.documents,
                mode="adaptive_v2",
            )
            index_documents: list[RetrievalDocument] = []
            for segment in segments:
                sources = _sources_for_segment(segment, query.documents)
                index_documents.append(
                    RetrievalDocument(
                        id=f"index:{segment.id}",
                        text=_compact_index_text(segment),
                        evidence_ids=frozenset(),
                        timestamp=segment.timestamp,
                        metadata={
                            "segment_id": segment.id,
                            "progression_id": segment.metadata.get("progression_id"),
                            "source_documents": sources,
                        },
                    )
                )
            cached = (tuple(index_documents), query.documents, stats)
            structure_cache[key] = cached

        index_documents, raw_documents, _ = cached
        if stage1_candidates is None:
            proportional = math.ceil(STAGE1_CORPUS_FRACTION * len(index_documents))
            candidate_limit = min(
                len(index_documents),
                max(MIN_STAGE1_CANDIDATES, min(MAX_STAGE1_CANDIDATES, proportional)),
            )
        else:
            candidate_limit = min(stage1_candidates, len(index_documents))
        stage1 = BM25Index(index_documents).retrieve(query.text, candidate_limit)

        # Ordered de-duplication matters because a raw episode may support multiple
        # progression segments. Multi-membership must not consume context twice.
        candidates: OrderedDict[str, RetrievalDocument] = OrderedDict()
        for index_document in stage1:
            for source in index_document.metadata.get("source_documents", ()):
                if isinstance(source, RetrievalDocument):
                    candidates.setdefault(source.id, source)

        # Degenerate fallback: if structure construction produced no candidate sources,
        # preserve correctness by exposing the original corpus rather than an empty set.
        candidate_documents = tuple(candidates.values()) or raw_documents
        candidate_source_counts.append(len(candidate_documents))
        selected_stage1_counts.append(len(stage1))

        if query.evidence_ids:
            candidate_evidence: set[str] = set()
            for source in candidate_documents:
                candidate_evidence.update(source.evidence_ids or {source.id})
            candidate_evidence_coverages.append(
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

    stat_rows = [stats for _, _, stats in structure_cache.values()]

    def avg_stat(name: str) -> float:
        return sum(float(row[name]) for row in stat_rows) / max(1, len(stat_rows))

    index_char_counts = [
        sum(len(document.text) for document in index_documents)
        for index_documents, _, _ in structure_cache.values()
    ]
    raw_char_counts = [
        sum(len(document.text) for document in raw_documents)
        for _, raw_documents, _ in structure_cache.values()
    ]

    stats: dict[str, object] = {
        "unique_corpora": len(structure_cache),
        "stage1_candidate_limit": stage1_candidates,
        "stage1_candidate_policy": (
            f"fixed:{stage1_candidates}"
            if stage1_candidates is not None
            else "adaptive:max(16,min(64,ceil(0.10*structural_units)))"
        ),
        "avg_stage1_selected_units": (
            sum(selected_stage1_counts) / max(1, len(selected_stage1_counts))
        ),
        "avg_candidate_source_documents": (
            sum(candidate_source_counts) / max(1, len(candidate_source_counts))
        ),
        "avg_stage1_evidence_coverage": (
            sum(candidate_evidence_coverages) / max(1, len(candidate_evidence_coverages))
        ),
        "avg_raw_documents": avg_stat("raw_document_count") if stat_rows else 0.0,
        "avg_structural_units": avg_stat("segment_count") if stat_rows else 0.0,
        "avg_progressions": avg_stat("progression_count") if stat_rows else 0.0,
        "avg_raw_chars": sum(raw_char_counts) / max(1, len(raw_char_counts)),
        "avg_index_chars": sum(index_char_counts) / max(1, len(index_char_counts)),
        "avg_index_to_raw_char_ratio": (
            sum(index_char_counts) / max(1, sum(raw_char_counts))
        ),
    }
    if stat_rows:
        stats["avg_segment_size"] = avg_stat("avg_segment_size")
        stats["avg_single_event_segment_fraction"] = avg_stat(
            "single_event_segment_fraction"
        )
    return transformed, stats
