"""Hierarchy-aware progression retrieval with lexical, dense, and temporal ablations.

Dense similarity requires an injected embedding function. The default zero-dense
configuration is explicit: it never mislabels token overlap as neural embeddings.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.hierarchy import HierarchyIndex
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine

Mode = Literal[
    "flat_bm25", "flat_hybrid", "hierarchical_bm25",
    "hierarchical_hybrid", "hierarchical_hybrid_temporal",
]
Embedder = Callable[[str], list[float]]
MODES: tuple[Mode, ...] = (
    "flat_bm25", "flat_hybrid", "hierarchical_bm25",
    "hierarchical_hybrid", "hierarchical_hybrid_temporal",
)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding dimensions mismatch or empty")
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denom if denom else 0.0


def _rank(
    query: str, documents: list[RetrievalDocument], *,
    hybrid: bool, embedder: Embedder | None,
) -> list[RetrievalDocument]:
    if not documents:
        return []
    if hybrid and embedder is None:
        raise ValueError("hybrid retrieval needs a genuine embedding callback")
    lexical = [doc for _, doc in BM25Index(documents).score(query)]
    if not hybrid:
        return lexical
    query_vector = embedder(query)
    dense = sorted(
        documents,
        key=lambda doc: (-_cosine(query_vector, embedder(doc.text)), doc.id),
    )
    ranks: dict[str, float] = {}
    for order in (lexical, dense):
        for position, doc in enumerate(order, 1):
            ranks[doc.id] = ranks.get(doc.id, 0.0) + 1.0 / (60 + position)
    return sorted(documents, key=lambda doc: (-ranks[doc.id], doc.id))


def _progression_id(document: RetrievalDocument) -> str | None:
    value = document.metadata.get("progression_id")
    return str(value) if value is not None else None


def retrieve(
    engine: StateEngine, query: str, *,
    mode: Mode = "hierarchical_bm25", k: int = 8,
    subject_id: str | None = None,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    embedder: Embedder | None = None,
    candidate_count: int = 4,
    global_rescue: int = 2,
) -> list[RetrievalDocument]:
    """Rank progressions then select state/evidence from routed branches.

    Global rescue slots remain available even when branch routing misses.
    Time-bound mode uses the same as-of filter for states and observations.
    """
    if mode not in MODES or k < 0 or candidate_count < 1 or global_rescue < 0:
        raise ValueError("invalid retrieval configuration")
    if mode == "hierarchical_hybrid_temporal" and valid_at is None:
        raise ValueError("temporal mode requires valid_at")
    if valid_at is not None and (valid_at.tzinfo is None or valid_at.utcoffset() is None):
        raise ValueError("valid_at must be timezone-aware")
    if known_at is not None and (known_at.tzinfo is None or known_at.utcoffset() is None):
        raise ValueError("known_at must be timezone-aware")
    hybrid = mode in ("flat_hybrid", "hierarchical_hybrid",
                      "hierarchical_hybrid_temporal")
    docs = list(build_documents(
        engine, view="full", subject_id=subject_id,
        valid_at=valid_at, known_at=known_at,
    ))
    if not docs or not k:
        return []
    if mode.startswith("flat"):
        return _rank(query, docs, hybrid=hybrid, embedder=embedder)[:k]
    hierarchy = HierarchyIndex(engine.progressions)
    state_docs = [d for d in docs if _progression_id(d) is not None]
    evidence_docs = [d for d in docs if _progression_id(d) is None]
    # No canonical progression found: do not lose ungrouped evidence.
    if not state_docs:
        return _rank(query, evidence_docs, hybrid=hybrid, embedder=embedder)[:k]
    ranked_states = _rank(query, state_docs, hybrid=hybrid, embedder=embedder)
    winners = {_progression_id(d) for d in ranked_states[:candidate_count]}
    branch_ids = set(winners)
    for pid in winners:
        if pid is not None:
            branch_ids.update(hierarchy.ancestors(pid))
            branch_ids.update(hierarchy.descendants(pid))
    source_ids = {
        delta.observation_id for delta in engine.deltas.values()
        if delta.progression_id in branch_ids
    }
    routed = [
        d for d in docs if (
            _progression_id(d) in branch_ids
            if _progression_id(d) is not None
            else bool(d.evidence_ids & source_ids)
        )
    ]
    # Route first, then rank within the branch, reserving rescue capacity.
    routed_ranked = _rank(query, routed, hybrid=hybrid, embedder=embedder)
    rescue_ranked = _rank(
        query, [d for d in docs if d.id not in {r.id for r in routed}],
        hybrid=hybrid, embedder=embedder,
    )
    reserved = min(global_rescue, max(0, k - 1))
    result = routed_ranked[:max(0, k - reserved)]
    result.extend(rescue_ranked[:k - len(result)])
    if len(result) < k:
        result.extend(d for d in routed_ranked if d.id not in {r.id for r in result})
    return result[:k]


def select_budget(
    documents: list[RetrievalDocument], budget: int,
) -> list[RetrievalDocument]:
    """Whole-document prefix budgeting, identical to existing paired evaluator."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    selected: list[RetrievalDocument] = []
    used = 0
    for doc in documents:
        if used + len(doc.text) > budget:
            break
        selected.append(doc)
        used += len(doc.text)
    return selected
