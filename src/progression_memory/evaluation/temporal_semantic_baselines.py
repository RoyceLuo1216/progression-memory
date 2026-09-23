"""Comparable temporal and semantic grouping baselines.

These baselines intentionally contain no progression state, routing heuristic, or
hierarchy.  They answer a narrower question: how much retrieval signal comes from
time alone, semantic similarity alone, or the two signals together?
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

import numpy as np

from progression_memory.evaluation.progressions import hashed_embedding, parse_timestamp
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.evaluation.semantic import TextEmbedder, cosine_scores

BaselineMode = Literal["temporal_only", "semantic_only", "temporal_semantic"]


def _group_document(group: list[RetrievalDocument], mode: str, index: int) -> RetrievalDocument:
    evidence: set[str] = set()
    for document in group:
        evidence.update(document.evidence_ids or {document.id})
    return RetrievalDocument(
        id=f"{mode}:unit:{index}",
        text="\n".join(document.text for document in group),
        evidence_ids=frozenset(evidence),
        timestamp=group[-1].timestamp,
        metadata={
            "baseline": mode,
            "source_documents": tuple(group),
            "unit_size": len(group),
        },
    )


def _time_values(documents: tuple[RetrievalDocument, ...]) -> list:
    return [parse_timestamp(document.timestamp, index) for index, document in enumerate(documents)]


def _semantic_values(
    documents: tuple[RetrievalDocument, ...], embedder: TextEmbedder | None,
) -> np.ndarray:
    if embedder is not None:
        return np.asarray(embedder.encode([document.text for document in documents]), dtype=np.float32)
    return np.asarray([hashed_embedding(document.text) for document in documents], dtype=np.float32)


def build_baseline_documents(
    documents: tuple[RetrievalDocument, ...],
    mode: BaselineMode,
    *,
    temporal_window_days: float = 14.0,
    semantic_threshold: float = 0.58,
    embedder: TextEmbedder | None = None,
) -> tuple[tuple[RetrievalDocument, ...], dict[str, float | int | str]]:
    """Build one of the controlled non-progression grouping baselines.

    Documents are processed in their source order. ``temporal_only`` closes a unit
    when the gap since its last source exceeds the window. ``semantic_only`` attaches to the
    closest existing semantic centroid when it clears the threshold. The combined
    baseline requires both conditions, so neither signal can silently override the
    other.
    """
    if mode not in {"temporal_only", "semantic_only", "temporal_semantic"}:
        raise ValueError(f"unsupported baseline mode: {mode}")
    if temporal_window_days <= 0:
        raise ValueError("temporal_window_days must be positive")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if not documents:
        return (), {
            "raw_document_count": 0, "baseline_unit_count": 0,
            "avg_unit_size": 0.0, "avg_document_compression_ratio": 0.0,
            "semantic_backend": "none",
        }

    times = _time_values(documents)
    vectors = _semantic_values(documents, embedder) if mode != "temporal_only" else None
    window = timedelta(days=temporal_window_days)
    groups: list[list[RetrievalDocument]] = []
    centroids: list[np.ndarray] = []
    last_times: list = []
    for index, document in enumerate(documents):
        if not groups:
            groups.append([document]); last_times.append(times[index])
            if vectors is not None: centroids.append(vectors[index].copy())
            continue
        group_index = len(groups) - 1
        if mode == "semantic_only":
            scores = cosine_scores(vectors[index], np.asarray(centroids, dtype=np.float32))
            group_index = int(np.argmax(scores))
            should_attach = float(scores[group_index]) >= semantic_threshold
        else:
            temporal_ok = times[index] - last_times[group_index] <= window
            semantic_ok = True
            if vectors is not None:
                semantic_ok = float(cosine_scores(vectors[index], centroids[group_index][None, :])[0]) >= semantic_threshold
            should_attach = temporal_ok if mode == "temporal_only" else temporal_ok and semantic_ok
        if should_attach:
            groups[group_index].append(document)
            last_times[group_index] = times[index]
            if vectors is not None:
                size = len(groups[group_index])
                centroids[group_index] += (vectors[index] - centroids[group_index]) / size
        else:
            groups.append([document]); last_times.append(times[index])
            if vectors is not None: centroids.append(vectors[index].copy())

    output = tuple(_group_document(group, mode, index) for index, group in enumerate(groups))
    sizes = [len(group) for group in groups]
    raw_chars = sum(len(document.text) for document in documents)
    unit_chars = sum(len(document.text) for document in output)
    return output, {
        "raw_document_count": len(documents),
        "baseline_unit_count": len(output),
        "avg_unit_size": sum(sizes) / len(sizes),
        "avg_document_compression_ratio": len(documents) / max(1, len(output)),
        "raw_chars": raw_chars,
        "unit_chars": unit_chars,
        "evidence_duplication_ratio": unit_chars / max(1, raw_chars),
        "semantic_backend": getattr(embedder, "model_name", "hashed") if vectors is not None else "none",
        "temporal_window_days": temporal_window_days,
        "semantic_threshold": semantic_threshold,
    }
