from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from progression_memory.evaluation.retrieval import RetrievalDocument, tokenize
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.routing.router import HeuristicProgressionRouter, RouteAction, cosine_similarity
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.segmenter import AdaptiveTemporalSegmenter, FixedWindowTemporalSegmenter

_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_BASE_TIME = datetime(2000, 1, 1)


def hashed_embedding(text: str, dim: int = 128) -> list[float]:
    vector = np.zeros(dim, dtype=float)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        idx = value % dim
        sign = -1.0 if (value >> 63) else 1.0
        vector[idx] += sign
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.tolist()


def extract_entities(text: str) -> list[str]:
    return sorted(set(_ENTITY_RE.findall(text)))[:24]


def parse_timestamp(raw: str | None, ordinal: int) -> datetime:
    if raw:
        value = raw.strip().replace("Sept ", "Sep ")
        for candidate in (value, value.split(" on ")[-1]):
            candidate = candidate.strip()
            try:
                return datetime.fromisoformat(candidate.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
            for fmt in (
                "%d %B, %Y", "%d %b, %Y", "%B %d, %Y", "%b %d, %Y",
                "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
                "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
            ):
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
    return _BASE_TIME + timedelta(days=ordinal)


def _refresh_progression_embedding(store: ProgressionStore, progression_id: str) -> None:
    progression = store.progressions[progression_id]
    vectors = [
        store.episodes[eid].embedding
        for eid in progression.episode_ids
        if store.episodes[eid].embedding is not None
    ]
    if vectors:
        progression.embedding = np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()
    entities: set[str] = set()
    for eid in progression.episode_ids[-32:]:
        entities.update(store.episodes[eid].entities)
    progression.scope = " ".join(sorted(entities)[:32])
    if progression.latest_episode_id:
        progression.current_state = store.episodes[progression.latest_episode_id].summary


def build_progression_documents(
    documents: tuple[RetrievalDocument, ...],
    *,
    mode: Literal["flat", "fixed", "adaptive"],
    attach_threshold: float = 0.62,
) -> tuple[tuple[RetrievalDocument, ...], dict[str, float | int]]:
    """Route raw memories into progressions and expose evidence-backed retrieval units."""

    store = ProgressionStore()
    router = HeuristicProgressionRouter(attach_threshold=attach_threshold)
    if mode == "flat":
        # One segment per progression isolates progression routing from temporal segmentation.
        segmenter = FixedWindowTemporalSegmenter(window=timedelta(days=365000))
    elif mode == "fixed":
        segmenter = FixedWindowTemporalSegmenter(window=timedelta(days=14))
    else:
        segmenter = AdaptiveTemporalSegmenter()
    source_by_episode: dict[str, RetrievalDocument] = {}
    created = 0

    for ordinal, document in enumerate(documents):
        embedding = hashed_embedding(document.text)
        state = (np.asarray(embedding, dtype=float) * 0.5).tolist()
        candidates = list(store.progressions.values())
        if candidates:
            best_semantic = max(
                (cosine_similarity(embedding, p.embedding) + 1.0) / 2.0
                for p in candidates
            )
            novelty = max(0.0, min(1.0, 1.0 - best_semantic))
        else:
            novelty = 1.0
        episode = Episode(
            id=f"episode:{ordinal}:{document.id}",
            summary=document.text,
            dialogue_time=parse_timestamp(document.timestamp, ordinal),
            embedding=embedding,
            state_vector=state,
            entities=extract_entities(document.text),
            importance=0.5,
            novelty=novelty,
            source_pointer=document.id,
            metadata=document.metadata,
        )
        store.ingest_episode(episode)
        source_by_episode[episode.id] = document
        decision = router.route(episode, candidates)

        if decision.action == RouteAction.CREATE:
            progression_id = f"progression:{created}"
            created += 1
            progression = Progression(
                id=progression_id,
                title=" ".join(tokenize(document.text)[:6]) or progression_id,
                scope=" ".join(episode.entities),
                embedding=embedding,
                current_state=document.text,
            )
            store.upsert_progression(progression)
            segmenter.process(store, episode.id, progression_id)
            _refresh_progression_embedding(store, progression_id)
        else:
            for progression_id in decision.progression_ids:
                segmenter.process(store, episode.id, progression_id)
                _refresh_progression_embedding(store, progression_id)

    output: list[RetrievalDocument] = []
    for progression in store.progressions.values():
        for segment_id in progression.segment_ids:
            segment = store.segments[segment_id]
            sources = [source_by_episode[eid] for eid in segment.episode_ids]
            evidence_ids: set[str] = set()
            for source in sources:
                evidence_ids.update(source.evidence_ids or {source.id})
            output.append(
                RetrievalDocument(
                    id=segment.id,
                    text="\n".join(source.text for source in sources),
                    evidence_ids=frozenset(evidence_ids),
                    timestamp=sources[-1].timestamp if sources else None,
                    metadata={
                        "progression_id": progression.id,
                        "segment_size": len(segment.episode_ids),
                        "mode": mode,
                    },
                )
            )

    raw_chars = sum(len(doc.text) for doc in documents)
    segment_chars = sum(len(doc.text) for doc in output)
    stats: dict[str, float | int] = {
        "raw_document_count": len(documents),
        "progression_count": len(store.progressions),
        "segment_count": len(output),
        "document_compression_ratio": len(documents) / max(1, len(output)),
        "raw_chars": raw_chars,
        "segment_chars": segment_chars,
        "evidence_duplication_ratio": segment_chars / max(1, raw_chars),
    }
    return tuple(output), stats
