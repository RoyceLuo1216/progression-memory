from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from progression_memory.evaluation.retrieval import RetrievalDocument, tokenize
from progression_memory.evaluation.semantic import TextEmbedder
from progression_memory.maintenance.manager import HeuristicProgressionManager, ProgressionManager
from progression_memory.maintenance.state import MaintenanceAction, condition_progression_state
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.routing.router import HeuristicProgressionRouter, RouteAction, cosine_similarity
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.controller import HysteresisLinearTemporalController
from progression_memory.temporal.segmenter import AdaptiveTemporalSegmenter, FixedWindowTemporalSegmenter

_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_BASE_TIME = datetime(2000, 1, 1)


def hashed_embedding(text: str, dim: int = 128) -> list[float]:
    """Legacy lexical hash representation retained as an explicit ablation baseline."""
    vector = np.zeros(dim, dtype=float)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        idx = value % dim
        vector[idx] += -1.0 if (value >> 63) else 1.0
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
            for fmt in ("%d %B, %Y", "%d %b, %Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
    return _BASE_TIME + timedelta(days=ordinal)


def _refresh_progression_embedding(store: ProgressionStore, progression_id: str) -> None:
    """Refresh the topic embedding without overwriting the recursively conditioned state."""
    progression = store.progressions[progression_id]
    vectors = [store.episodes[eid].embedding for eid in progression.episode_ids if store.episodes[eid].embedding is not None]
    if vectors:
        progression.embedding = np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()
    entities: set[str] = set()
    for eid in progression.episode_ids[-32:]:
        entities.update(store.episodes[eid].entities)
    progression.scope = " ".join(sorted(entities)[:32])


def build_progression_documents(
    documents: tuple[RetrievalDocument, ...], *,
    mode: Literal["flat", "fixed", "adaptive", "adaptive_v2"],
    attach_threshold: float = 0.62,
    embedder: TextEmbedder | None = None,
    manager: ProgressionManager | None = None,
) -> tuple[tuple[RetrievalDocument, ...], dict[str, float | int | str]]:
    """Route memories into evolving progressions with evidence-conditioned state updates."""
    store = ProgressionStore()
    router = HeuristicProgressionRouter(attach_threshold=attach_threshold)
    manager = manager or HeuristicProgressionManager()
    if mode == "flat":
        segmenter = FixedWindowTemporalSegmenter(window=timedelta(days=365000))
    elif mode == "fixed":
        segmenter = FixedWindowTemporalSegmenter(window=timedelta(days=14))
    elif mode == "adaptive_v2":
        segmenter = AdaptiveTemporalSegmenter(HysteresisLinearTemporalController())
    else:
        segmenter = AdaptiveTemporalSegmenter()
    source_by_episode: dict[str, RetrievalDocument] = {}
    created = 0
    semantic_embeddings = embedder.encode([doc.text for doc in documents]) if embedder is not None else None

    for ordinal, document in enumerate(documents):
        embedding = semantic_embeddings[ordinal].tolist() if semantic_embeddings is not None else hashed_embedding(document.text)
        state = embedding.copy()
        candidates = list(store.progressions.values())
        if candidates:
            best_semantic = max((cosine_similarity(embedding, p.embedding) + 1.0) / 2.0 for p in candidates)
            novelty = max(0.0, min(1.0, 1.0 - best_semantic))
        else:
            novelty = 1.0
        metadata = dict(document.metadata)
        if mode == "adaptive_v2":
            metadata["state_confidence"] = min(1.0, max(0.25, len(tokenize(document.text)) / 12.0))
        episode = Episode(
            id=f"episode:{ordinal}:{document.id}", summary=document.text,
            dialogue_time=parse_timestamp(document.timestamp, ordinal), embedding=embedding,
            state_vector=state, entities=extract_entities(document.text), importance=0.5,
            novelty=novelty, source_pointer=document.id, metadata=metadata,
        )
        store.ingest_episode(episode)
        source_by_episode[episode.id] = document
        decision = manager.decide(episode, candidates) if embedder is not None else router.route(episode, candidates)
        action = decision.action
        if action in (MaintenanceAction.CREATE, RouteAction.CREATE):
            progression_id = f"progression:{created}"
            created += 1
            progression = Progression(
                id=progression_id, title=" ".join(tokenize(document.text)[:6]) or progression_id,
                scope=" ".join(episode.entities), embedding=embedding,
            )
            store.upsert_progression(progression)
            segmenter.process(store, episode.id, progression_id)
            condition_progression_state(progression, episode)
            _refresh_progression_embedding(store, progression_id)
        else:
            progression_ids = ([decision.progression_id] if hasattr(decision, "progression_id") else decision.progression_ids)
            for progression_id in filter(None, progression_ids):
                progression = store.progressions[progression_id]
                before = progression.current_state
                action_segment, segment = segmenter.process(store, episode.id, progression_id)
                checkpoint = condition_progression_state(progression, episode, posterior_state=getattr(decision, "updated_state", None))
                segment.state_before = before if not segment.state_before else segment.state_before
                segment.state_after = checkpoint.posterior_state
                segment.delta = getattr(decision, "state_delta", None) or episode.summary
                segment.state_embedding = checkpoint.posterior_embedding
                segment.delta_embedding = checkpoint.delta_embedding
                if action == MaintenanceAction.TRANSITION and action_segment.value != "split":
                    # Model-declared transitions are semantic boundaries; move this episode
                    # from the old segment into a fresh checkpoint segment.
                    old = segment
                    if episode.id in old.episode_ids: old.episode_ids.remove(episode.id)
                    segment = segmenter._create_segment(store, episode.id, progression_id)
                    segment.state_before, segment.state_after = before, checkpoint.posterior_state
                    segment.delta = getattr(decision, "state_delta", None) or episode.summary
                    segment.state_embedding, segment.delta_embedding = checkpoint.posterior_embedding, checkpoint.delta_embedding
                _refresh_progression_embedding(store, progression_id)

    output: list[RetrievalDocument] = []
    segment_sizes: list[int] = []
    state_updates: list[int] = []
    for progression in store.progressions.values():
        state_updates.append(len(progression.state_history))
        for segment_id in progression.segment_ids:
            segment = store.segments[segment_id]
            sources = [source_by_episode[eid] for eid in segment.episode_ids]
            segment_sizes.append(len(segment.episode_ids))
            evidence_ids: set[str] = set()
            for source in sources:
                evidence_ids.update(source.evidence_ids or {source.id})
            output.append(RetrievalDocument(
                id=segment.id, text="\n".join(source.text for source in sources),
                evidence_ids=frozenset(evidence_ids), timestamp=sources[-1].timestamp if sources else None,
                metadata={"progression_id": progression.id, "segment_size": len(segment.episode_ids), "mode": mode,
                          "embedding_backend": getattr(embedder, "model_name", "hashed") if embedder else "hashed",
                          "progression_state": progression.current_state,
                          "state_update_count": len(progression.state_history)},
            ))

    raw_chars = sum(len(doc.text) for doc in documents)
    segment_chars = sum(len(doc.text) for doc in output)
    stats: dict[str, float | int | str] = {
        "raw_document_count": len(documents), "progression_count": len(store.progressions), "segment_count": len(output),
        "document_compression_ratio": len(documents) / max(1, len(output)), "raw_chars": raw_chars,
        "segment_chars": segment_chars, "evidence_duplication_ratio": segment_chars / max(1, raw_chars),
        "avg_segment_size": sum(segment_sizes) / max(1, len(segment_sizes)),
        "single_event_segment_fraction": sum(size == 1 for size in segment_sizes) / max(1, len(segment_sizes)),
        "embedding_backend": getattr(embedder, "model_name", "hashed") if embedder else "hashed",
        "avg_state_updates_per_progression": sum(state_updates) / max(1, len(state_updates)),
    }
    return tuple(output), stats
