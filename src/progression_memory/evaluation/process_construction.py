"""Process-aware construction over the canonical progression store."""
from __future__ import annotations

import math
from collections.abc import Sequence

from progression_memory.evaluation.progressions import parse_timestamp
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.evaluation.semantic import TextEmbedder, cosine_scores
from progression_memory.maintenance.process_writer import ProcessWriter
from progression_memory.maintenance.state import condition_progression_state
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.segmenter import AdaptiveTemporalSegmenter


def _speaker(text: str) -> str:
    if ":" not in text:
        return ""
    value = text.split(":", 1)[0].strip()
    return value if value and len(value) <= 120 else ""


def _candidate_rows(
    text: str,
    now,
    progressions: Sequence[Progression],
    embedder: TextEmbedder,
    *,
    candidate_k: int,
) -> list[dict]:
    if not progressions:
        return []
    query = embedder.encode([text])[0]
    identities = [f"{p.scope}: {p.title}" for p in progressions]
    states = [p.current_state or p.title for p in progressions]
    facts = [p.state_history[-1].evidence if p.state_history else p.title for p in progressions]
    views = embedder.encode(identities + states + facts)
    n = len(progressions)
    identity_scores = cosine_scores(query, views[:n])
    state_scores = cosine_scores(query, views[n:2 * n])
    fact_scores = cosine_scores(query, views[2 * n:])
    source_speaker = _speaker(text).casefold()
    rows = []
    for progression, identity, state, fact in zip(
        progressions, identity_scores, state_scores, fact_scores, strict=True
    ):
        age = (
            abs((now - progression.last_updated_time).total_seconds()) / 86400
            if progression.last_updated_time else 0.0
        )
        recency = math.exp(-math.log(2) * age / 90)
        subject = float(bool(source_speaker) and source_speaker == progression.scope.casefold())
        score = (
            0.35 * float(identity)
            + 0.25 * float(state)
            + 0.20 * float(fact)
            + 0.10 * subject
            + 0.10 * recency
        )
        rows.append({
            "id": progression.id,
            "subject": progression.scope,
            "process": progression.title,
            "state": progression.current_state,
            "recent_fact": facts[len(rows)],
            "score": round(score, 4),
            "signals": {
                "identity": round(float(identity), 4),
                "state": round(float(state), 4),
                "recent_fact": round(float(fact), 4),
                "subject": subject,
                "recency": round(recency, 4),
            },
        })
    return sorted(rows, key=lambda row: (-row["score"], row["id"]))[:candidate_k]


def build_process_documents(
    documents: tuple[RetrievalDocument, ...],
    writer: ProcessWriter,
    embedder: TextEmbedder,
    *,
    candidate_k: int = 6,
):
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    store = ProgressionStore()
    segmenter = AdaptiveTemporalSegmenter()
    units: list[RetrievalDocument] = []
    previous: list[dict] = []

    for ordinal, document in enumerate(documents):
        now = parse_timestamp(document.timestamp, ordinal)
        embedding = embedder.encode([document.text])[0].tolist()
        episode = Episode(
            id=document.id,
            summary=document.text,
            dialogue_time=now,
            embedding=embedding,
            state_vector=embedding,
            source_pointer=document.id,
            metadata=dict(document.metadata),
        )
        store.ingest_episode(episode)
        candidates = _candidate_rows(
            document.text,
            now,
            list(store.progressions.values()),
            embedder,
            candidate_k=candidate_k,
        )
        source = {"id": document.id, "text": document.text, "timestamp": document.timestamp}
        updates = writer.decide(source, previous[-3:], candidates)
        state_vectors = (
            embedder.encode([u.state for u in updates] + [u.delta for u in updates])
            if updates else []
        )

        for index, update in enumerate(updates):
            if update.action == "create":
                identity = (update.subject.casefold(), update.process.casefold())
                if any(
                    (p.scope.casefold(), p.title.casefold()) == identity
                    for p in store.progressions.values()
                ):
                    raise ValueError("writer created duplicate process identity")
                progression_id = f"process:{len(store.progressions)}"
                identity_embedding = embedder.encode(
                    [f"{update.subject}: {update.process}"]
                )[0].tolist()
                progression = Progression(
                    id=progression_id,
                    title=update.process,
                    scope=update.subject,
                    high_level_summary=f"{update.subject}: {update.process}",
                    embedding=identity_embedding,
                )
                store.upsert_progression(progression)
            else:
                progression_id = str(update.progression_id)
                progression = store.progressions[progression_id]

            prior_state = progression.current_state
            if update.action == "transition" and progression.segment_ids:
                store.attach_episode(episode.id, progression_id)
                segment = segmenter._create_segment(store, episode.id, progression_id)
            else:
                _, segment = segmenter.process(store, episode.id, progression_id)
            checkpoint = condition_progression_state(
                progression,
                episode,
                posterior_state=update.state,
                state_delta=update.delta,
                evidence=update.fact,
                action=update.action,
                posterior_embedding=state_vectors[index].tolist(),
                delta_embedding=state_vectors[len(updates) + index].tolist(),
            )
            segment.state_before = prior_state
            segment.state_after = checkpoint.posterior_state
            segment.delta = checkpoint.state_delta
            segment.state_embedding = checkpoint.posterior_embedding
            segment.delta_embedding = checkpoint.delta_embedding
            units.append(RetrievalDocument(
                id=f"{progression_id}:update:{len(progression.state_history)}",
                text=(f"{progression.scope}: {progression.title}\n{update.fact}\n"
                      f"State: {update.state}\nChange: {update.delta}"),
                timestamp=document.timestamp,
                metadata={
                    "progression_id": progression_id,
                    "segment_id": segment.id,
                    "source_documents": (document,),
                },
            ))

        if not updates:
            units.append(RetrievalDocument(
                id=f"unassigned:{document.id}",
                text=document.text,
                timestamp=document.timestamp,
                metadata={"source_documents": (document,)},
            ))
        previous.append(source)

    sizes = [len(set(p.episode_ids)) for p in store.progressions.values()]
    assignments = {episode_id for p in store.progressions.values() for episode_id in p.episode_ids}
    stats = {
        "raw_document_count": len(documents),
        "progression_count": len(store.progressions),
        "segment_count": len(store.segments),
        "assigned_source_count": len(assignments),
        "unassigned_source_count": len(documents) - len(assignments),
        "largest_progression_fraction": max(sizes, default=0) / max(1, len(documents)),
        "singleton_progression_fraction": sum(size == 1 for size in sizes) / max(1, len(sizes)),
        "source_membership_count": sum(sizes),
        "membership_duplication_ratio": sum(sizes) / max(1, len(assignments)),
        "structural_units": len(units),
    }
    return tuple(units), store.progressions, stats
