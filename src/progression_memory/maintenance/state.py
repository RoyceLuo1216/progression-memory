from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression, ProgressionStateCheckpoint
from progression_memory.routing.router import cosine_similarity


class MaintenanceAction(StrEnum):
    EXTEND = "extend"
    TRANSITION = "transition"
    CREATE_CHILD = "create_child"
    CREATE = "create"
    MERGE_CANDIDATE = "merge_candidate"


@dataclass(frozen=True)
class MaintenanceDecision:
    action: MaintenanceAction
    compatibility: float
    semantic_similarity: float
    state_similarity: float
    contradiction: float


def _unit(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm else vector).tolist()


def condition_progression_state(
    progression: Progression,
    episode: Episode,
    *,
    prior_weight: float = 0.7,
) -> ProgressionStateCheckpoint:
    """Update Z_t = f(Z_{t-1}, E_t) while preserving the complete state trajectory.

    The embedding update is a normalized Bayesian-filter-like recursive update: the
    previous state acts as a prior and new evidence supplies the innovation. It is not
    presented as literal Naive Bayes; no conditional-independence assumption is made.
    """
    evidence_embedding = np.asarray(episode.embedding or episode.state_vector or [], dtype=float)
    prior_embedding = np.asarray(progression.state_embedding or progression.embedding or [], dtype=float)
    if evidence_embedding.size and prior_embedding.size == evidence_embedding.size:
        posterior_embedding = _unit(prior_weight * prior_embedding + (1.0 - prior_weight) * evidence_embedding)
        delta = (np.asarray(posterior_embedding) - prior_embedding).tolist()
    elif evidence_embedding.size:
        posterior_embedding = _unit(evidence_embedding)
        delta = posterior_embedding
    else:
        posterior_embedding = progression.state_embedding
        delta = None

    prior_state = progression.current_state
    # Keep the state evidence-grounded and cumulative. A learned/LLM state updater can
    # replace this deterministic reducer without changing the state-history contract.
    posterior_state = episode.summary if not prior_state else f"{prior_state}\n→ {episode.summary}"
    checkpoint = ProgressionStateCheckpoint(
        episode_id=episode.id,
        timestamp=episode.dialogue_time,
        prior_state=prior_state,
        evidence=episode.summary,
        posterior_state=posterior_state,
        prior_embedding=progression.state_embedding or progression.embedding,
        posterior_embedding=posterior_embedding,
        delta_embedding=delta,
        confidence=float(episode.metadata.get("state_confidence", 1.0)),
    )
    progression.current_state = posterior_state
    progression.state_embedding = posterior_embedding
    progression.state_history.append(checkpoint)
    return checkpoint


def maintenance_decision(
    episode: Episode,
    progression: Progression,
    *,
    contradiction: float = 0.0,
) -> MaintenanceDecision:
    semantic = (cosine_similarity(episode.embedding, progression.embedding) + 1.0) / 2.0
    state = (cosine_similarity(episode.state_vector, progression.state_embedding or progression.embedding) + 1.0) / 2.0
    continuity = max(0.0, min(1.0, progression.continuity or 0.5))
    compatibility = 0.45 * semantic + 0.35 * state + 0.20 * continuity - 0.25 * contradiction

    if compatibility >= 0.72 and contradiction < 0.45:
        action = MaintenanceAction.EXTEND
    elif semantic >= 0.62 and (state < 0.55 or contradiction >= 0.45):
        action = MaintenanceAction.TRANSITION
    elif semantic >= 0.55:
        action = MaintenanceAction.CREATE_CHILD
    else:
        action = MaintenanceAction.CREATE
    return MaintenanceDecision(action, compatibility, semantic, state, contradiction)


def merge_compatibility(left: Progression, right: Progression) -> float:
    """Trajectory-aware merge signal; callers should require persistence across updates."""
    semantic = (cosine_similarity(left.embedding, right.embedding) + 1.0) / 2.0
    state = (cosine_similarity(left.state_embedding, right.state_embedding) + 1.0) / 2.0
    return 0.55 * semantic + 0.45 * state
