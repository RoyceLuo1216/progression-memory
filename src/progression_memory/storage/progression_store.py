from __future__ import annotations

import hashlib

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression


class ProgressionStore:
    """In-memory v1 store with idempotent episode attachment semantics."""

    def __init__(self) -> None:
        self.progressions: dict[str, Progression] = {}
        self.episodes: dict[str, Episode] = {}

    def upsert_progression(self, progression: Progression) -> None:
        self.progressions[progression.id] = progression

    def ingest_episode(self, episode: Episode) -> bool:
        """Insert an episode once. Returns False for duplicate episode IDs."""
        if episode.id in self.episodes:
            return False
        self.episodes[episode.id] = episode
        return True

    def attach_episode(self, episode_id: str, progression_id: str) -> None:
        episode = self.episodes[episode_id]
        progression = self.progressions[progression_id]

        if progression_id not in episode.progression_ids:
            episode.progression_ids.append(progression_id)

        if episode_id not in progression.episode_ids:
            progression.episode_ids.append(episode_id)
            progression.latest_episode_id = self._latest_episode_id(progression)
            progression.evidence_version = self._evidence_version(progression)

    def _latest_episode_id(self, progression: Progression) -> str | None:
        if not progression.episode_ids:
            return None
        return max(
            progression.episode_ids,
            key=lambda episode_id: self.episodes[episode_id].semantic_time
            or self.episodes[episode_id].dialogue_time,
        )

    @staticmethod
    def _evidence_version(progression: Progression) -> str:
        canonical = "|".join(sorted(set(progression.episode_ids)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
