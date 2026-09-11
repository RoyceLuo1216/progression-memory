from __future__ import annotations

import hashlib
from math import prod

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.models.temporal_segment import TemporalSegment


class ProgressionStore:
    """In-memory v1 store with evidence-preserving, idempotent semantics."""

    def __init__(self) -> None:
        self.progressions: dict[str, Progression] = {}
        self.episodes: dict[str, Episode] = {}
        self.segments: dict[str, TemporalSegment] = {}

    def upsert_progression(self, progression: Progression) -> None:
        self.progressions[progression.id] = progression

    def upsert_segment(self, segment: TemporalSegment) -> None:
        progression = self.progressions[segment.progression_id]
        self.segments[segment.id] = segment
        if segment.id not in progression.segment_ids:
            progression.segment_ids.append(segment.id)

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
            self._recompute_progression_statistics(progression)

    def segment_for_episode(
        self,
        progression_id: str,
        episode_id: str,
    ) -> TemporalSegment | None:
        progression = self.progressions[progression_id]
        matches = [
            self.segments[segment_id]
            for segment_id in progression.segment_ids
            if episode_id in self.segments[segment_id].episode_ids
        ]
        if len(matches) > 1:
            raise ValueError(
                f"episode {episode_id!r} belongs to multiple finest-level segments "
                f"for progression {progression_id!r}"
            )
        return matches[0] if matches else None

    def validate_segment_partition(self, progression_id: str) -> bool:
        progression = self.progressions[progression_id]
        assigned: list[str] = []
        for segment_id in progression.segment_ids:
            assigned.extend(self.segments[segment_id].episode_ids)
        return sorted(assigned) == sorted(progression.episode_ids) and len(assigned) == len(
            set(assigned)
        )

    def _latest_episode_id(self, progression: Progression) -> str | None:
        if not progression.episode_ids:
            return None
        return max(
            progression.episode_ids,
            key=lambda episode_id: self.episodes[episode_id].event_time,
        )

    def _recompute_progression_statistics(self, progression: Progression) -> None:
        episodes = [self.episodes[episode_id] for episode_id in progression.episode_ids]
        times = [episode.event_time for episode in episodes]
        progression.start_time = min(times)
        progression.last_updated_time = max(times)
        progression.importance = sum(episode.importance for episode in episodes) / len(episodes)
        progression.contradiction_rate = (
            sum(episode.contradiction for episode in episodes) / len(episodes)
        )

        span_days = max(
            1.0,
            (progression.last_updated_time - progression.start_time).total_seconds() / 86400.0,
        )
        progression.event_frequency = len(episodes) / span_days

        contributions = [
            min(0.95, max(0.05, 0.25 + 0.5 * episode.importance))
            for episode in episodes
        ]
        progression.persistence = 1.0 - prod(
            1.0 - contribution for contribution in contributions
        )

    @staticmethod
    def _evidence_version(progression: Progression) -> str:
        canonical = "|".join(sorted(set(progression.episode_ids)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
