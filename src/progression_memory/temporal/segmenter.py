from __future__ import annotations

from datetime import timedelta
from typing import Protocol

import numpy as np

from progression_memory.models.temporal_segment import TemporalSegment
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.controller import (
    RuleBasedTemporalController,
    SegmentAction,
    SegmentFeatures,
    extract_segment_features,
)


class TemporalController(Protocol):
    def decide(self, features: SegmentFeatures) -> SegmentAction: ...


def _mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    array = np.asarray(vectors, dtype=float)
    return np.mean(array, axis=0).tolist()


def _refresh_segment_state(store: ProgressionStore, segment: TemporalSegment) -> None:
    episodes = [store.episodes[episode_id] for episode_id in segment.episode_ids]
    embeddings = [episode.embedding for episode in episodes if episode.embedding is not None]
    states = [episode.state_vector for episode in episodes if episode.state_vector is not None]
    segment.embedding = _mean_vector(embeddings)
    segment.state_vector = _mean_vector(states)
    segment.start_time = min(episode.event_time for episode in episodes)
    segment.end_time = max(episode.event_time for episode in episodes)
    segment.summary = episodes[-1].summary


class AdaptiveTemporalSegmenter:
    def __init__(self, controller: TemporalController | None = None) -> None:
        self.controller = controller or RuleBasedTemporalController()

    def process(
        self,
        store: ProgressionStore,
        episode_id: str,
        progression_id: str,
    ) -> tuple[SegmentAction, TemporalSegment]:
        existing = store.segment_for_episode(progression_id, episode_id)
        if existing is not None:
            return SegmentAction.EXTEND, existing

        store.attach_episode(episode_id, progression_id)
        progression = store.progressions[progression_id]
        episode = store.episodes[episode_id]

        if not progression.segment_ids:
            segment = self._create_segment(store, episode_id, progression_id)
            return SegmentAction.SPLIT, segment

        active = store.segments[progression.segment_ids[-1]]
        features = extract_segment_features(episode, progression, active)
        action = self.controller.decide(features)

        progression.volatility = 0.8 * progression.volatility + 0.2 * min(
            features.state_distance,
            1.0,
        )

        if action == SegmentAction.SPLIT:
            segment = self._create_segment(store, episode_id, progression_id)
            return action, segment

        if episode_id not in active.episode_ids:
            active.episode_ids.append(episode_id)
        _refresh_segment_state(store, active)
        return action, active

    @staticmethod
    def _create_segment(
        store: ProgressionStore,
        episode_id: str,
        progression_id: str,
    ) -> TemporalSegment:
        progression = store.progressions[progression_id]
        episode = store.episodes[episode_id]
        segment = TemporalSegment(
            id=f"{progression_id}:segment:{len(progression.segment_ids)}",
            progression_id=progression_id,
            start_time=episode.event_time,
            end_time=episode.event_time,
            summary=episode.summary,
            episode_ids=[episode_id],
            embedding=episode.embedding,
            state_vector=episode.state_vector,
        )
        store.upsert_segment(segment)
        return segment


class FixedWindowTemporalSegmenter(AdaptiveTemporalSegmenter):
    """Calendar-window baseline for comparison with adaptive segmentation."""

    def __init__(self, window: timedelta = timedelta(days=14)) -> None:
        self.window = window

    def process(
        self,
        store: ProgressionStore,
        episode_id: str,
        progression_id: str,
    ) -> tuple[SegmentAction, TemporalSegment]:
        existing = store.segment_for_episode(progression_id, episode_id)
        if existing is not None:
            return SegmentAction.EXTEND, existing

        store.attach_episode(episode_id, progression_id)
        progression = store.progressions[progression_id]
        episode = store.episodes[episode_id]

        if not progression.segment_ids:
            segment = self._create_segment(store, episode_id, progression_id)
            return SegmentAction.SPLIT, segment

        active = store.segments[progression.segment_ids[-1]]
        if episode.event_time - active.start_time >= self.window:
            segment = self._create_segment(store, episode_id, progression_id)
            return SegmentAction.SPLIT, segment

        active.episode_ids.append(episode_id)
        _refresh_segment_state(store, active)
        return SegmentAction.EXTEND, active
