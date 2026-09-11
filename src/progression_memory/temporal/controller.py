from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.models.temporal_segment import TemporalSegment
from progression_memory.routing.router import cosine_similarity


class SegmentAction(StrEnum):
    EXTEND = "extend"
    SPLIT = "split"


@dataclass(frozen=True)
class SegmentFeatures:
    semantic_distance: float
    state_distance: float
    novelty: float
    contradiction: float
    importance: float
    delta_days: float
    volatility: float
    segment_size: int

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.semantic_distance,
                self.state_distance,
                self.novelty,
                self.contradiction,
                self.importance,
                min(self.delta_days / 30.0, 1.0),
                self.volatility,
                min(self.segment_size / 20.0, 1.0),
            ],
            dtype=float,
        )


def euclidean_distance(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape or av.size == 0:
        return 0.0
    return float(np.linalg.norm(av - bv))


def extract_segment_features(
    episode: Episode,
    progression: Progression,
    segment: TemporalSegment,
) -> SegmentFeatures:
    semantic_distance = 1.0 - max(
        -1.0,
        min(1.0, cosine_similarity(episode.embedding, segment.embedding)),
    )
    semantic_distance /= 2.0
    state_distance = euclidean_distance(episode.state_vector, segment.state_vector)

    reference_time = segment.end_time or segment.start_time
    delta_days = max(0.0, (episode.event_time - reference_time).total_seconds() / 86400.0)

    return SegmentFeatures(
        semantic_distance=semantic_distance,
        state_distance=state_distance,
        novelty=episode.novelty,
        contradiction=episode.contradiction,
        importance=episode.importance,
        delta_days=delta_days,
        volatility=progression.volatility,
        segment_size=len(segment.episode_ids),
    )


class RuleBasedTemporalController:
    """Interpretable weighted-change baseline for EXTEND/SPLIT decisions."""

    def __init__(self, threshold: float = 0.36) -> None:
        self.threshold = threshold

    def split_score(self, features: SegmentFeatures) -> float:
        return float(
            0.20 * features.semantic_distance
            + 0.45 * min(features.state_distance, 1.0)
            + 0.10 * features.novelty
            + 0.10 * features.contradiction
            + 0.05 * features.importance
            + 0.05 * min(features.delta_days / 30.0, 1.0)
            + 0.05 * min(features.volatility, 1.0)
        )

    def decide(self, features: SegmentFeatures) -> SegmentAction:
        return (
            SegmentAction.SPLIT
            if self.split_score(features) >= self.threshold
            else SegmentAction.EXTEND
        )


class LinearTemporalController:
    """Small logistic policy matching the initial formal specification."""

    def __init__(
        self,
        weights: list[float] | None = None,
        *,
        bias: float = -2.2,
        threshold: float = 0.5,
    ) -> None:
        self.weights = np.asarray(
            weights
            if weights is not None
            else [1.0, 4.0, 0.6, 0.8, 0.2, 0.3, 0.4, 0.1],
            dtype=float,
        )
        if self.weights.shape != (8,):
            raise ValueError("LinearTemporalController requires exactly 8 weights")
        self.bias = bias
        self.threshold = threshold

    def split_probability(self, features: SegmentFeatures) -> float:
        logit = float(np.dot(self.weights, features.as_vector()) + self.bias)
        return 1.0 / (1.0 + math.exp(-logit))

    def decide(self, features: SegmentFeatures) -> SegmentAction:
        return (
            SegmentAction.SPLIT
            if self.split_probability(features) >= self.threshold
            else SegmentAction.EXTEND
        )
