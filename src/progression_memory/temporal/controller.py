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
    information_weight: float = 1.0

    def as_vector(self) -> np.ndarray:
        # Keep the formal v1 controller eight-dimensional. information_weight is
        # controller context, not a ninth learned feature, so v1 remains a clean ablation.
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
    raw_information_weight = episode.metadata.get("state_confidence", 1.0)
    try:
        information_weight = float(raw_information_weight)
    except (TypeError, ValueError):
        information_weight = 1.0
    information_weight = max(0.0, min(1.0, information_weight))

    return SegmentFeatures(
        semantic_distance=semantic_distance,
        state_distance=state_distance,
        novelty=episode.novelty,
        contradiction=episode.contradiction,
        importance=episode.importance,
        delta_days=delta_days,
        volatility=progression.volatility,
        segment_size=len(segment.episode_ids),
        information_weight=information_weight,
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


class HysteresisLinearTemporalController(LinearTemporalController):
    """Adaptive-v2 controller that dampens noisy state change without changing v1's basis.

    High-confidence memories reduce exactly to the v1 logistic policy once a segment has
    enough evidence. Low-confidence short events damp semantic/state/novelty signals and
    require a slightly higher threshold, while contradiction or a very strong raw change
    can still trigger an immediate split.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        *,
        bias: float = -2.2,
        threshold: float = 0.5,
        strong_threshold: float = 0.80,
        min_segment_size: int = 3,
    ) -> None:
        super().__init__(weights, bias=bias, threshold=threshold)
        if not 0.0 <= strong_threshold <= 1.0:
            raise ValueError("strong_threshold must be in [0, 1]")
        if min_segment_size < 1:
            raise ValueError("min_segment_size must be positive")
        self.strong_threshold = strong_threshold
        self.min_segment_size = min_segment_size

    def _scaled_vector(self, features: SegmentFeatures) -> np.ndarray:
        vector = features.as_vector().copy()
        confidence = max(0.0, min(1.0, features.information_weight))
        change_scale = 0.35 + 0.65 * confidence
        vector[:3] *= change_scale
        return vector

    def split_probability(self, features: SegmentFeatures) -> float:
        logit = float(np.dot(self.weights, self._scaled_vector(features)) + self.bias)
        return 1.0 / (1.0 + math.exp(-logit))

    def decide(self, features: SegmentFeatures) -> SegmentAction:
        probability = self.split_probability(features)
        strong_raw_change = (
            features.semantic_distance >= 0.80
            and features.state_distance >= 0.80
            and features.novelty >= 0.50
        )
        if (
            features.contradiction >= 0.85
            or strong_raw_change
            or probability >= self.strong_threshold
        ):
            return SegmentAction.SPLIT

        if features.segment_size < self.min_segment_size:
            return SegmentAction.EXTEND

        confidence = max(0.0, min(1.0, features.information_weight))
        effective_threshold = self.threshold + 0.10 * (1.0 - confidence)
        return (
            SegmentAction.SPLIT
            if probability >= effective_threshold
            else SegmentAction.EXTEND
        )
