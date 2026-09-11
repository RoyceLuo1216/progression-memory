from __future__ import annotations

import math
from enum import StrEnum
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression


class RouteAction(StrEnum):
    ATTACH = "attach"
    CREATE = "create"
    MULTI_ATTACH = "multi_attach"
    STATIC = "static"


class RoutingDecision(BaseModel):
    action: RouteAction
    progression_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape or av.size == 0:
        return 0.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def entity_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = {x.casefold() for x in a}
    right = {x.casefold() for x in b}
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def temporal_compatibility(
    episode: Episode,
    progression: Progression,
    half_life_days: float = 90.0,
) -> float:
    if progression.last_updated_time is None:
        return 0.5
    delta_seconds = abs((episode.event_time - progression.last_updated_time).total_seconds())
    delta_days = delta_seconds / 86400.0
    return math.exp(-math.log(2.0) * delta_days / half_life_days)


def progression_route_score(episode: Episode, progression: Progression) -> float:
    """Initial R(e, P): semantic + entity + temporal compatibility."""
    semantic = (cosine_similarity(episode.embedding, progression.embedding) + 1.0) / 2.0
    entities = entity_jaccard(episode.entities, progression.scope.split())
    temporal = temporal_compatibility(episode, progression)
    return float(0.65 * semantic + 0.20 * entities + 0.15 * temporal)


class HeuristicProgressionRouter:
    """Deterministic v1 router used before a learned/LLM-backed router is introduced."""

    def __init__(
        self,
        *,
        attach_threshold: float = 0.68,
        multi_attach_margin: float = 0.04,
    ) -> None:
        self.attach_threshold = attach_threshold
        self.multi_attach_margin = multi_attach_margin

    def route(
        self,
        episode: Episode,
        candidates: Iterable[Progression],
        *,
        progressive: bool = True,
    ) -> RoutingDecision:
        if not progressive:
            return RoutingDecision(action=RouteAction.STATIC, confidence=1.0)

        scored = sorted(
            (
                (progression.id, progression_route_score(episode, progression))
                for progression in candidates
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        scores = {progression_id: score for progression_id, score in scored}
        if not scored or scored[0][1] < self.attach_threshold:
            confidence = 1.0 if not scored else max(0.0, 1.0 - scored[0][1])
            return RoutingDecision(
                action=RouteAction.CREATE,
                confidence=confidence,
                scores=scores,
            )

        top_id, top_score = scored[0]
        selected = [top_id]
        if len(scored) > 1:
            second_id, second_score = scored[1]
            if (
                second_score >= self.attach_threshold
                and top_score - second_score <= self.multi_attach_margin
            ):
                selected.append(second_id)

        action = RouteAction.MULTI_ATTACH if len(selected) > 1 else RouteAction.ATTACH
        return RoutingDecision(
            action=action,
            progression_ids=selected,
            confidence=min(1.0, top_score),
            scores=scores,
        )
