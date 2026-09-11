from datetime import datetime, timezone

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.routing.router import HeuristicProgressionRouter, RouteAction


def test_router_attaches_semantically_compatible_episode() -> None:
    now = datetime.now(timezone.utc)
    episode = Episode(
        id="e1",
        summary="Implemented another part of the memory research project.",
        dialogue_time=now,
        embedding=[0.98, 0.10, 0.05],
        entities=["research", "project"],
    )
    research = Progression(
        id="research",
        title="Research",
        scope="research project",
        embedding=[1.0, 0.08, 0.04],
        last_updated_time=now,
    )
    fitness = Progression(
        id="fitness",
        title="Fitness",
        scope="fitness running",
        embedding=[0.05, 0.10, 1.0],
        last_updated_time=now,
    )

    decision = HeuristicProgressionRouter().route(episode, [research, fitness])

    assert decision.action == RouteAction.ATTACH
    assert decision.progression_ids == ["research"]


def test_router_creates_when_no_candidate_is_compatible() -> None:
    now = datetime.now(timezone.utc)
    episode = Episode(
        id="e1",
        summary="Started a completely unrelated cooking project.",
        dialogue_time=now,
        embedding=[0.0, 1.0, 0.0],
    )
    candidate = Progression(
        id="career",
        title="Career",
        scope="career jobs",
        embedding=[1.0, 0.0, 0.0],
        last_updated_time=now,
    )

    decision = HeuristicProgressionRouter().route(episode, [candidate])

    assert decision.action == RouteAction.CREATE
