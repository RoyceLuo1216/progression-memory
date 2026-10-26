from datetime import UTC, datetime

import pytest

from progression_memory.v2.lifecycle import LifecyclePolicy, apply_lifecycle_policy
from progression_memory.v2.models import (
    ChangeKind,
    Observation,
    Progression,
    StateDelta,
    StorageTier,
)
from progression_memory.v2.state_engine import StateEngine


def when(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


def test_stale_archive_idempotency_and_preserved_evidence():
    engine = StateEngine()
    engine.create_progression(Progression(id="p", subject_id="u", name="Career"))
    obs = Observation(id="a", source_id="a", subject_id="u", text="job search",
                      recorded_at=when(1))
    engine.ingest(obs, StateDelta(
        id="d", observation_id="a", progression_id="p",
    ))
    policy = LifecyclePolicy(stale_after_days=20, archive_after_days=40)
    assert apply_lifecycle_policy(engine, policy, now=when(10)) == {}
    assert apply_lifecycle_policy(engine, policy, now=when(25)) == {"p": "stale"}
    assert apply_lifecycle_policy(engine, policy, now=when(25)) == {}
    later = datetime(2025, 3, 1, tzinfo=UTC)
    assert apply_lifecycle_policy(engine, policy, now=later) == {"p": "archived"}
    assert engine.progressions["p"].storage_tier == StorageTier.ARCHIVED
    assert len(engine.evidence_for("p")) == 1


def test_recent_report_about_old_event_does_not_archive():
    engine = StateEngine()
    engine.create_progression(Progression(id="p", subject_id="u", name="Learning"))
    event = Observation(id="a", source_id="a", subject_id="u", text="old class",
                        recorded_at=when(20), semantic_at=when(1))
    engine.ingest(event, StateDelta(id="d", observation_id="a", progression_id="p"))
    policy = LifecyclePolicy(stale_after_days=10, archive_after_days=20)
    assert apply_lifecycle_policy(engine, policy, now=when(25)) == {}


def test_no_evidence_and_invalid_clock_are_safe():
    engine = StateEngine()
    engine.create_progression(Progression(id="p", subject_id="u", name="No evidence"))
    assert apply_lifecycle_policy(
        engine, LifecyclePolicy(), now=when(1),
    ) == {}
    with pytest.raises(ValueError, match="timezone-aware"):
        apply_lifecycle_policy(
            engine, LifecyclePolicy(), now=datetime(2025, 1, 1),
        )
    with pytest.raises(ValueError):
        LifecyclePolicy(stale_after_days=30, archive_after_days=20)
