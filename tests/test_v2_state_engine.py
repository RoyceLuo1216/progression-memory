"""V2 contract tests: deterministic replay, semantic time, boundaries, and evidence."""
from datetime import datetime, timezone

import pytest

from progression_memory.v2.models import (
    AttributeOperation, AttributePatch, ChangeKind, Observation,
    Progression, StateDelta, StorageTier, TimeSource,
)
from progression_memory.v2.state_engine import StateEngine


def t(day: int) -> datetime:
    return datetime(2026, 10, day, tzinfo=timezone.utc)


def event(eid: str, recorded: int, semantic: int | None = None) -> Observation:
    return Observation(
        id=eid, subject_id="person", source_id=eid, text=eid,
        recorded_at=t(recorded),
        semantic_at=t(semantic) if semantic is not None else None,
    )


def patch(eid: str, value: str, change: ChangeKind = ChangeKind.APPEND) -> StateDelta:
    return StateDelta(
        id=f"delta-{eid}", progression_id="job", observation_id=eid,
        change=change,
        patches=(AttributePatch(key="status", operation=AttributeOperation.SET, value=value),),
    )


def engine() -> StateEngine:
    e = StateEngine()
    e.create_progression(Progression(id="career", subject_id="person", name="Career"))
    e.create_progression(Progression(
        id="job", subject_id="person", name="Job search", parent_id="career",
    ))
    return e


def test_semantic_time_preferred_and_wall_fallback():
    assert event("one", 12, 8).effective_at == t(8)
    assert event("one", 12, 8).time_source == TimeSource.SEMANTIC
    assert event("two", 12).effective_at == t(12)
    assert event("two", 12).time_source == TimeSource.WALL_FALLBACK


def test_late_arrival_replays_by_event_time_and_supports_knowledge_cutoff():
    e = engine()
    e.ingest(event("latest", 10, 10), patch("latest", "interview"))
    e.ingest(event("late", 12, 8), patch("late", "applied"))
    assert e.progressions["job"].state["status"] == "interview"
    assert e.state_at("job", valid_at=t(8), known_at=t(10)) == {}
    assert e.state_at("job", valid_at=t(8), known_at=t(12)) == {"status": "applied"}
    assert e.state_at("job", valid_at=t(10), known_at=t(12)) == {"status": "interview"}
    assert len(e.revisions["job"]) == 2


def test_duplicate_ingestion_idempotent_and_conflict_rejected():
    e = engine()
    obs, delta = event("a", 8), patch("a", "applied")
    assert e.ingest(obs, delta)
    assert not e.ingest(obs, delta)
    assert len(e.evidence_for("job")) == 1
    with pytest.raises(ValueError, match="conflicting"):
        e.ingest(obs, patch("a", "interview"))


def test_phase_boundary_and_close_and_revival():
    e = engine()
    e.ingest(event("a", 7), patch("a", "applying"))
    e.ingest(event("b", 8), patch("b", "screen", ChangeKind.TRANSITION))
    e.ingest(event("c", 9), patch("c", "hired", ChangeKind.CLOSE))
    assert len(e.progressions["job"].segment_ids) == 2
    assert e.progressions["job"].lifecycle.value == "completed"
    e.move_to_tier("job", StorageTier.ARCHIVED)
    assert e.progressions["job"].storage_tier == StorageTier.ARCHIVED
    e.ingest(event("d", 12), patch("d", "searching again", ChangeKind.REVIVE))
    assert e.progressions["job"].lifecycle.value == "active"
    assert e.progressions["job"].storage_tier == StorageTier.ACTIVE
    assert len(e.progressions["job"].segment_ids) == 3
    assert len(e.evidence_for("job")) == 4


def test_invalid_patch_is_atomic():
    e = engine()
    e.ingest(event("a", 7), patch("a", "applied"))
    invalid = StateDelta(
        id="delta-b", progression_id="job", observation_id="b",
        patches=(AttributePatch(key="status", operation=AttributeOperation.ADD, value="test"),),
    )
    with pytest.raises(ValueError, match="not a list"):
        e.ingest(event("b", 8), invalid)
    assert e.progressions["job"].version == 1
    assert "b" not in e.observations
    assert "delta-b" not in e.deltas


def test_hierarchy_subject_guard():
    e = engine()
    with pytest.raises(ValueError, match="subject mismatch"):
        e.create_progression(Progression(
            id="other", subject_id="different", name="Other", parent_id="career",
        ))


def test_timezone_naive_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        Observation(
            id="bad", subject_id="person", source_id="bad", text="x",
            recorded_at=datetime(2026, 10, 8),
        )


def test_one_observation_can_update_multiple_progressions():
    e = engine()
    shared = event("shared", 10)
    e.ingest(shared, patch("shared", "interviewing"))
    second = StateDelta(
        id="delta-shared-career", progression_id="career", observation_id="shared",
        patches=(AttributePatch(key="focus", operation=AttributeOperation.SET,
                                value="internships"),),
    )
    assert e.ingest(shared, second)
    assert e.progressions["job"].state["status"] == "interviewing"
    assert e.progressions["career"].state["focus"] == "internships"
    assert len(e.observations) == 1
