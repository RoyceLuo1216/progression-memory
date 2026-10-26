from datetime import UTC, datetime

from progression_memory.v2.maintenance import MaintenanceWorker
from progression_memory.v2.models import (
    AttributeOperation,
    AttributePatch,
    Observation,
    Progression,
    StateDelta,
)
from progression_memory.v2.read_path import build_documents, retrieve
from progression_memory.v2.state_engine import StateEngine


def date(day: int) -> datetime:
    return datetime(2026, 10, day, tzinfo=UTC)


def test_read_views_and_historical_evidence_cutoffs():
    engine = StateEngine()
    engine.create_progression(Progression(id="parent", subject_id="u", name="Education"))
    engine.create_progression(Progression(
        id="child", subject_id="u", name="CSC373", parent_id="parent",
    ))
    first = Observation(
        id="a", source_id="a", subject_id="u", text="CSC373 studying",
        recorded_at=date(4), semantic_at=date(2),
    )
    later = Observation(
        id="b", source_id="b", subject_id="u", text="CSC373 exam finished",
        recorded_at=date(7), semantic_at=date(5),
    )
    for observation, status in ((first, "studying"), (later, "completed")):
        engine.ingest(observation, StateDelta(
            id=f"d:{observation.id}", observation_id=observation.id,
            progression_id="child", patches=(
                AttributePatch(key="status", operation=AttributeOperation.SET, value=status),
            ),
        ))
    MaintenanceWorker(engine).run_pending()
    state = build_documents(engine, view="state", subject_id="u")
    assert len(state) == 1
    assert state[0].evidence_ids == {"a", "b"}
    full = build_documents(engine, view="full", subject_id="u")
    assert len(full) == 3
    assert any(d.id == "v2:state:parent" and d.evidence_ids == {"a", "b"} for d in full)
    assert retrieve(engine, "CSC373 exam", subject_id="u")
    historical = build_documents(
        engine, view="full", subject_id="u", valid_at=date(3), known_at=date(6),
    )
    assert all("b" not in d.evidence_ids for d in historical)
    assert any("studying" in d.text for d in historical)
    assert all("completed" not in d.text for d in historical)
    unknown_then = build_documents(
        engine, view="full", subject_id="u", valid_at=date(3), known_at=date(3),
    )
    assert unknown_then == ()


def test_other_subject_does_not_leak():
    engine = StateEngine()
    engine.create_progression(Progression(id="other", subject_id="v", name="Private"))
    engine.observations["external"] = Observation(
        id="external", source_id="external", subject_id="v",
        text="sensitive", recorded_at=date(1),
    )
    assert build_documents(engine, view="full", subject_id="u") == ()
