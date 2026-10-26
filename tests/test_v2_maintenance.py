"""Hierarchy and propagation regression tests."""
from datetime import UTC, datetime

import pytest

from progression_memory.v2.aggregation import effective_state
from progression_memory.v2.hierarchy import HierarchyIndex
from progression_memory.v2.maintenance import MaintenanceWorker
from progression_memory.v2.models import (
    AttributeOperation,
    AttributePatch,
    ChangeKind,
    Observation,
    Progression,
    StateDelta,
)
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import (
    ProposedUpdate,
    ScriptedWriter,
    WritePipeline,
    WriterProposal,
)


def build() -> StateEngine:
    engine = StateEngine()
    for id_, parent in [
        ("career", None), ("school", "career"),
        ("course1", "school"), ("course2", "school"),
        ("other", "career"),
    ]:
        engine.create_progression(Progression(
            id=id_, subject_id="u", name=id_, parent_id=parent,
        ))
    return engine


def write(engine: StateEngine, pid: str, day: int, close: bool = False) -> None:
    obs = Observation(
        id=f"{pid}-{day}", source_id=f"{pid}-{day}", subject_id="u",
        text="source", recorded_at=datetime(2026, 10, day, tzinfo=UTC),
    )
    delta = StateDelta(
        id=f"delta-{pid}-{day}", progression_id=pid, observation_id=obs.id,
        change=ChangeKind.CLOSE if close else ChangeKind.APPEND,
        patches=(AttributePatch(
            key="note", operation=AttributeOperation.SET,
            value=f"direct {pid}",
        ),),
    )
    engine.ingest(obs, delta)


def test_hierarchy_paths_and_bottom_up_aggregation():
    engine = build()
    index = HierarchyIndex(engine.progressions)
    assert index.ancestors("course1") == ["school", "career"]
    assert index.children("school") == ["course1", "course2"]
    assert set(index.descendants("career")) == {
        "school", "course1", "course2", "other",
    }
    write(engine, "school", 1)
    write(engine, "course1", 2, close=True)
    completed = MaintenanceWorker(engine).run_pending()
    assert completed.index("school") < completed.index("career")
    assert engine.progressions["school"].derived_state["child_count"] == 2
    assert engine.progressions["school"].derived_state["completed_children"] == 1
    assert engine.progressions["school"].state == {"note": "direct school"}
    assert effective_state(engine.progressions["school"])["direct"] == {
        "note": "direct school",
    }
    assert engine.progressions["career"].derived_state["child_count"] == 2
    assert engine.dirty_progressions == set()
    assert MaintenanceWorker(engine).run_pending() == []


def test_unrelated_branch_is_not_invalidated_after_initial_refresh():
    engine = build()
    MaintenanceWorker(engine).run_pending()
    write(engine, "course1", 3)
    assert engine.dirty_progressions == {"school", "career"}
    MaintenanceWorker(engine).run_pending()
    write(engine, "other", 4)
    assert engine.dirty_progressions == {"career"}


def test_failed_writer_batch_has_no_dirty_side_effect():
    engine = build()
    MaintenanceWorker(engine).run_pending()
    proposal = WriterProposal(updates=(
        ProposedUpdate(action="append", progression_id="course1", reason="valid"),
        ProposedUpdate(action="append", progression_id="missing", reason="invalid"),
    ))
    pipeline = WritePipeline(engine, ScriptedWriter({"bad": proposal}))
    obs = Observation(
        id="bad", source_id="bad", subject_id="u", text="test",
        recorded_at=datetime(2026, 10, 5, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="unproposed"):
        pipeline.process(obs)
    assert not engine.dirty_progressions
    assert "bad" not in engine.observations


def test_stale_summarizer_cannot_commit():
    engine = build()
    MaintenanceWorker(engine).run_pending()
    write(engine, "course1", 1)

    def concurrent_write(parent: Progression, children: list[Progression]) -> str:
        if parent.id == "school":
            write(engine, "course2", 2, close=True)
        return "outdated"

    done = MaintenanceWorker(engine, concurrent_write).run_pending()
    assert "school" not in done
    assert "school" in engine.dirty_progressions
    refreshed = MaintenanceWorker(engine).run_pending()
    assert "school" in refreshed
    assert engine.progressions["school"].derived_state["completed_children"] == 1


def test_invalid_hierarchy_cycle_rejected():
    engine = build()
    engine.progressions["career"].parent_id = "course1"
    with pytest.raises(ValueError, match="cycle"):
        HierarchyIndex(engine.progressions)
