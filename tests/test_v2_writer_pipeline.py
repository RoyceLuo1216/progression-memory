"""Milestone 2: end-to-end writer integration checks."""
from datetime import UTC, datetime

import pytest

from progression_memory.v2.model_writer import ModelWriter
from progression_memory.v2.models import (
    AttributeOperation,
    AttributePatch,
    Observation,
    Progression,
)
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import (
    ProposedUpdate,
    ScriptedWriter,
    WritePipeline,
    WriterProposal,
)


def observation(oid: str, day: int, subject: str = "u") -> Observation:
    return Observation(
        id=oid,
        source_id=oid,
        subject_id=subject,
        text=f"Grounded source {oid}",
        recorded_at=datetime(2026, 10, day, tzinfo=UTC),
    )


def update(
    action: str,
    progression_id: str | None = None,
    name: str | None = None,
    status: str = "active",
) -> ProposedUpdate:
    return ProposedUpdate(
        action=action,
        progression_id=progression_id,
        name=name,
        reason="source evidence",
        patches=(
            AttributePatch(
                key="status",
                operation=AttributeOperation.SET,
                value=status,
            ),
        ),
    )


def test_recruitment_lifecycle_from_create_to_close():
    engine = StateEngine()
    writer = ScriptedWriter({
        "a": WriterProposal(updates=(update("create", name="Internship search"),)),
        "b": WriterProposal(updates=(
            update("transition", "v2:a:0", status="interviewing"),
        )),
        "c": WriterProposal(updates=(
            update("close", "v2:a:0", status="hired"),
        )),
    })
    pipeline = WritePipeline(engine, writer)
    for index, oid in enumerate(("a", "b", "c"), start=1):
        pipeline.process(observation(oid, index))
    progression = engine.progressions["v2:a:0"]
    assert progression.state["status"] == "hired"
    assert progression.lifecycle.value == "completed"
    assert len(progression.segment_ids) == 2
    assert len(engine.evidence_for(progression.id)) == 3


def test_school_child_and_cycle_split():
    engine = StateEngine()
    engine.create_progression(Progression(
        id="school", subject_id="u", name="University",
    ))
    proposals = {
        "a": WriterProposal(updates=(ProposedUpdate(
            action="create", name="CSC412", parent_id="school",
            reason="enrolled in course",
        ),)),
        "b": WriterProposal(updates=(
            update("split", "school", status="new academic year"),
        )),
    }
    pipeline = WritePipeline(engine, ScriptedWriter(proposals))
    pipeline.process(observation("a", 1))
    pipeline.process(observation("b", 2))
    assert engine.progressions["v2:a:0"].parent_id == "school"
    assert len(engine.progressions["school"].segment_ids) == 1


def test_multi_update_is_atomic_and_no_update_preserves_raw():
    engine = StateEngine()
    engine.create_progression(Progression(id="one", subject_id="u", name="Job"))
    proposal = WriterProposal(updates=(
        update("append", "one"),
        update("append", "missing"),
    ))
    pipeline = WritePipeline(engine, ScriptedWriter({"a": proposal}))
    with pytest.raises(ValueError, match="unproposed"):
        pipeline.process(observation("a", 1))
    assert engine.progressions["one"].version == 0
    assert not engine.observations

    pipeline = WritePipeline(
        engine, ScriptedWriter({"b": WriterProposal()}),
    )
    pipeline.process(observation("b", 2))
    assert "b" in engine.observations
    assert not engine.deltas


def test_cross_subject_target_is_rejected():
    engine = StateEngine()
    engine.create_progression(Progression(
        id="other", subject_id="another", name="Job search",
    ))
    pipeline = WritePipeline(engine, ScriptedWriter({
        "a": WriterProposal(updates=(update("append", "other"),)),
    }))
    with pytest.raises(ValueError, match="unproposed"):
        pipeline.process(observation("a", 1))


def test_model_writer_accepts_schema_and_rejects_invalid():
    class FakeModel:
        def __init__(self, result: str) -> None:
            self.result = result
            self.schema_seen = False

        def generate_messages(self, messages, json_schema=None):
            self.schema_seen = json_schema is not None
            return self.result

    good = FakeModel('{"updates":[]}')
    result = ModelWriter(good).decide(observation("a", 1), [])
    assert result == WriterProposal()
    assert good.schema_seen

    bad = FakeModel('{"updates":[{"action":"append"}]}')
    engine = StateEngine()
    pipeline = WritePipeline(engine, ModelWriter(bad))
    with pytest.raises(ValueError):
        pipeline.process(observation("a", 1))
    assert not engine.observations


def test_duplicate_update_targets_not_committed():
    engine = StateEngine()
    engine.create_progression(Progression(id="one", subject_id="u", name="Job"))
    pipeline = WritePipeline(engine, ScriptedWriter({
        "a": WriterProposal(updates=(
            update("append", "one"),
            update("transition", "one"),
        )),
    }))
    with pytest.raises(ValueError, match="multiple updates"):
        pipeline.process(observation("a", 1))
    assert engine.progressions["one"].version == 0
