"""Paired benchmark smoke tests, independent of external datasets."""
from datetime import UTC, datetime

from progression_memory.benchmarks.v2_paired import TraceFixture, run
from progression_memory.v2.models import Observation
from progression_memory.v2.writer_pipeline import ProposedUpdate, WriterProposal


def test_paired_protocol_same_questions_and_budget():
    fixture = TraceFixture(
        observations=[
            Observation(id="a", source_id="a", subject_id="u",
                        text="Alex started studying robotics",
                        recorded_at=datetime(2026, 10, 1, tzinfo=UTC)),
            Observation(id="b", source_id="b", subject_id="u",
                        text="Alex completed robotics course",
                        recorded_at=datetime(2026, 10, 2, tzinfo=UTC)),
        ],
        proposals={
            "a": WriterProposal(updates=(ProposedUpdate(
                action="create", name="Robotics course",
                reason="explicitly studying robotics", stage="started",
            ),)),
            "b": WriterProposal(updates=(ProposedUpdate(
                action="close", progression_id="v2:a:0",
                reason="explicitly completed robotics", stage="completed",
            ),)),
        },
        questions=[{
            "id": "q1", "subject_id": "u",
            "question": "When did Alex complete robotics?",
            "evidence_ids": ["b"], "category": "temporal",
        }],
    )
    result = run(fixture, [100, 800])
    assert set(result["systems"]) == {"rag", "v2_state", "v2_full"}
    for output in result["systems"].values():
        assert output["query_count"] == 1
        assert set(output["budgets"]) == {"100", "800"}
        assert output["budgets"]["800"]["per_query"][0]["total_evidence"] == 1
    assert result["systems"]["rag"]["budgets"]["800"]["macro_evidence_recall"] == 1.0


def test_empty_writer_trace_not_false_state():
    fixture = TraceFixture(
        observations=[Observation(
            id="a", source_id="a", subject_id="u", text="Just asking",
            recorded_at=datetime(2026, 10, 1, tzinfo=UTC),
        )],
        proposals={"a": WriterProposal()},
        questions=[{
            "id": "q", "subject_id": "u", "question": "asking",
            "evidence_ids": ["a"],
        }],
    )
    result = run(fixture, [500])
    assert result["systems"]["v2_state"]["budgets"]["500"]["macro_evidence_recall"] == 0
    assert result["systems"]["rag"]["budgets"]["500"]["macro_evidence_recall"] == 1
