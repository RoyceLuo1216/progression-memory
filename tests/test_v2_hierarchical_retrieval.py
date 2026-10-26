from datetime import UTC, datetime

import pytest

from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.benchmarks.v2_retrieval_ablation import evaluate, prepare
from progression_memory.v2.hierarchical_retrieval import retrieve, select_budget
from progression_memory.v2.models import Observation
from progression_memory.v2.writer_pipeline import ProposedUpdate, WriterProposal


def fixture():
    return TraceFixture(
        observations=[
            Observation(
                id="a", source_id="a", subject_id="u",
                text="University computer science degree",
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Observation(
                id="b", source_id="b", subject_id="u",
                text="CSC373 algorithms course passed",
                recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
            ),
            Observation(
                id="c", source_id="c", subject_id="u",
                text="Weekend cooking class",
                recorded_at=datetime(2026, 3, 1, tzinfo=UTC),
            ),
        ],
        proposals={
            "a": WriterProposal(updates=(ProposedUpdate(
                action="create", name="University", reason="university",
            ),)),
            "b": WriterProposal(updates=(ProposedUpdate(
                action="create", name="CSC373", parent_id="v2:a:0",
                reason="course",
            ),)),
            "c": WriterProposal(updates=(ProposedUpdate(
                action="create", name="Cooking", reason="class",
            ),)),
        },
        questions=[{
            "id": "q", "subject_id": "u", "question": "algorithms CSC373 passed",
            "evidence_ids": ["b"], "category": "state",
        }],
    )


def embed(text: str) -> list[float]:
    return [float("CSC373" in text), float("cooking" in text)]


def test_modes_and_branch_rescue():
    engine = prepare(fixture())
    for mode in ("flat_bm25", "flat_hybrid", "hierarchical_bm25",
                 "hierarchical_hybrid", "hierarchical_hybrid_temporal"):
        kwargs = {"embedder": embed}
        if mode.endswith("temporal"):
            kwargs["valid_at"] = datetime(2026, 3, 2, tzinfo=UTC)
        results = retrieve(engine, "CSC373 algorithms", mode=mode, k=4, **kwargs)
        assert results
        assert len({r.id for r in results}) == len(results)
        assert sum(len(x.text) for x in select_budget(results, 500)) <= 500


def test_hybrid_requires_embedder_and_temporal_requires_horizon():
    engine = prepare(fixture())
    with pytest.raises(ValueError, match="embedding"):
        retrieve(engine, "algorithms", mode="flat_hybrid")
    with pytest.raises(ValueError, match="valid_at"):
        retrieve(engine, "algorithms", mode="hierarchical_hybrid_temporal", embedder=embed)


def test_as_of_never_returns_future_observation():
    engine = prepare(fixture())
    documents = retrieve(
        engine, "CSC373 algorithms", mode="hierarchical_bm25", k=20,
        valid_at=datetime(2026, 1, 15, tzinfo=UTC),
    )
    assert all("b" not in doc.evidence_ids for doc in documents)
    assert all("c" not in doc.evidence_ids for doc in documents)


def test_ablation_reports_skipped_dense_instead_of_faking():
    result = evaluate(fixture(), [300, 1000])
    assert result["systems"]["flat_bm25"]["status"] == "completed"
    assert result["systems"]["hierarchical_bm25"]["status"] == "completed"
    assert result["systems"]["flat_hybrid"]["status"] == "skipped"
    with_dense = evaluate(fixture(), [1000], embedder=embed)
    assert all(value["status"] == "completed" for value in with_dense["systems"].values())
