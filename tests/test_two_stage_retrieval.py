from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.evaluation.two_stage import build_two_stage_queries


def test_two_stage_returns_only_raw_source_documents() -> None:
    docs = (
        RetrievalDocument(
            id="d1",
            text="Scout the beagle likes hiking and long forest trails.",
            evidence_ids=frozenset({"d1"}),
            timestamp="2026-01-01",
        ),
        RetrievalDocument(
            id="d2",
            text="Scout visited a veterinarian for an ankle injury.",
            evidence_ids=frozenset({"d2"}),
            timestamp="2026-01-02",
        ),
        RetrievalDocument(
            id="d3",
            text="I started learning pottery and made a blue bowl.",
            evidence_ids=frozenset({"d3"}),
            timestamp="2026-02-01",
        ),
    )
    query = RetrievalQuery(
        id="q",
        text="Which memory mentioned the veterinarian and ankle injury?",
        evidence_ids=frozenset({"d2"}),
        documents=docs,
        category="test",
    )

    transformed, stats = build_two_stage_queries([query], stage1_candidates=2)

    assert transformed[0].documents
    assert all(document.id in {"d1", "d2", "d3"} for document in transformed[0].documents)
    assert all(not document.id.startswith("index:") for document in transformed[0].documents)
    assert stats["stage1_candidate_limit"] == 2
    assert 0.0 <= float(stats["avg_stage1_evidence_coverage"]) <= 1.0


def test_two_stage_structural_index_does_not_receive_evidence_credit() -> None:
    docs = (
        RetrievalDocument(
            id="evidence",
            text="The exact target phrase is aurora lantern.",
            evidence_ids=frozenset({"evidence"}),
            timestamp="2026-01-01",
        ),
        RetrievalDocument(
            id="noise",
            text="Completely unrelated memory.",
            evidence_ids=frozenset({"noise"}),
            timestamp="2026-01-02",
        ),
    )
    query = RetrievalQuery(
        id="q",
        text="aurora lantern",
        evidence_ids=frozenset({"evidence"}),
        documents=docs,
        category="test",
    )

    transformed, stats = build_two_stage_queries([query], stage1_candidates=1)

    assert transformed[0].documents
    assert all(document.evidence_ids for document in transformed[0].documents)
    assert float(stats["avg_stage1_evidence_coverage"]) == 1.0
