from progression_memory.evaluation.dag_retrieval import build_dag_two_stage_queries
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery


def test_dag_two_stage_returns_only_raw_sources_and_reports_graph_stats() -> None:
    documents = (
        RetrievalDocument(
            id="d1",
            text="Roy adopted a beagle named Scout and started hiking.",
            evidence_ids=frozenset({"d1"}),
            timestamp="2026-01-01",
        ),
        RetrievalDocument(
            id="d2",
            text="Scout now visits the dog park every weekend.",
            evidence_ids=frozenset({"d2"}),
            timestamp="2026-01-03",
        ),
        RetrievalDocument(
            id="d3",
            text="Roy started pottery and made a blue bowl.",
            evidence_ids=frozenset({"d3"}),
            timestamp="2026-02-01",
        ),
    )
    query = RetrievalQuery(
        id="q",
        text="Which beagle did Roy adopt?",
        evidence_ids=frozenset({"d1"}),
        documents=documents,
        category="test",
    )

    transformed, stats = build_dag_two_stage_queries([query])
    selected = transformed[0].documents

    assert selected
    assert all(document.id in {"d1", "d2", "d3"} for document in selected)
    assert all(not document.id.startswith("index:") for document in selected)
    assert stats["avg_graph_nodes"] >= stats["avg_structural_units"]
    assert stats["avg_graph_edges"] == stats["avg_structural_units"]
    assert stats["avg_stage1_evidence_coverage"] == 1.0


def test_dag_two_stage_validates_rank_fusion_parameters() -> None:
    query = RetrievalQuery(
        id="q",
        text="test",
        evidence_ids=frozenset({"d"}),
        documents=(RetrievalDocument(id="d", text="test", evidence_ids=frozenset({"d"})),),
        category="test",
    )

    try:
        build_dag_two_stage_queries([query], parent_rrf_weight=-0.1)
        raise AssertionError("negative parent weight should fail")
    except ValueError:
        pass

    try:
        build_dag_two_stage_queries([query], rrf_k=0.0)
        raise AssertionError("non-positive rrf_k should fail")
    except ValueError:
        pass
