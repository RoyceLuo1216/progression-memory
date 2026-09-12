from progression_memory.evaluation.dag_retrieval import (
    V2_PARENT_INDEX_TERMS,
    _build_hierarchy_v2,
    _hierarchical_segment_ranking_v2,
    build_dag_two_stage_queries,
    build_dag_two_stage_v2_queries,
)
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery


def _documents() -> tuple[RetrievalDocument, ...]:
    return (
        RetrievalDocument(id="d1", text="Roy adopted a beagle named Scout and started hiking.", evidence_ids=frozenset({"d1"}), timestamp="2026-01-01"),
        RetrievalDocument(id="d2", text="Scout now visits the dog park every weekend.", evidence_ids=frozenset({"d2"}), timestamp="2026-01-03"),
        RetrievalDocument(id="d3", text="Roy started pottery and made a blue bowl.", evidence_ids=frozenset({"d3"}), timestamp="2026-02-01"),
    )


def _query() -> RetrievalQuery:
    return RetrievalQuery(id="q", text="Which beagle did Roy adopt?", evidence_ids=frozenset({"d1"}), documents=_documents(), category="test")


def test_dag_two_stage_returns_only_raw_sources_and_reports_graph_stats() -> None:
    transformed, stats = build_dag_two_stage_queries([_query()])
    selected = transformed[0].documents
    assert selected
    assert all(document.id in {"d1", "d2", "d3"} for document in selected)
    assert all(not document.id.startswith("index:") for document in selected)
    assert stats["avg_graph_nodes"] >= stats["avg_structural_units"]
    assert stats["avg_graph_edges"] == stats["avg_structural_units"]
    assert stats["avg_stage1_evidence_coverage"] == 1.0


def test_dag_two_stage_validates_rank_fusion_parameters() -> None:
    query = RetrievalQuery(id="q", text="test", evidence_ids=frozenset({"d"}), documents=(RetrievalDocument(id="d", text="test", evidence_ids=frozenset({"d"})),), category="test")
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


def test_dag_v2_parent_indexes_are_capped_and_evidence_free() -> None:
    parents, _, _, _ = _build_hierarchy_v2(_documents())
    assert parents
    assert all(len(parent.text.split()) <= V2_PARENT_INDEX_TERMS for parent in parents)
    assert all(not parent.evidence_ids for parent in parents)


def test_dag_v2_returns_only_raw_sources_and_reports_diversity_stats() -> None:
    transformed, stats = build_dag_two_stage_v2_queries([_query()])
    selected = transformed[0].documents
    assert selected
    assert all(document.id in {"d1", "d2", "d3"} for document in selected)
    assert stats["parent_index_max_terms"] == V2_PARENT_INDEX_TERMS
    assert stats["avg_seeded_parent_count"] >= 0.0
    assert stats["avg_stage1_unique_progressions"] >= 1.0


def test_dag_v2_diversifies_across_positive_parent_progressions() -> None:
    parents = (
        RetrievalDocument(id="p1", text="alpha hiking", metadata={"progression_id": "p1"}),
        RetrievalDocument(id="p2", text="alpha pottery", metadata={"progression_id": "p2"}),
    )
    segments = (
        RetrievalDocument(id="s1", text="alpha hiking trail", metadata={"progression_id": "p1"}),
        RetrievalDocument(id="s2", text="alpha hiking boots", metadata={"progression_id": "p1"}),
        RetrievalDocument(id="s3", text="alpha pottery bowl", metadata={"progression_id": "p2"}),
        RetrievalDocument(id="s4", text="alpha pottery clay", metadata={"progression_id": "p2"}),
    )
    selected, seeded_parents, unique_progressions = _hierarchical_segment_ranking_v2(
        "alpha", parents, segments, candidate_limit=4, parent_weight=0.35, rrf_k=60.0, diversity_fraction=0.5, max_diversity_parents=2
    )
    assert len(selected) == 4
    assert seeded_parents == 2
    assert unique_progressions == 2
