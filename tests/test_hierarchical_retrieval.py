from progression_memory.evaluation.hierarchical_retrieval import (
    build_hierarchical_beam_queries,
    build_progression_hierarchy,
)
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery


def _documents() -> tuple[RetrievalDocument, ...]:
    return (
        RetrievalDocument(id="d1", text="Shopify delivery prediction rust shipping", evidence_ids=frozenset({"d1"}), timestamp="2026-06-01"),
        RetrievalDocument(id="d2", text="Shopify duplicate prediction optimization shipping", evidence_ids=frozenset({"d2"}), timestamp="2026-07-01"),
        RetrievalDocument(id="d3", text="graduate application SOP masters statement", evidence_ids=frozenset({"d3"}), timestamp="2026-08-01"),
        RetrievalDocument(id="d4", text="graduate application GRE preparation masters", evidence_ids=frozenset({"d4"}), timestamp="2026-08-02"),
        RetrievalDocument(id="d5", text="running ankle recovery rehabilitation", evidence_ids=frozenset({"d5"}), timestamp="2026-09-01"),
        RetrievalDocument(id="d6", text="running return training ankle", evidence_ids=frozenset({"d6"}), timestamp="2026-09-10"),
    )


def test_hierarchy_builds_coarse_to_fine_layers() -> None:
    by_progression = {
        "career-shopify": [RetrievalDocument(id="s1", text="shopify shipping rust", evidence_ids=frozenset())],
        "career-recruiting": [RetrievalDocument(id="s2", text="recruiting software career", evidence_ids=frozenset())],
        "education-grad": [RetrievalDocument(id="s3", text="masters graduate sop", evidence_ids=frozenset())],
        "education-course": [RetrievalDocument(id="s4", text="university course education", evidence_ids=frozenset())],
        "fitness-running": [RetrievalDocument(id="s5", text="running fitness training", evidence_ids=frozenset())],
    }
    nodes, roots, depth = build_progression_hierarchy(by_progression, branch_factor=2, max_pillars=2)
    assert len(roots) <= 2
    assert depth >= 2
    assert all(node.leaf_progression_ids for node in nodes.values())


def test_hierarchical_beam_retrieval_preserves_raw_source_evidence() -> None:
    query = RetrievalQuery(
        id="q1",
        text="What happened with Shopify shipping predictions?",
        answer="",
        evidence_ids=frozenset({"d2"}),
        documents=_documents(),
        category="fact",
    )
    transformed, stats = build_hierarchical_beam_queries([query], max_pillars=2)
    assert transformed[0].documents
    assert all(not doc.id.startswith("hierarchy:") for doc in transformed[0].documents)
    assert stats["hierarchy"].startswith("inferred_pillars")
    assert stats["avg_nodes_visited_per_query"] > 0
    assert stats["temporal_anchor_expansion"] is True
