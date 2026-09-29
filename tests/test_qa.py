import json

from progression_memory.evaluation.qa import (
    ExtractiveBM25Reader,
    MultiEvidenceExtractiveReader,
    QARunConfig,
    answer_contains,
    evaluate_qa,
    exact_match,
    plan_query,
    select_reader_context,
    token_f1,
    write_longmemeval_hypotheses,
)
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.benchmarks.qa_benchmark import paired_qa
from progression_memory.benchmarks.baseline_retrieval import transform_source_queries


def _query() -> RetrievalQuery:
    documents = (
        RetrievalDocument(
            id="d1",
            text="user: I adopted a beagle named Scout.",
            evidence_ids=frozenset({"d1"}),
            timestamp="2026-01-01",
        ),
        RetrievalDocument(
            id="d2",
            text="assistant: The weather is sunny today.",
            evidence_ids=frozenset({"d2"}),
            timestamp="2026-01-02",
        ),
    )
    return RetrievalQuery(
        id="q1",
        text="What is the name of the beagle I adopted?",
        answer="Scout",
        evidence_ids=frozenset({"d1"}),
        documents=documents,
        category="single-session-user",
    )


def test_qa_metrics_normalize_answers() -> None:
    assert exact_match("The Scout.", "Scout") == 1.0
    assert answer_contains("I adopted Scout yesterday.", "Scout") == 1.0
    assert token_f1("Scout the beagle", "Scout") > 0.0


def test_extractive_reader_pipeline_tracks_retrieval_and_context() -> None:
    result = evaluate_qa(
        [_query()],
        ExtractiveBM25Reader(),
        QARunConfig(top_k=1),
    )
    overall = result["overall"]
    assert overall["query_count"] == 1
    assert overall["retrieval_recall"] == 1.0
    assert overall["answer_contains"] == 1.0
    assert overall["avg_context_units"] == 1.0


def test_context_budget_skips_oversized_document_instead_of_stopping() -> None:
    query = RetrievalQuery(
        id="q-budget",
        text="Scout dog",
        answer="Scout",
        evidence_ids=frozenset({"small"}),
        documents=(
            RetrievalDocument(
                id="large",
                text="Scout dog " + "x" * 100,
                evidence_ids=frozenset({"large"}),
            ),
            RetrievalDocument(
                id="small",
                text="Scout dog",
                evidence_ids=frozenset({"small"}),
            ),
        ),
        category="test",
    )
    context = select_reader_context(query, QARunConfig(char_budget=20))
    assert [document.id for document in context] == ["small"]


def test_evidence_pack_keeps_timestamp_and_source_provenance() -> None:
    query = RetrievalQuery(
        id="q-pack",
        text="Where did Scout move?",
        answer="Toronto",
        evidence_ids=frozenset({"session-1"}),
        documents=(
            RetrievalDocument(
                id="session-1",
                text="We talked about dinner. Scout moved to Toronto. Then we discussed work.",
                evidence_ids=frozenset({"session-1"}),
                timestamp="2026-02-03",
            ),
        ),
        category="single-session-user",
    )
    context = select_reader_context(
        query,
        QARunConfig(char_budget=200, context_strategy="evidence_pack"),
    )
    assert context
    assert "[2026-02-03]" in context[0].text
    assert context[0].metadata["source_document_id"] == "session-1"
    assert context[0].evidence_ids == frozenset({"session-1"})


def test_query_planner_marks_multi_evidence_and_update_queries() -> None:
    assert plan_query("How many trips did I take in total?").needs_multiple_evidence
    update = plan_query("What is my most recent job now?")
    assert update.kind == "update"
    assert update.prefer_recent


def test_multi_evidence_reader_can_return_multiple_sources() -> None:
    context = (
        RetrievalDocument(
            id="a",
            text="I visited Montreal in May.",
            evidence_ids=frozenset({"a"}),
        ),
        RetrievalDocument(
            id="b",
            text="I visited Vancouver in June.",
            evidence_ids=frozenset({"b"}),
        ),
    )
    answer = MultiEvidenceExtractiveReader(max_spans=4).answer(
        "Which cities were visited?", context
    )
    assert "Montreal" in answer
    assert "Vancouver" in answer


def test_longmemeval_hypothesis_export(tmp_path) -> None:
    result = evaluate_qa(
        [_query()],
        ExtractiveBM25Reader(),
        QARunConfig(top_k=1),
    )
    path = tmp_path / "hypotheses.jsonl"
    write_longmemeval_hypotheses(result, path)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["question_id"] == "q1"
    assert "hypothesis" in row


def test_all_grouping_baselines_enter_same_source_level_qa_reader() -> None:
    query = _query()
    results = {}
    for system in ("append", "temporal_only", "semantic_only", "temporal_semantic", "progression_only"):
        transformed, stats = transform_source_queries([query], system)
        assert stats["unique_corpora"] == 1
        assert all(doc.id in {"d1", "d2"} for doc in transformed[0].documents)
        results[system] = evaluate_qa(transformed, ExtractiveBM25Reader(), QARunConfig(char_budget=200))
    comparisons = paired_qa(results)
    assert set(comparisons) == {"temporal_only", "semantic_only", "temporal_semantic", "progression_only"}
    assert all(row["question_count"] == 1 for row in comparisons.values())
