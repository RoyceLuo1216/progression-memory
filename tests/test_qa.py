import json

from progression_memory.evaluation.qa import (
    ExtractiveBM25Reader,
    QARunConfig,
    answer_contains,
    evaluate_qa,
    exact_match,
    token_f1,
    write_longmemeval_hypotheses,
)
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery


def _query() -> RetrievalQuery:
    documents = (
        RetrievalDocument(
            id="d1",
            text="user: I adopted a beagle named Scout.",
            evidence_ids=frozenset({"d1"}),
        ),
        RetrievalDocument(
            id="d2",
            text="assistant: The weather is sunny today.",
            evidence_ids=frozenset({"d2"}),
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
