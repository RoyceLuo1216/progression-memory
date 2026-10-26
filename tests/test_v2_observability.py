import pytest

from progression_memory.benchmarks.v2_observability import compile_report, markdown_report


def test_compares_matching_artifacts_without_inventing_unmeasured_scores():
    base = {"input_sha256": "same", "observations": 197, "questions": 65}
    paired = {**base, "systems": {"rag": {"budgets": {
        "400": {"eligible_query_count": 65, "macro_evidence_recall": 0.8,
                "mrr": 0.7, "avg_retrieved_chars": 300.0},
    }}}}
    ablation = {"input_sha256": "same", "systems": {
        "flat_bm25": {"status": "completed", "budgets": {
            "400": {"eligible_query_count": 65, "macro_evidence_recall": 0.9,
                    "mrr": 0.8, "avg_retrieved_chars": 310.0},
        }},
        "flat_hybrid": {"status": "skipped", "reason": "embedding not configured"},
    }}
    result = compile_report(paired, ablation)
    assert result["question_count"] == 65
    assert len(result["rows"]) == 3
    assert result["measurement_status"]["writer_live_model_accuracy"] == "not_measured"
    assert result["rows"][-1]["status"] == "skipped"
    assert "0.90" not in markdown_report(result)  # report uses percentage formatting
    assert "90.00%" in markdown_report(result)


def test_refuses_cross_fixture_comparison():
    with pytest.raises(ValueError, match="different fixtures"):
        compile_report(
            {"input_sha256": "a", "systems": {}},
            {"input_sha256": "b", "systems": {}},
        )
