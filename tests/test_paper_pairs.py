from __future__ import annotations

import pytest

from progression_memory.benchmarks.paper_pairs import compare, specification


def _result(system: str, answers: dict[str, bool]) -> dict:
    paper = system.split("+")[0]
    pinned = specification()["papers"].get(paper, {}).get("revision", "combined-sha")
    return {
        "system": system, "dataset": "locomo", "paper_revision": pinned,
        "implementation_revision": "modified-sha" if "+progression" in system else pinned,
        "dataset_sha256": "dataset-sha", "dataset_split": "all",
        "llm_model": "fixed-llm", "embedding_model": "fixed-embedder",
        "reader_model": "fixed-reader", "retrieval_budget": "20 chunks / 2400 characters",
        "evaluator_revision": "official-sha", "native_run_manifest": "runs/manifest.json",
        "official_evaluator_output": "runs/evaluator.json", "question_results": answers,
    }


def test_paper_revisions_and_declared_pairs():
    spec = specification()
    assert all(len(item["revision"]) == 40 for item in spec["papers"].values())
    assert spec["pairs"] == [
        ["timem", "timem+progression"], ["tsm", "tsm+progression"],
        ["timem+tsm", "timem+tsm+progression"],
    ]


def test_comparison_counts_improvement_and_regression():
    result = compare(_result("tsm", {"a": False, "b": True, "c": False}),
                     _result("tsm+progression", {"a": True, "b": False, "c": True}))
    assert result["delta_accuracy"] == pytest.approx(1 / 3)
    assert result["improved_ids"] == ["a", "c"]
    assert result["regressed_ids"] == ["b"]


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(llm_model="another-llm"),
    lambda row: row.update(dataset_sha256="another-dataset"),
    lambda row: row.update(retrieval_budget="unbounded"),
    lambda row: row.update(question_results={"different": True}),
    lambda row: row.update(native_run_manifest=""),
    lambda row: row.update(official_evaluator_output=""),
    lambda row: row.update(question_results={"a": 1}),
    lambda row: row.update(implementation_revision=specification()["papers"]["timem"]["revision"]),
])
def test_rejects_invalid_pairs(mutation):
    baseline = _result("timem", {"a": False})
    altered = _result("timem+progression", {"a": True})
    mutation(altered)
    with pytest.raises(ValueError):
        compare(baseline, altered)


def test_rejects_our_component_ablation_as_paper_baseline():
    with pytest.raises(ValueError, match="undeclared"):
        compare(_result("temporal_only", {"a": False}),
                _result("timem+progression", {"a": True}))
