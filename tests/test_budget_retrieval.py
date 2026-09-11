from progression_memory.evaluation.retrieval import (
    RetrievalDocument,
    RetrievalQuery,
    evaluate_retrieval,
    evaluate_retrieval_by_char_budget,
)
from progression_memory.evaluation.statistics import exact_mcnemar, paired_bootstrap_delta


def _query() -> RetrievalQuery:
    docs = (
        RetrievalDocument(
            id="d1", text="target target target", evidence_ids=frozenset({"x"})
        ),
        RetrievalDocument(id="d2", text="target", evidence_ids=frozenset({"e"})),
    )
    return RetrievalQuery(
        id="q",
        text="target",
        evidence_ids=frozenset({"e"}),
        documents=docs,
        category="test",
    )


def test_mrr_is_truncated_at_cutoff() -> None:
    result = evaluate_retrieval([_query()], [1, 2])
    assert result["cutoffs"]["1"]["mrr"] == 0.0
    assert result["cutoffs"]["2"]["mrr"] == 0.5


def test_character_budget_is_strict_ranked_prefix() -> None:
    doc = RetrievalDocument(id="e", text="target", evidence_ids=frozenset({"e"}))
    query = RetrievalQuery(
        id="q",
        text="target",
        evidence_ids=frozenset({"e"}),
        documents=(doc,),
        category="x",
    )
    result = evaluate_retrieval_by_char_budget(
        [query], [5, 6], include_per_query=True
    )
    assert result["budgets"]["5"]["macro_evidence_recall"] == 0.0
    assert result["budgets"]["5"]["avg_retrieved_chars"] == 0.0
    assert result["budgets"]["6"]["macro_evidence_recall"] == 1.0
    assert result["budgets"]["6"]["avg_retrieved_chars"] == 6.0


def test_paired_bootstrap_detects_positive_delta() -> None:
    baseline = {f"q{i}": 0.0 for i in range(20)}
    candidate = {f"q{i}": 1.0 for i in range(20)}
    result = paired_bootstrap_delta(baseline, candidate, samples=200, seed=7)
    assert result["mean_delta"] == 1.0
    assert result["ci_95"] == [1.0, 1.0]
    assert result["probability_gt_zero"] == 1.0


def test_exact_mcnemar_counts_discordant_pairs() -> None:
    baseline = {"a": False, "b": False, "c": False, "d": True}
    candidate = {"a": True, "b": True, "c": True, "d": True}
    result = exact_mcnemar(baseline, candidate)
    assert result["candidate_improved"] == 3
    assert result["candidate_regressed"] == 0
    assert result["exact_two_sided_p"] == 0.25
