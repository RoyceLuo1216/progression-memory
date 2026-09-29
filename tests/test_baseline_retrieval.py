from datetime import date, timedelta

import numpy as np

from progression_memory.benchmarks.baseline_retrieval import evaluate_system, paired_budget_comparison
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.evaluation.temporal_semantic_baselines import build_baseline_documents


class SimpleEmbedder:
    model_name = "test-embeddings"

    def encode(self, texts):
        return np.asarray([
            [1.0, 0.0] if "pottery" in text else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


def test_progression_uses_shared_encoder_and_emits_matched_source_outcomes():
    docs = (
        RetrievalDocument("a", "pottery cup", timestamp="2026-01-01"),
        RetrievalDocument("b", "new job", timestamp="2026-01-02"),
        RetrievalDocument("c", "pottery bowl", timestamp="2026-01-03"),
    )
    query = RetrievalQuery("q", "pottery bowl", frozenset({"c"}), docs, "test")
    embedder = SimpleEmbedder()
    baseline = evaluate_system([query], "temporal_semantic", cutoffs=(2,), budgets=(100,), embedder=embedder)
    progression = evaluate_system([query], "progression_only", cutoffs=(2,), budgets=(100,), embedder=embedder)
    assert progression["semantic_backend"] == "test-embeddings"
    pair = paired_budget_comparison({"temporal_semantic": baseline, "progression_only": progression},
                                    "temporal_semantic", "progression_only", 100)
    assert pair["questions_with_evidence"] == 1
    assert pair["delta_macro_recall"] == (pair["per_question_delta"]["q"])


def test_temporal_group_can_be_large_but_source_top_k_does_not_get_group_credit():
    docs = tuple(RetrievalDocument(
        id=f"source-{i}", text=f"pottery note {i}",
        evidence_ids=frozenset({f"source-{i}"}),
        timestamp=str(date(2026, 1, 1) + timedelta(days=i)),
    ) for i in range(30))
    query = RetrievalQuery("q", "pottery note", frozenset({"source-29"}), docs, "test")
    grouped, _ = build_baseline_documents(docs, "temporal_only")
    assert len(grouped) == 1
    result = evaluate_system([query], "temporal_only", cutoffs=(8,), budgets=(60,))
    assert result["corpus_stats"]["avg_stage1_evidence_coverage"] == 1.0
    assert result["source_top_k"]["cutoffs"]["8"]["macro_evidence_recall"] == 0.0
    assert result["source_char_budget"]["budgets"]["60"]["macro_evidence_recall"] == 0.0


def test_semantic_only_can_rejoin_theme_across_unrelated_turn():
    docs = (
        RetrievalDocument("a", "pottery cup", timestamp="2026-01-01"),
        RetrievalDocument("b", "new job", timestamp="2026-01-01"),
        RetrievalDocument("c", "pottery bowl", timestamp="2026-02-01"),
    )
    semantic, _ = build_baseline_documents(docs, "semantic_only", embedder=SimpleEmbedder())
    combined, _ = build_baseline_documents(docs, "temporal_semantic", embedder=SimpleEmbedder())
    assert [unit.evidence_ids for unit in semantic] == [frozenset({"a", "c"}), frozenset({"b"})]
    assert len(combined) == 3
