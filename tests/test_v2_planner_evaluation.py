from progression_memory.benchmarks.v2_planner_evaluation import (
    SYSTEMS,
    cluster_ci,
    lexical_semantic_rank,
    personas_and_checkpoints,
    score_output,
)
from progression_memory.evaluation.retrieval import RetrievalDocument


def test_four_arms_and_checkpoint_fixture():
    fixture, checkpoints = personas_and_checkpoints(distractors=1)
    assert len(SYSTEMS) == 4
    assert len(fixture.observations) == 150
    assert len(checkpoints) == 42


def test_scoring_missing_and_correct_values():
    gold = {"long_term_goal": "startup building",
            "priority": "financial runway", "constraint": "tuition is high"}
    correct = score_output({
        "goal": gold["long_term_goal"],
        "priority": gold["priority"],
        "constraint": gold["constraint"],
        "tradeoffs": ["risk", "learning"],
        "next_steps": ["compare", "interview", "decide"]}, gold)
    assert correct["goal_fidelity"] == 1
    assert correct["constraint_field_mismatch"] == 0
    wrong = score_output({}, gold)
    assert wrong["priority_field_mismatch"] == 1


def test_retriever_and_persona_cluster_ci():
    docs = [RetrievalDocument(id="a", text="startup founder building products"),
            RetrievalDocument(id="b", text="graduate study tuition")]
    ranked = lexical_semantic_rank(docs, "founder startup")
    assert ranked[0].id == "a"
    rows = []
    for persona in ("p1", "p2", "p3"):
        for system, value in (("v2_oracle", 1), ("hybrid_rag_summary", 0)):
            rows.append({"persona": persona, "id": "checkpoint",
                         "system": system, "scores": {"goal_fidelity": value}})
    ci = cluster_ci(rows, "v2_oracle", "hybrid_rag_summary",
                    "goal_fidelity", samples=100)
    assert ci["cluster_count"] == 3
    assert ci["difference"] == 1
    assert ci["ci95"] == [1, 1]
