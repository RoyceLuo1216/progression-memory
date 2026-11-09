from progression_memory.benchmarks.v2_gold_state import evaluate
from progression_memory.benchmarks.v2_long_horizon_dataset import generate


def test_independent_oracle_state_and_visibility_metrics():
    report = evaluate(generate(events_per_stream=36), budgets=(400,))
    assert report["state"]["total"] > 40
    assert report["state"]["exact_state_accuracy"] == 1.0
    assert report["dataset"]["observations"] == 149
    assert report["dataset"]["horizon_days"] > 300
    assert report["writer"]["status"] == "not_measured_live_model"
    assert report["llm_answer_accuracy"]["status"] == "not_measured"
    assert set(report["literal_answerability"]) == {
        "rag", "v2_flat", "v2_hierarchical",
    }
