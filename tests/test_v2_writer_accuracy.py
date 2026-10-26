import pytest

from progression_memory.benchmarks.v2_writer_accuracy import Case, LabeledTrace, evaluate


def test_writer_accuracy_counts_false_attachments_and_actions():
    trace = LabeledTrace(cases=[
        Case(observation_id="a", gold_targets=["job"],
             predicted_targets=["job"], gold_actions=["transition"],
             predicted_actions=["append"]),
        Case(observation_id="b", gold_targets=["course"],
             predicted_targets=["hobby"], gold_actions=["close"],
             predicted_actions=["close"], unsupported_patches=1),
        Case(observation_id="c"),
    ])
    result = evaluate(trace)
    assert result["attachment_precision"] == 0.5
    assert result["attachment_recall"] == 0.5
    assert result["target_exact_rate"] == pytest.approx(2 / 3)
    assert result["action_exact_rate"] == pytest.approx(1 / 3)
    assert result["unsupported_patch_count"] == 1


def test_duplicate_observation_and_duplicate_targets_rejected():
    with pytest.raises(ValueError, match="duplicate observation"):
        evaluate(LabeledTrace(cases=[Case(observation_id="a"), Case(observation_id="a")]))
    with pytest.raises(ValueError, match="duplicate target"):
        evaluate(LabeledTrace(cases=[
            Case(observation_id="a", gold_targets=["job", "job"],
                 gold_actions=["append", "append"]),
        ]))
