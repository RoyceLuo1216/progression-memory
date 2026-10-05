import pytest

from progression_memory.benchmarks.personal_facts_dev import normalize, summarize


def example(**changes):
    row = dict(context="I am studying engineering.", text="I study engineering.",
               main_category="Routine activities", time="Present", referent="Self",
               duration="Long-term", broken="No", broken_reason="None", followup="None")
    return {**row, **changes}


def test_filter_is_only_a_candidate():
    assert normalize(example())["durable_self_candidate"]
    assert not normalize(example(referent="Other"))["durable_self_candidate"]
    assert not normalize(example(broken="Yes"))["durable_self_candidate"]
    assert not normalize(example(duration="Short-term"))["durable_self_candidate"]
    result = summarize([example(), example(referent="Other")], "train", "sha")
    assert result["rows"] == 2 and result["durable_self_candidates"] == 1


def test_schema_drift_fails():
    with pytest.raises(ValueError, match="missing fields"):
        normalize({"context": "hi"})
