from progression_memory.benchmarks.v2_project_continuity import generate
from progression_memory.benchmarks.v2_paired import run


def test_project_continuity_fixture_long_horizon_and_labeled():
    fixture = generate(distractors_per_stage=1)
    assert len(fixture.observations) == 210
    assert len(fixture.questions) == 220
    assert len({x.subject_id for x in fixture.observations}) == 10
    assert all(q.evidence_ids <= {o.id for o in fixture.observations}
               for q in fixture.questions)
    assert all("authoritative-" not in q.question for q in fixture.questions)
    assert all(x.id in fixture.proposals for x in fixture.observations)


def test_project_continuity_rag_and_v2_replay():
    fixture = generate(distractors_per_stage=1)
    result = run(fixture, [400])
    assert result["query_count"] == 220
    assert set(result["systems"]) == {"rag", "v2_state", "v2_full"}
    assert all(0 <= system["budgets"]["400"]["macro_evidence_recall"] <= 1
               for system in result["systems"].values())
