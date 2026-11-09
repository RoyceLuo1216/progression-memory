from progression_memory.benchmarks.v2_decision_continuity import make_fixture, score


def test_long_horizon_personas_and_scoring():
    fixture, checkpoints = make_fixture(distractors=1)
    assert len(checkpoints) == 42
    assert len(fixture.observations) == 6 * (1 + 12 * 2)
    report = score(distractors=1, budgets=(400,))
    assert report["dataset"]["checkpoints"] == 42
    assert report["dataset"]["personas"] == 6
    assert 0 <= report["systems"]["rag_bm25"]["400"]["field_visibility_accuracy"] <= 1
    assert report["systems"]["v2_oracle_state"]["400"]["field_visibility_accuracy"] == 1.0
