from progression_memory.benchmarks.v2_strict_visibility import evaluate


def test_strict_visibility_excludes_progression_only_provenance():
    report = evaluate(distractors=1, budgets=(400, 1600))
    assert report["dataset"]["questions"] == 220
    for system in report["systems"].values():
        assert 0 <= system["400"]["visible_evidence_recall"] <= 1
        assert 0 <= system["1600"]["full_support_rate"] <= 1
    assert report["systems"]["rag"]["400"]["questions"] == 220
