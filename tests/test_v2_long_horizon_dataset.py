"""Smoke-test the complete long-horizon fixture against actual benchmark runners."""
from progression_memory.benchmarks.v2_long_horizon_dataset import generate
from progression_memory.benchmarks.v2_paired import run as run_paired
from progression_memory.benchmarks.v2_retrieval_ablation import evaluate


def test_year_long_single_domain_fixture_is_valid():
    fixture = generate()
    assert len(fixture.observations) == 197
    assert len(fixture.questions) == 65
    assert len({o.id for o in fixture.observations}) == len(fixture.observations)
    assert len({o.subject_id for o in fixture.observations}) == 1
    assert (fixture.observations[-1].recorded_at -
            fixture.observations[0].recorded_at).days >= 365
    observed = {o.id for o in fixture.observations}
    assert all(q.evidence_ids <= observed for q in fixture.questions)
    assert all(obs.id in fixture.proposals for obs in fixture.observations)


def test_long_horizon_paired_and_hierarchical_smoke():
    # Smaller still sufficiently longitudinal fixture to keep CI economical.
    fixture = generate(events_per_stream=36)
    paired = run_paired(fixture, [400, 1600])
    assert paired["observations"] == 149
    assert paired["questions"] == 49
    assert set(paired["systems"]) == {"rag", "v2_state", "v2_full"}
    variants = evaluate(
        fixture, [400, 1600],
        modes=("flat_bm25", "hierarchical_bm25"),
    )
    assert all(v["status"] == "completed" for v in variants["systems"].values())
    assert all(v["budgets"]["1600"]["eligible_query_count"] == 49
               for v in variants["systems"].values())
