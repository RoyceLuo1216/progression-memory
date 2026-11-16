"""No external model download required for evaluation-policy tests."""
from progression_memory.benchmarks.meme_final_hybrid import (
    wants_history, budget_pack, to_compact, candidates, fusion,
)
from progression_memory.evaluation.retrieval import RetrievalDocument


def test_history_only_when_question_asks_for_temporal_sequence():
    assert wants_history("List all vehicles I've had over time, from earliest to latest.")
    assert wants_history("What was my previous career goal?")
    assert not wants_history("What medication am I taking?")
    assert not wants_history("Summarize my current career goal")


def test_budget_skip_overlarge_unit_and_fill_remaining():
    docs=[RetrievalDocument("large","0123456789"),
          RetrievalDocument("small","short")]
    selected,total=budget_pack(docs,6)
    assert [d.id for d in selected]==["small"]
    assert total==5


def test_compact_v2_state_uses_only_current_when_not_requested():
    raw=RetrievalDocument(id="v2:state:p",text=(
        '{"progression":"Vehicle","direct_state":{'
        '"value":"scooter","history":["car","bike","scooter"]}}'))
    current=to_compact([raw],include_history=False)[0]
    historical=to_compact([raw],include_history=True)[0]
    assert "scooter" in current.text
    assert "car" not in current.text
    assert "car" in historical.text
    assert len(current.text)<len(historical.text)


def test_wiki_progression_hybrid_preserves_evidence_rescue():
    current=RetrievalDocument(id="wiki:Vehicle",
                              text="Topic: Vehicle. Current value: scooter.",
                              metadata={"kind":"wiki"})
    history=RetrievalDocument(id="wiki:Vehicle",
                              text="Topic: Vehicle. Current value: scooter. History: car; scooter.",
                              metadata={"kind":"wiki"})
    raw=RetrievalDocument(id="s0:t0",text="User switched to scooter")
    groups={"raw":[raw],"wiki_current":[current],"wiki_history":[history],
            "v2_current":[],"v2_history":[],"compact_current":[],
            "compact_history":[]}
    temporal=candidates(groups,"What did I own before?",400)
    ordinary=candidates(groups,"What vehicle do I own?",400)
    assert temporal["wiki_v2_hybrid_rag"][0].id=="wiki:Vehicle"
    assert ordinary["wiki_v2_hybrid_rag"][-1].id=="s0:t0"
    assert "History:" in temporal["wiki_v2_adaptive"][0].text
    assert "History:" not in ordinary["wiki_v2_adaptive"][0].text
