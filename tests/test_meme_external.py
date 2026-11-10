from progression_memory.benchmarks.meme_evidence_retrieval import (
    task_labelled_questions, gold_evidence, ranked_recency, prefix
)
from progression_memory.benchmarks.meme_v2_oracle_wiki import (
    target_atoms, wiki_docs, oracle_engine
)
from progression_memory.v2.read_path import build_documents
from progression_memory.evaluation.retrieval import RetrievalDocument
from datetime import UTC, datetime, timedelta


def test_question_time_cutoff_is_honored():
    ep={"sessions":[{},{}], "before_questions":{
        "position_after_session":1,
        "questions":[{"task_type":"Cas","question":"Which preference?"}]}}
    qs=list(task_labelled_questions(ep))
    assert len(qs)==1 and qs[0][1]==1


def test_gold_evidence_is_scoring_only():
    ep={"dependency_edges_used":[]}
    facts={"career":[(0,"s0:t2"),(1,"s1:t4")]}
    q={"entity":["career"],"task_type":"Cas"}
    assert gold_evidence(ep,facts,q,1)[0]=={"s0:t2"}
    assert gold_evidence(ep,facts,q,2)[0]=={"s1:t4"}
    q["task_type"]="Tr"
    assert gold_evidence(ep,facts,q,2)[0]=={"s0:t2","s1:t4"}


def test_value_atoms_and_wiki_history():
    q={"entity_values":{"vehicle":["car","bike"],"goal":"research"}}
    assert target_atoms(q)==["car","bike","research"]
    facts=[
        {"session":0,"entity":"vehicle","value":"car","text":"I have a car"},
        {"session":1,"entity":"vehicle","value":"bike","text":"Now I have a bike"}
    ]
    current=wiki_docs(facts,2,False)
    history=wiki_docs(facts,2,True)
    assert "car" not in current[0].text
    assert "car" in history[0].text
    assert "bike" in history[0].text


def test_actual_state_engine_tracks_temporal_revisions():
    facts=[
        {"session":0,"entity":"goal","value":"school","text":"I plan school"},
        {"session":1,"entity":"goal","value":"job","text":"I now plan a job"},
    ]
    engine=oracle_engine(facts,"test-episode")
    earlier=datetime(2023,1,1,tzinfo=UTC)+timedelta(days=1)
    later=datetime(2023,1,1,tzinfo=UTC)+timedelta(days=2)
    before=build_documents(engine,subject_id="meme:test-episode",
                           view="state",valid_at=earlier,known_at=earlier)
    after=build_documents(engine,subject_id="meme:test-episode",
                           view="state",valid_at=later,known_at=later)
    assert "school" in before[0].text
    assert "job" not in before[0].text
    assert "job" in after[0].text


def test_recency_prefix_budget():
    docs=[RetrievalDocument(id="old",text="older"),RetrievalDocument(id="new",text="newer")]
    ranked=ranked_recency(docs)
    chosen,used=prefix(ranked,5)
    assert chosen[0].id=="new" and used==5
