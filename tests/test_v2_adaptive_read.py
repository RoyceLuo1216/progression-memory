from datetime import UTC, datetime

import pytest

from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.v2.adaptive_read import (
    asks_for_history, compact_progression, pack_skip_oversized, read_adaptive,
)
from progression_memory.v2.models import AttributeOperation, AttributePatch, Observation
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import (
    ProposedUpdate, ScriptedWriter, WritePipeline, WriterProposal,
)


def engine():
    state = StateEngine()
    start=datetime(2025,1,1,tzinfo=UTC)
    later=datetime(2025,2,1,tzinfo=UTC)
    a=Observation(id="first",source_id="first",subject_id="user",
                  text="I own a car",recorded_at=start)
    b=Observation(id="second",source_id="second",subject_id="user",
                  text="I switched to a scooter",recorded_at=later)
    proposals={
        "first":WriterProposal(updates=(ProposedUpdate(
            action="create",name="Vehicle",reason="first vehicle statement",
            patches=(AttributePatch(key="value",operation=AttributeOperation.SET,value="car"),
                     AttributePatch(key="history",operation=AttributeOperation.SET,value=["car"])),
        ),)),
        "second":WriterProposal(updates=(ProposedUpdate(
            action="transition",progression_id="v2:first:0",
            reason="explicit vehicle change",
            patches=(AttributePatch(key="value",operation=AttributeOperation.SET,value="scooter"),
                     AttributePatch(key="history",operation=AttributeOperation.SET,
                                    value=["car","scooter"])),
        ),)),
    }
    writer=WritePipeline(state,ScriptedWriter(proposals))
    writer.process(a);writer.process(b)
    return state,start,later


def test_history_routing_and_compact_current_state():
    state,_,later=engine()
    current=read_adaptive(state,"What vehicle do I own?",
        subject_id="user",budget_chars=500,valid_at=later,known_at=later)
    history=read_adaptive(state,"What did I own before?",
        subject_id="user",budget_chars=500,valid_at=later,known_at=later)
    assert current.includes_history is False
    assert "scooter" in current.text and "car;" not in current.text
    assert history.includes_history is True
    assert "car; scooter" in history.text


def test_historical_cutoff_excludes_future_state_and_source():
    state,early,_=engine()
    snap=read_adaptive(state,"What did I own before?",
        subject_id="user",budget_chars=500,valid_at=early,known_at=early,
        source_rescue=True)
    assert "car" in snap.text
    assert "scooter" not in snap.text


def test_one_representation_per_topic_and_wiki_fallback():
    state,_,later=engine()
    wiki_now=[
        RetrievalDocument(id="wiki:Vehicle",text="Topic: Vehicle. Current value: scooter."),
        RetrievalDocument(id="wiki:Career",text="Topic: Career. Current value: engineer."),
    ]
    wiki_past=[
        RetrievalDocument(id="wiki:Vehicle",text="Topic: Vehicle. Historical values in order: car; scooter."),
        RetrievalDocument(id="wiki:Career",text="Topic: Career. Historical values in order: intern; engineer."),
    ]
    now=read_adaptive(state,"What is my current status?",subject_id="user",
                      wiki_current=wiki_now,budget_chars=600,valid_at=later,known_at=later)
    historical=read_adaptive(state,"How has my history changed?",subject_id="user",
                   wiki_history=wiki_past,budget_chars=600,valid_at=later,known_at=later)
    assert now.text.count("Topic: Vehicle.")==1
    assert "engineer" in now.text
    assert historical.text.count("Topic: Vehicle.")==1
    assert "intern; engineer" in historical.text


def test_budget_validation_and_ranker_cannot_inject_future_docs():
    state,_,later=engine()
    with pytest.raises(ValueError,match="nonnegative"):
        read_adaptive(state,"current",subject_id="user",budget_chars=-1)
    def malicious(_query,_docs):
        return [RetrievalDocument(id="future",text="future secret")]
    with pytest.raises(ValueError,match="outside"):
        read_adaptive(state,"current",subject_id="user",valid_at=later,
                      ranker=malicious)
    chosen,used=pack_skip_oversized([
        RetrievalDocument(id="big",text="0123456789"),
        RetrievalDocument(id="short",text="small"),
    ],5)
    assert [d.id for d in chosen]==["short"] and used==5
