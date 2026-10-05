from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from progression_memory.benchmarks.calibrate_progression import load_pairs
from progression_memory.benchmarks.representation_rag_benchmark import evaluate, replay_decisions
from progression_memory.evaluation.representation_rag import records, retrieve
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, LinearAttachmentModel, Memory, TargetUpdate,
)
from progression_memory.maintenance.model_progression_writer import ModelProgressionWriter, construct


class TinyEncoder:
    def encode(self, texts):
        words = ("shopify", "offer", "education", "restaurant")
        return np.array([[text.casefold().count(w) for w in words] for text in texts], dtype=float)


def career_memory():
    memory = Memory()
    def add(event_id, text, decision):
        memory.apply(Event(event_id, "user", text, datetime(2026, 1, 1), event_id), decision)
    add("e1", "Started at Shopify", Decision(Action.CREATE_ROOT, identity="Career",
        scope="Work and education", state="Working", reason="Career event"))
    add("e2", "Started a Shopify internship", Decision(Action.CREATE_CHILD,
        parent_id="progression:0", identity="Shopify", scope="Employment at Shopify",
        state="Interning at Shopify", reason="Distinct employer"))
    add("e3", "Received a Shopify return offer", Decision(Action.ATTACH,
        target_ids=("progression:1",), state="Has a Shopify return offer",
        change="Internship led to an offer", new_segment=True, reason="Employer offer"))
    add("e4", "Asked about a restaurant", Decision(Action.NO_UPDATE,
        reason="No continuing career process"))
    return memory


def test_synthetic_attachment_fit_is_group_held_out():
    rows = load_pairs(Path("evaluation/progression_calibration_scenarios.json"))
    model, report = LinearAttachmentModel.train(rows)
    assert report["rows"] == 100
    assert report["scenarios"] == 25
    assert np.isfinite(model.coefficients).all()
    assert report["out_of_fold_false_attachments"] + report["out_of_fold_missed_attachments"] < 100


def test_child_state_and_unassigned_event():
    memory = career_memory()
    assert memory.progressions["progression:0"].child_ids == {"progression:1"}
    assert memory.progressions["progression:1"].current_state.evidence_ids == {"e2", "e3"}
    assert len(memory.progressions["progression:1"].segment_ids) == 2
    assert "e4" in {doc.id for doc in records(memory, "progression")}


def test_multiattach_is_atomic_on_invalid_target():
    memory = career_memory()
    with pytest.raises(KeyError):
        memory.apply(Event("e5", "user", "offer", None, "e5"),
                     Decision(Action.MULTI_ATTACH, ("progression:1", "missing"),
                              state="An offer", reason="Two processes"))
    assert "e5" not in memory.events


def test_multiattach_has_separate_states_and_single_source():
    memory = career_memory()
    memory.apply(Event("e5", "user", "Started a job search", None, "e5"),
                 Decision(Action.CREATE_CHILD, parent_id="progression:0",
                          identity="Job search", scope="New roles", state="Searching",
                          reason="Distinct job search"))
    memory.apply(Event("e6", "user", "Shopify offer", None, "e6"),
                 Decision(Action.MULTI_ATTACH, ("progression:1", "progression:2"),
                          target_updates=(
                              TargetUpdate("progression:1", "Has a Shopify offer", "Offer", True),
                              TargetUpdate("progression:2", "Comparing an offer", "Offer received", True),
                          ), reason="An offer affects work and the search"))
    assert memory.progressions["progression:1"].current_state.text != memory.progressions["progression:2"].current_state.text
    assert set(memory.events) == {"e1", "e2", "e3", "e4", "e5", "e6"}


def test_model_writer_structured_contract():
    class FakeModel:
        def generate_messages(self, messages, json_schema=None):
            assert "CURRENT" in messages[0]["content"]
            assert json_schema is not None
            return json.dumps({"action": "create_root", "target_ids": [], "parent_id": None,
                               "identity": "Career", "scope": "Work", "state": "Started working",
                               "change": "Started", "new_segment": True, "target_updates": [],
                               "reason": "Current event starts work"})

    memory = Memory()
    event = Event("e1", "user", "Started working", None, "e1")
    decision = ModelProgressionWriter(FakeModel()).decide(event, memory)
    memory.apply(event, decision)
    assert memory.progressions["progression:0"].current_state.evidence_ids == {"e1"}
    with pytest.raises(ValueError, match="unresolved subject"):
        construct((RetrievalDocument("e2", "Unknown: mentioned work"),),
                  lambda document: None, ModelProgressionWriter(FakeModel()))


def test_shared_dense_retrieval_counts_only_delivered_sources():
    memory = career_memory()
    for mode in ("append", "structure", "progression"):
        result = retrieve("Shopify offer", memory, TinyEncoder(), representation=mode,
                          search_depth=len(records(memory, mode)), context_chars=1000)
        assert set(result.source_event_ids) == set(memory.events)
        assert result.used_chars <= 1000
        assert {doc.id for doc in result.context if not doc.id.startswith("summary:")} == set(memory.events)


def test_paired_rag_uses_identical_sources_and_reader():
    memory = career_memory()
    documents = tuple(RetrievalDocument(e.id, e.text) for e in memory.events.values())
    query = RetrievalQuery("q", "Shopify offer", frozenset({"e3"}), documents,
                           "temporal", "Shopify")
    append = evaluate([query], TinyEncoder(), representation="append",
                      search_depth=len(documents), context_chars=1000)
    progression = evaluate([query], TinyEncoder(), representation="progression",
                           search_depth=len(records(memory, "progression")),
                           context_chars=1000, memory_by_corpus={id(documents): memory})
    assert append["overall"]["retrieval_recall"] == 1.0
    assert progression["overall"]["retrieval_recall"] == 1.0
    assert append["reader"] == progression["reader"]
    assert append["question_ids_sha256"] == progression["question_ids_sha256"]


def test_writer_trace_replay_rejects_source_mismatch():
    docs = (RetrievalDocument("e1", "Started working"),)
    query = RetrievalQuery("person:q0", "Where?", frozenset({"e1"}), docs,
                           "fact", "work")
    rows = [{"source_id": "e1", "subject_id": "person",
             "decision": {"action": "create_root", "identity": "Career",
                          "scope": "Work", "state": "Working", "reason": "Started work"}}]
    assert replay_decisions(query, rows).progressions["progression:0"].current_state.text == "Working"
    rows[0]["source_id"] = "wrong"
    with pytest.raises(ValueError, match="source order"):
        replay_decisions(query, rows)
