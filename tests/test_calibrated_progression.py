from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from progression_memory.benchmarks.calibrate_progression import load_pairs
from progression_memory.evaluation.representation_rag import records, retrieve
from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, LinearAttachmentModel, Memory,
)


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


def test_shared_dense_retrieval_counts_only_delivered_sources():
    memory = career_memory()
    for mode in ("append", "structure", "progression"):
        result = retrieve("Shopify offer", memory, TinyEncoder(), representation=mode,
                          search_depth=len(records(memory, mode)), context_chars=1000)
        assert set(result.source_event_ids) == set(memory.events)
        assert result.used_chars <= 1000
        assert {doc.id for doc in result.context if not doc.id.startswith("summary:")} == set(memory.events)
