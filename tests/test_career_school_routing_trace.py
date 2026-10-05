"""Replay the full sequence and check hierarchy, subject, phase and evidence."""

import json
from collections import Counter
from pathlib import Path

from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, Memory, TargetUpdate,
)


PATH = Path("evaluation/data/progression_routing/career_school_trace_v1.json")


def test_career_school_trace_replays_coherently():
    rows = json.loads(PATH.read_text())["rows"]
    memory = Memory()
    actions = Counter()
    for row in rows:
        raw = row["decision"]
        actions[raw["action"]] += 1
        decision = Decision(action=Action(raw["action"]),
                            target_ids=tuple(raw.get("target_ids", [])),
                            parent_id=raw.get("parent_id"),
                            identity=raw.get("identity", ""), scope=raw.get("scope", ""),
                            state=raw.get("state", ""), change=raw.get("change", ""),
                            new_segment=raw.get("new_segment", False), reason=raw["reason"],
                            target_updates=tuple(TargetUpdate(**u) for u in raw.get("target_updates", [])))
        memory.apply(Event(row["source_id"], row["subject_id"], row["text"], None,
                           row["source_id"]), decision)
        assert row["source_id"] in memory.events
        if raw["action"] in ("attach", "multi_attach"):
            for target in raw["target_ids"]:
                assert row["source_id"] in memory.progressions[target].current_state.evidence_ids
    assert len(rows) == 29
    assert set(actions) == {action.value for action in Action}
    assert len(memory.progressions) == 12
    assert {p.identity for p in memory.progressions.values() if not p.parent_ids} == {"Education", "Career"}
    assert memory.progressions["progression:1"].child_ids >= {
        "progression:2", "progression:3", "progression:4", "progression:5"}
    assert memory.progressions["progression:7"].child_ids == {
        "progression:8", "progression:9", "progression:10"}
    assert memory.events["career-school:19"].subject_id == "friend"
    assert all("career-school:19" not in p.current_state.evidence_ids
               for p in memory.progressions.values())
    assert len(memory.progressions["progression:8"].segment_ids) > 1
    assert memory.progressions["progression:10"].current_state.text.startswith("Received a full-time return offer")
