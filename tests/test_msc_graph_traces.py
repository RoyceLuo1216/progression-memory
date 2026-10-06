"""Validate provisional full-conversation MSC graph annotations without raw text."""

import json
import os
from collections import Counter
from pathlib import Path

from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, Memory, TargetUpdate,
)


PATH = Path("evaluation/data/progression_routing/msc_graph_traces_v1.json")


def test_full_sequence_graph_traces_replay():
    data = json.loads(PATH.read_text())
    assert data["annotation_status"].startswith("single-model provisional")
    assert len(data["traces"]) == 2
    audit_path = os.getenv("MSC_DEV_AUDIT_PATH")
    audit = json.loads(Path(audit_path).read_text()) if audit_path else None
    source = {c["initial_data_id"]: c for c in audit["conversations"]} if audit else {}
    actions = Counter()
    for trace in data["traces"]:
        memory = Memory()
        conversation_id = trace["conversation_id"]
        assert len(trace["rows"]) == 24
        turns = source.get(conversation_id, {}).get("turns", [])
        lookup = {f"s{t['session']}:t{t['turn']}": t for t in turns}
        if audit:
            assert set(lookup) == {r["ref"] for r in trace["rows"]}
        for row in trace["rows"]:
            ref = row["ref"]
            t = lookup.get(ref)
            evidence = row["evidence"]
            assert evidence
            if row["action"] == "no_update":
                assert "state" not in row
            subject = f"{conversation_id}:{row['subject']}"
            source_id = t["source_id"] if t else f"msc:{conversation_id}:{ref}"
            if t and row["action"] != "no_update" and row["subject"] != int(t["speaker_id"].rsplit(":", 1)[-1]):
                assert (conversation_id, ref) == ("train:ordered_4998", "s2:t2")
            targets = tuple(f"progression:{i}" for i in row.get("targets", []))
            updates = tuple(TargetUpdate(f"progression:{u['target']}", u["state"], evidence,
                                         u.get("segment", False)) for u in row.get("updates", []))
            decision = Decision(
                action=Action(row["action"]), target_ids=targets,
                identity=row.get("identity", ""), scope=row.get("scope", ""),
                state=row.get("state", ""), change=evidence,
                new_segment=row.get("segment", False), reason=evidence,
                target_updates=updates,
            )
            memory.apply(Event(source_id, subject, t["text"] if t else evidence, None, source_id), decision)
            actions[row["action"]] += 1
        assert len(memory.events) == len(trace["rows"])
        assert all(p.current_state.evidence_ids for p in memory.progressions.values())
    assert actions["no_update"] and actions["create_root"] and actions["attach"] and actions["multi_attach"]
