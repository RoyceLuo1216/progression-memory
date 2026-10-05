"""Contract checks on the checked-in real dialogue annotation snapshot."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from progression_memory.benchmarks.calibrate_progression import load_real_pairs


def test_msc_pair_snapshot():
    path = Path("evaluation/data/progression_attachment/msc_train_pairs_v1.json")
    data = json.loads(path.read_text())
    pairs = data["pairs"]
    assert len(pairs) == 100
    counts = Counter(row["scenario_id"] for row in pairs)
    assert len(counts) == 25 and set(counts.values()) == {4}
    assert {row["attach"] for row in pairs} == {0, 1}
    assert all(row["event_text"] and row["candidate_anchor_text"]
               and row["candidate_state"] for row in pairs)
    assert all(not row["attach"] or row["event_speaker_id"] == row["candidate_subject_id"]
               for row in pairs)
    by_event = defaultdict(set)
    for row in pairs:
        by_event[row["scenario_id"]].add(row["event_source_id"])
        assert row["event_source_id"].split(":s")[0] == row["candidate_anchor_source_id"].split(":s")[0]
        def order(source_id):
            session, turn = source_id.rsplit(":s", 1)[1].split(":t")
            return int(session), int(turn)
        assert order(row["candidate_anchor_source_id"]) < order(row["event_source_id"])
    assert all(len(ids) == 1 for ids in by_event.values())
    model_rows = load_real_pairs(path)
    assert len(model_rows) == 100
    assert len({row["scenario"] for row in model_rows}) == 6
